import asyncio
import logging
from typing import List

from fastapi import FastAPI, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
# 引入 FastAPI 的 CORS 中介軟體，解決跨來源請求問題
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from cba import models
from cba.crud import save_weather_forecasts
from cba.database import SessionLocal, engine
from cba.etl.fetcher import CWAETLFetcher
from cba.etl.transformer import transform_forecast_data
# ✨ 1. 從 rag_handler 引入我們最終的 RAGHandlerV3 ✨
from cba.rag_handler import RAGHandlerV5

# --- 基本設定 ---
# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 建立所有資料庫表格
models.Base.metadata.create_all(bind=engine)

# 初始化 FastAPI 應用
app = FastAPI(
    title="天氣 RAG 聊天機器人 API",
    description="一個基於模組化 RAG 架構 (V3) 的高級天氣問答系統。",
    version="3.0.0",
)

# --- ✨ 2. 設定 CORS 中介軟體 ✨ ---
# 允許所有來源，這在開發階段非常方便，可以讓您的 index.html 順利存取 API
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ✨ 3. 初始化 RAG 處理器 ✨ ---
# 在應用程式啟動時，就將 V3 處理器實例化並載入所有模型。
# 這樣可以避免每次使用者請求時都重新載入模型，大幅提升回應速度。
logging.info("正在初始化 RAG Handler V4...")
rag_handler = RAGHandlerV5()
logging.info("RAG Handler V4 初始化完成！")


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色，'user' 或 'bot'")
    content: str = Field(..., description="訊息內容")


class ChatRequest(BaseModel):
    query: str = Field(..., description="使用者最新的問題")
    chat_history: List[ChatMessage] = Field([], description="過去的對話歷史")


class ChatResponse(BaseModel):
    answer: str
    chat_history: List[ChatMessage]


# --- 資料庫依賴注入 ---
def get_db():
    """為每個請求建立一個獨立的資料庫 session。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- API 端點定義 ---

@app.get("/", summary="根目錄", description="顯示歡迎訊息，確認服務已啟動。")
async def root():
    """
    根目錄端點，返回一個歡迎訊息。
    可以用來檢查服務是否正常運行。
    """
    return {"message": "歡迎使用天氣 RAG API (V3)。服務已啟動！"}


@app.get("/etl-update", summary="執行 ETL 流程", description="從 CWA 抓取最新的天氣預報，並存入資料庫。")
async def etl_update(db: Session = Depends(get_db)):
    """
    執行完整的 ETL (Extract, Transform, Load) 流程。
    1. 從中央氣象署 API 提取資料。
    2. 轉換資料格式並進行正規化 (如「臺」轉「台」)。
    3. 將處理後的資料載入到 SQLite 資料庫中。
    """
    logging.info("開始執行 ETL 流程...")
    try:
        cwa_fetcher = CWAETLFetcher()
        # 這是一筆測試用的資料集，您也可以替換為完整的鄉鎮ID列表
        locational_ids_list = ["F-D0047-" + str(num).zfill(3) for num in list(range(1, 92, 2))]
        raw_data = []
        # 分批次獲取資料，避免請求過長
        for i in range(0, len(locational_ids_list), 5):
            locational_ids = locational_ids_list[i:i + 5]
            locational_ids_str = ",".join(locational_ids)
            raw_data.append(cwa_fetcher.fetch_data("F-D0047-093", locational_ids_str))

        logging.info("資料提取完成，開始轉換...")
        data_list = transform_forecast_data(raw_data)

        if data_list:
            logging.info(f"資料轉換完成，共 {len(data_list)} 筆記錄。開始載入資料庫...")
            inserted_count = save_weather_forecasts(db, data_list)
            logging.info(f"ETL 流程成功完成，插入/更新了 {inserted_count} 筆記錄。")
            return {"message": f"ETL 流程成功完成", "inserted_records": inserted_count}
        else:
            logging.warning("ETL 流程完成，但沒有處理任何資料。")
            return {"message": "ETL 流程完成，但沒有資料可處理。"}
    except Exception as e:
        logging.error(f"ETL 流程失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ETL 流程失敗: {str(e)}")


@app.get("/rag-setup", summary="建立/更新 RAG 索引", description="從資料庫讀取資料，並為 RAG 系統建立向量索引。")
async def rag_setup(db: Session = Depends(get_db)):
    """
    為 RAG 系統建立或更新向量索引。
    在執行 ETL 後，應呼叫此端點以確保 RAG 使用最新的資料。
    """
    logging.info("開始建立 RAG 索引...")
    try:
        count = rag_handler.setup_vector_store(db)
        logging.info(f"RAG 索引建立完成，共索引了 {count} 份父文檔。")
        return {"message": "RAG 索引建立成功", "indexed_parent_documents": count}
    except Exception as e:
        logging.error(f"RAG 索引建立失敗: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG 索引建立失敗: {str(e)}")


@app.get("/ask-weather", response_model=ChatResponse, summary="天氣對話", description="進行具備記憶功能的多輪天氣對話。")
async def ask_weather(request: ChatRequest):
    """
    天氣對話的核心端點 (V4)。
    - **request**: 包含使用者最新問題和對話歷史的請求主體。
    """
    logging.info(f"收到天氣對話請求: '{request.query}'")
    try:
        # 直接將請求中的 query 和 chat_history 傳給 V4 處理器
        result = rag_handler.ask(request.query, [msg.dict() for msg in request.chat_history])
        return result
    except Exception as e:
        logging.error(f"回答問題時出錯，查詢: '{request.query}', 錯誤: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"處理您的請求時發生內部錯誤: {str(e)}")


@app.get("/debug-rag-retrieval", summary="[偵錯] 檢視 RAG 檢索結果", include_in_schema=False)
async def debug_rag_retrieval(query: str):
    """
    這個端點不經過 LLM 生成答案，而是直接回傳檢索鏈(Retriever)的結果。
    這可以幫助我們診斷檢索器是否找到了正確的文檔。
    """
    if not query:
        raise HTTPException(status_code=400, detail="查詢參數 'query' 不可為空。")

    logging.info(f"【偵錯模式】開始檢索，查詢: '{query}'")

    try:
        # 1. 直接呼叫 RAG Handler 中的最終檢索器
        #    這會觸發 MultiQuery -> Re-ranking -> ParentDocument 的完整檢索流程
        retrieved_docs = rag_handler.final_retriever.invoke(query)

        logging.info(f"【偵錯模式】檢索完成，找到了 {len(retrieved_docs)} 份文檔。")

        # 2. 將找到的文檔內容和元數據格式化後回傳
        formatted_docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in retrieved_docs
        ]

        return {
            "query": query,
            "retrieved_documents_count": len(retrieved_docs),
            "retrieved_documents": formatted_docs
        }

    except Exception as e:
        logging.error(f"【偵錯模式】檢索時出錯: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"偵錯檢索時發生錯誤: {str(e)}")


@app.post("/stream-ask-weather", summary="天氣對話 (流式 V5)", description="進行具備路由和記憶功能的流式多輪天氣對話。")
async def stream_ask_weather(request: ChatRequest):
    """
    天氣對話的流式核心端點 (V5)。
    """
    langchain_chat_history = []
    # ✨ 關鍵修正：使用 . (點) 來存取 Pydantic 物件的屬性，而不是 [] ✨
    for msg in request.chat_history:
        if msg.role == 'user':
            langchain_chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == 'bot':
            langchain_chat_history.append(AIMessage(content=msg.content))

    gen = stream_generator(request.query, langchain_chat_history)

    return StreamingResponse(gen, media_type="text/plain; charset=utf-8")


# stream_generator 函式本身是正確的，無需修改
async def stream_generator(query: str, chat_history: list):
    try:
        async for chunk in rag_handler.full_router_chain.astream({
            "chat_history": chat_history,
            "input": query
        }):
            if isinstance(chunk, dict) and "answer" in chunk and chunk["answer"]:
                answer_piece = chunk["answer"]
                yield answer_piece.encode("utf-8")
            elif isinstance(chunk, str) and chunk:
                answer_piece = chunk
                yield answer_piece.encode("utf-8")
            await asyncio.sleep(0.001)
    except Exception as e:
        logging.error(f"流式生成時出錯: {e}", exc_info=True)
        error_message = f"\n\n處理時發生錯誤: {e}"
        yield error_message.encode("utf-8")
