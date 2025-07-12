# --- START OF FILE cba/rag_handler.py ---
from typing import List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from . import crud
from .models import WeatherForecast

# --- 全域設定 ---
# 模型路徑，請根據你下載的位置修改
MODEL_PATH = "./Taiwan-LLM-13B-v2.0-chat-Q5_1.gguf"
# Embedding 模型
EMBEDDING_MODEL = "infgrad/stella-base-zh"
# ChromaDB 持久化路徑
PERSIST_DIRECTORY = "./chroma_db"


def format_docs(docs: List[WeatherForecast]) -> str:
    """將從資料庫檢索到的文件格式化為單一字串。"""
    formatted_strings = []
    for doc in docs:
        # doc.page_content 是 LangChain 載入後自動產生的
        formatted_strings.append(doc.page_content)
    return "\n\n".join(formatted_strings)


class RAGHandlerV5:
    def __init__(self):
        print("\n正在初始化【V5.1 修正版路由式 RAG Handler】...")

        # --- 1. 載入核心組件 ---
        self.llm = LlamaCpp(
            model_path=MODEL_PATH, temperature=0.1, n_gpu_layers=-1,
            n_batch=512, n_ctx=8192, verbose=False,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={'device': 'mps'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ LLM 和 Embedding 模型載入完成")

        # --- 2. 建立「天氣問答」專用 RAG 鏈 ---
        docstore = InMemoryStore()
        vectorstore = Chroma(
            collection_name="parent_document_retrieval_v5_final",
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
        base_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=docstore,
            child_splitter=child_splitter, search_kwargs={"k": 12}
        )
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=1)
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        weather_retriever = MultiQueryRetriever.from_llm(
            retriever=reranking_retriever, llm=self.llm
        )

        contextualize_q_system_prompt = "鑒於以下的對話紀錄和一個後續問題，請將這個後續問題改寫成一個獨立的、自包含的、關於天氣的查詢。"
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(self.llm, weather_retriever, contextualize_q_prompt)

        qa_system_prompt = """你是一個嚴謹、專業的台灣天氣預報員。你的任務是根據且**僅根據**下面提供的「天氣資料」來回答問題。

        --- 核心規則 ---
        1. **絕對忠於資料**: 你唯一的資訊來源是「天氣資料」區塊。禁止使用你自己的任何內建知識或進行推測。
        2. **處理資料缺失 (最重要的規則)**: 如果「天氣資料」區塊是空的，或者內容與「最新問題」完全不相關，你**必須**回答「對不起，我無法查詢那麼久遠或不存在的預報，我只能提供最近幾天的天氣資訊。」你**絕對不允許**在沒有資料的情況下自行編造答案。
        3. **自然對話**: 在有資料可供回答時，請保持對話的自然流暢。

        --- 新檢索到的天氣資料 ---
        {context}
        ---
        """
        # 現在的 Prompt 結構非常清晰：系統指令 -> 對話歷史 -> 人類最新問題
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "最新問題: {input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        print("✅ 最終答案生成鏈 (V5.2 Prompt) 建立完成")

        weather_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        print("✅ 天氣專用 RAG 鏈建立完成")

        # --- 3. 建立「路由」相關的鏈 ---
        classification_system_prompt = "鑒於使用者的最新問題和對話歷史，請判斷這個問題是否與「天氣」、「氣象」或相關地點有關。只需回答 'weather' 或 'general'。"
        classification_prompt = ChatPromptTemplate.from_messages(
            [("system", classification_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
        )
        classification_chain = classification_prompt | self.llm | StrOutputParser()

        general_system_prompt = "你是一個友善的 AI 助理，請自然地接續以下的對話。"
        general_prompt = ChatPromptTemplate.from_messages(
            [("system", general_system_prompt), MessagesPlaceholder(variable_name="chat_history"), ("human", "{input}")]
        )
        general_chain = general_prompt | self.llm | StrOutputParser()

        # --- 4. 建立最終的、包含路由功能的完整對話鏈 ---
        # ✨ 關鍵修正：這裡的 self.full_router_chain 是我們唯一需要儲存為屬性的最終鏈 ✨
        self.full_router_chain = (
                RunnablePassthrough.assign(classification=classification_chain)
                | RunnableBranch(
            (lambda x: "weather" in x["classification"].lower(), weather_rag_chain),
            general_chain,
        )
        )
        print("✅ V5.1 路由版對話式 RAG 鏈完全建立完成！")

    def setup_vector_store(self, db: Session):
        """為 V5 Handler 建立索引的函式。"""
        print("正在為 V5 準備資料並建立索引...")
        forecasts = crud.get_all_forecasts(db)
        if not forecasts:
            print("資料庫中沒有資料，跳過。")
            return 0

        grouped_data = {}
        for f in forecasts:
            key = (f.county, f.town)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(f)

        parent_docs = []
        for (county, town), daily_forecasts in grouped_data.items():
            sorted_forecasts = sorted(daily_forecasts, key=lambda x: x.start_time)
            full_day_text = f"這是 {county}{town} 的詳細天氣預報：\n"
            for f in sorted_forecasts:
                full_day_text += (
                    f"- 從 {f.start_time.strftime('%Y-%m-%d %H:%M')} 到 {f.end_time.strftime('%Y-%m-%d %H:%M')}，"
                    f"天氣為「{f.wx}」，溫度約為 {f.t}°C，相對濕度 {f.rh}%。\n"
                )
            parent_docs.append(Document(page_content=full_day_text, metadata={"county": county, "town": town}))

        if parent_docs:
            vectorstore = Chroma(
                collection_name="parent_document_retrieval_v5_final",
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.embeddings
            )
            docstore = InMemoryStore()
            indexing_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=docstore,
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=256),
            )
            indexing_retriever.add_documents(parent_docs, ids=None, add_to_docstore=True)
            print(f"✅ V5 索引建立完成！共處理了 {len(parent_docs)} 份父文檔。")
            return len(parent_docs)
        else:
            print("沒有可以建立索引的父文檔。")
            return 0

    def ask(self, query: str, chat_history: list):
        """使用 V5 路由鏈處理查詢。"""
        print(f"V5 收到問題: '{query}'")

        langchain_chat_history = []
        for msg in chat_history:
            if msg['role'] == 'user':
                langchain_chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'bot':
                langchain_chat_history.append(AIMessage(content=msg['content']))

        response = self.full_router_chain.invoke({
            "chat_history": langchain_chat_history,
            "input": query
        })

        if isinstance(response, dict):
            answer = response.get("answer", "我好像出了點問題，找不到答案。").strip()
        else:
            answer = response.strip()

        updated_history = chat_history + [
            {"role": "user", "content": query},
            {"role": "bot", "content": answer}
        ]
        return {"answer": answer, "chat_history": updated_history}
