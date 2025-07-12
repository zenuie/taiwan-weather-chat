好的，這是一個專業、結構清晰且內容完整的 `README.md` 檔案。它不僅說明了如何使用您的專案，還詳細介紹了其先進的 RAG 架構和我們在偵錯與優化過程中所採用的各種技術，可以充分展示這個專案的技術深度。

您可以直接將以下所有內容複製到一個名為 `README.md` 的新檔案中，並將其放置在您專案的根目錄下。

---

# 台灣天氣 RAG 聊天機器人

這是一個基於本地大型語言模型（LLM）和先進檢索增強生成（RAG）技術的智慧天氣對話系統。使用者可以透過自然語言，與系統進行多輪、具備記憶功能、可流式輸出的天氣問答。

專案的核心目標是在保障資料隱私和無網路連線成本的前提下，利用本地運行的 AI 模型，提供一個準確、可靠且不會產生幻覺的天氣資訊助理。

![圖片](https://github.com/zenuie/taiwan-weather-chat/blob/main/wetherchat.png "圖片")

## ✨ 專案亮點

*   **完全本地運行**: 所有 AI 模型（LLM、Embedding、Re-ranker）均在本地運行，無需 API 金鑰，保障了資料隱私和零成本。
*   **先進的 RAG 架構 (RAG V6)**: 採用了多項業界前沿的 RAG 優化策略，確保了回答的準確性和可靠性。
*   **防幻覺機制**: 透過嚴格的 Prompt Engineering，有效抑制了 LLM 在資料不足時編造答案的傾向。
*   **對話式記憶**: 系統能夠理解多輪對話的上下文，支援「明天呢？」、「那裡會下雨嗎？」等追問。
*   **智慧路由**: 能自動判斷使用者意圖，將問題分流至「天氣 RAG 鏈」或「通用對話鏈」，提升互動的自然度。
*   **流式輸出**: 回答以打字機效果逐字出現，極大提升了使用者互動體驗。
*   **完整的 ETL 流程**: 自動從台灣中央氣象署（CWA）獲取最新的天氣預報資料，並存入本地資料庫。

## 🔧 技術

*   **後端框架**: [FastAPI](https://fastapi.tiangolo.com/)
*   **AI 應用框架**: [LangChain](https://www.langchain.com/)
*   **大型語言模型 (LLM)**: [Taiwan-LLM-13B-v2.1-chat-GGUF](https://huggingface.co/audreyt/Taiwan-LLM-13B-v2.0-chat-GGUF)
*   **LLM 推理引擎**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (支援 Apple Metal GPU 加速)
*   **嵌入模型 (Embedding)**: [infgrad/stella-base-zh](https://huggingface.co/infgrad/stella-base-zh) (針對中文優化)
*   **重排模型 (Re-ranker)**: [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
*   **向量資料庫**: [ChromaDB](https://www.trychroma.com/)
*   **資料庫**: [SQLite](https://www.sqlite.org/index.html)
*   **前端**: 純 HTML, CSS, JavaScript (無框架)

## 🚀 系統架構：從 ETL 到 RAG V6

本專案的 RAG 系統經過了多次迭代，最終演進為一個具備路由功能的、可靠的對話夥伴。

1.  **ETL (數據提取、轉換、加載)**:
    *   `fetcher.py`: 從 CWA OpenData API 獲取天氣預報 JSON 資料。
    *   `transformer.py`: 將原始資料轉換、清洗（如「臺」->「台」），並處理成結構化的 DataFrame。
    *   `crud.py`: 將處理後的資料儲存到本地 SQLite 資料庫 (`weather.db`)。

2.  **RAG 索引建立 (`rag-setup`)**:
    *   從 SQLite 讀取所有預報資料。
    *   **父文檔策略**: 將每個鄉鎮的單日預報合併成一個「父文檔」。
    *   **子文檔切分**: `ParentDocumentRetriever` 自動將父文檔切分成更小的「子文檔」。
    *   **向量化**: 使用 `stella-base-zh` 模型將子文檔轉換為向量，並存入 ChromaDB。父文檔原文則存入 `InMemoryStore`。

3.  **對話式查詢 (`stream-ask-weather`)**:
    *   **智慧路由**: 首先，一個「意圖分類鏈」會判斷問題是關於「天氣」還是「通用閒聊」。
    *   **通用閒聊分支**: 如果是閒聊，則直接由 LLM 根據對話歷史生成回答。
    *   **天氣 RAG 分支**:
        1.  **問題重述**: `create_history_aware_retriever` 根據對話歷史，將後續問題（如「明天呢？」）改寫為完整的查詢（如「台中市中區明天的天氣如何？」）。
        2.  **多查詢檢索**: `MultiQueryRetriever` 將重述後的問題，從不同角度生成多個子查詢，以提高召回率。
        3.  **父文檔檢索**: 使用子查詢在 ChromaDB 中進行向量搜索，找到最相關的「子文檔」，然後回溯到其對應的「父文檔」。
        4.  **重排 (Re-ranking)**: `CrossEncoderReranker` 對召回的多個父文檔進行精準的相關性打分，僅選出 `top_n=1` 的最相關文檔。
        5.  **防幻覺生成**: 將最相關的文檔作為 `context`，連同對話歷史和使用者問題，傳遞給帶有嚴格「防幻覺」指令的最終 Prompt，由 LLM 生成安全、可靠的答案。
    *   **流式輸出**: 所有生成過程均以流式傳輸，答案逐字呈現在前端。

## ⚙️ 安裝與設定

### 1. 前置需求

*   Python 3.11+
*   一台具備 Apple Silicon (M1/M2/M3/M4) 的 Mac (推薦，以獲得 Metal GPU 加速) 或其他支援的作業系統。
*   足夠的記憶體 (推薦 16GB+)。

### 2. 安裝依賴

首先，clone 本專案，然後安裝必要的 Python 套件。

```bash
git clone [您的專案 Git URL]
cd [專案目錄]
python -m venv .venv
source .venv/bin/activate
```

安裝核心依賴：
```bash
poetry install
```


**特別注意**: `llama-cpp-python` 的安裝對效能至關重要。對於 Apple Silicon Mac，請使用以下指令以啟用 Metal GPU 加速：
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### 3. 下載 LLM 模型

從 Hugging Face 下載 `Taiwan-LLM-13B-v2.1-chat` 的 GGUF 格式模型。推薦使用 `Q5_K_M` 版本，它在品質和效能上取得了很好的平衡。

*   **下載連結**: [weiren119/Taiwan-LLM-13B-v2.1-chat-GGUF](https://huggingface.co/audreyt/Taiwan-LLM-13B-v2.0-chat-GGUF)
*   **目標檔案**: `taiwan-llm-13b-v2.1-chat.Q5_K_M.gguf`

將下載好的 `.gguf` 檔案放置在專案的根目錄下。

### 4. 設定環境變數

在專案根目錄下建立一個名為 `.env` 的檔案，並加入以下內容以關閉 ChromaDB 的匿名遙測功能，避免日誌中出現不必要的網路連線錯誤。

```
ANONYMIZED_TELEMETRY=False
```

## 🚀 如何運行

請依照以下順序執行，以確保系統正常工作。

### 1. 啟動後端服務

在專案根目錄的終端機中，運行以下指令：

```bash
uvicorn cba.main:app --reload
```

服務啟動時，您會看到模型載入的日誌，這可能需要一到兩分鐘。服務將運行在 `http://127.0.0.1:8000`。

### 2. 執行 ETL 並建立索引 (一次性)

在服務啟動後，您需要先準備好資料。請使用 API 工具 (如 Postman, Insomnia) 或 `curl`，向後端發送 `POST` 請求。

**第一步：抓取天氣資料**
```bash
curl -X POST http://127.0.0.1:8000/etl-update
```
**第二步：為 RAG 建立向量索引**
```bash
curl -X POST http://127.0.0.1:8000/rag-setup
```
*這兩個步驟只需要在您想要更新天氣資料時執行一次即可。*

### 3. 開啟前端介面

直接用您的瀏覽器（如 Chrome, Firefox, Safari）打開專案根目錄下的 `index.html` 檔案。

現在，您就可以開始與您的本地天氣 AI 進行對話了！
