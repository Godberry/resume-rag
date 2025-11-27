# resume-rag
一個基於我個人履歷的知識庫，使用 RAG (檢索增強生成) 技術，可以透過自然語言問答的方式深入了解我的技能、專案經驗與背景。

目前架構已調整為 **前後端分離**：

- `backend/`：FastAPI 後端服務與向量資料庫建置
	- `backend/main.py`：提供 RESTful API（例如 `POST /chat`）供前端呼叫
	- `backend/ingest.py`：將 `knowledge_base/` 中的 Markdown 文件切分後寫入 Chroma 向量資料庫
- `frontend/`：簡單的網頁聊天介面（純 HTML + JavaScript），負責呼叫後端 API
- `knowledge_base/`：履歷與專案說明的 Markdown 檔案

## 安裝與執行

### 1. 安裝依賴

在專案根目錄（`resume-rag/`）下，先建立虛擬環境（可選），再安裝後端依賴：

```bash
cd backend
pip install -r requirements.txt
```

請在系統環境或在 `backend/.env` 中設定 `OPENAI_API_KEY`：

```bash
echo "OPENAI_API_KEY=你的 API Key" > .env
```

### 2. 建立 / 更新向量資料庫

在 `backend/` 目錄下執行：

```bash
cd backend
python ingest.py
```

這會讀取 `../knowledge_base/` 中的 Markdown 檔案，切分後寫入 `../chroma_db/`。

### 3. 啟動 FastAPI 後端

仍在 `backend/` 目錄，啟動 uvicorn：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

啟動後可以用瀏覽器打開 `http://localhost:8000/docs` 查看自動產生的 API 文件，
或請求 `GET http://localhost:8000/health` 確認服務狀態。

### 4. 啟動前端

目前前端是一個靜態網頁 `frontend/index.html`，你可以直接用瀏覽器開啟，
或者使用任何靜態檔案伺服器（例如 VS Code Live Server、`python -m http.server` 等）提供服務。

開啟後端口預設呼叫 `http://localhost:8000/chat`，請確保後端已啟動。

---

> 備註：原本的 `app.py`（Streamlit 版本）仍保留在根目錄，可視需要逐步移除或改為呼叫 FastAPI API。
