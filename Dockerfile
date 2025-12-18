# 1. 基底映像
FROM python:3.11-slim

# 2. 安裝 uv（官方推薦方式）
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# 3. 先複製 requirements，利用快取
COPY requirements.txt .

# 4. 用 uv 安裝依賴
RUN uv pip install --system --no-cache -r requirements.txt

# 5. 再複製程式碼
COPY . .

# 6. 啟動 FastAPI（Cloud Run 預設會設 PORT 環境變數）
ENV PORT=8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]