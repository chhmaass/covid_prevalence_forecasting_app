# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# requirements.txt is in backend/, but the build context is the repo root.
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code (app/, artifacts/, etc.) into the image
COPY backend/ .

CMD ["sh", "-c", \"uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}\"]
