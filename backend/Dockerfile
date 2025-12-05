# Backend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# requirements.txt is inside the backend folder relative to the repo root
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code (including app/, artifacts/, etc.)
COPY backend/ .

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

