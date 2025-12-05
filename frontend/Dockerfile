# Frontend Dockerfile
FROM python:3.10-slim

WORKDIR /app

# requirements.txt is inside the frontend folder relative to the repo root
COPY frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend code (Streamlit app)
COPY frontend/ .

CMD ["sh", "-c", "streamlit run streamlit_app.py \
    --server.port ${PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false"]

