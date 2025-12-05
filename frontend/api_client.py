import os
import requests

# Prefer BACKEND_URL, fall back to BACKEND_BASE_URL, then localhost
BACKEND_URL = (
    os.getenv("BACKEND_URL")
    or os.getenv("BACKEND_BASE_URL")
    or "http://localhost:8000"
)


def post_quantile_forecast(payload):
    """Send a POST request to the backend quantile forecast API."""
    url = f"{BACKEND_URL}/v1/quantile_forecast"
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()
