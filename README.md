# COVID Prevalence Forecasting App

A horizon–aware **quantile forecasting** system for COVID-19 prevalence with:

- 🧠 PyTorch TorchScript models for **pre-Omicron** and **Omicron** regimes  
- ⚙️ FastAPI backend API for quantile forecasts  
- 📊 Streamlit frontend for interactive exploration and visualization  
- 📦 Complete artifacts and processed training data for reproducible inference  

---

## 1. Repository Structure

```text
COVID_PREVALENCE_FORECASTING_APP
├── backend/                 # FastAPI backend service
│   ├── .venv/               # Local virtualenv (ignored for Docker)
│   └── app/
│       ├── inference/
│       │   └── quantile_forecast.py
│       ├── models/
│       │   ├── common.py
│       │   ├── omicron.py
│       │   └── pre_omicron.py
│       ├── config.py
│       ├── features.py
│       ├── main.py
│       ├── schemas.py
│       ├── serving.py
│       └── utils.py
│
├── artifacts/
│   ├── pre_omicron/
│   └── omicron/
│
├── data/
│   ├── df_final_pre_omicron.csv
│   └── df_final_omicron.csv
│
├── frontend/                # Streamlit UI
│   ├── api_client.py
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── training/                # (optional) training notebooks / scripts
├── shared/                  # placeholder; currently unused
├── Dockerfile               # backend Dockerfile
├── docker-compose.yml       # multi-service orchestration
├── requirements.txt         # backend dependencies
├── How_to_run_app.txt
└── README.md

