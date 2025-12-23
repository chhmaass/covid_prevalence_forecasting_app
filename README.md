COVID-19 Prevalence Forecasting App

A horizon-aware quantile forecasting system for retrospective COVID-19 prevalence analysis under policy as-if scenarios.

The app generates probabilistic, multi-horizon forecasts of COVID-19 point prevalence across countries. It is a research forecasting prototype, not a real-time surveillance system, early-warning system, or operational decision-support tool.

Key Features

📈 Probabilistic forecasting via conditional quantile trajectories

⏱️ Horizon-aware global models with direct multi-horizon prediction

🦠 Separate pre-Omicron and Omicron regime models

⚙️ FastAPI backend for forecasting inference

📊 Streamlit frontend for interactive scenario exploration

📦 Trained model artifacts and processed data for reproducible inference

⚖️ Explicit epistemic limits (scenario conditioning, not causal inference)

1. Repository Structure
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
├── artifacts/               # Trained model artifacts
│   ├── pre_omicron/
│   └── omicron/
│
├── data/                    # Processed training data
│   ├── df_final_pre_omicron.csv
│   └── df_final_omicron.csv
│
├── frontend/                # Streamlit UI
│   ├── api_client.py
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── training/                # Training notebooks / scripts (optional)
├── shared/                  # Placeholder; currently unused
├── Dockerfile               # Backend Dockerfile
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Backend dependencies
├── How_to_run_app.txt
└── README.md

2. Modeling Approach

The forecasting system is implemented as a global, horizon-aware neural quantile regressor trained jointly across countries and forecast horizons.

Two regime-specific models are used.

Pre-Omicron

Omicron

At inference time, regime-specific forecasts are blended based on the forecast anchor week or manually overridden, producing a single probabilistic forecast distribution per horizon.

Forecast outputs are conditional quantile trajectories (for example q10 / q50 / q90), allowing explicit inspection of uncertainty and dispersion across horizons.

3. Policy-Sensitive Scenario Forecasting

Policy inputs enter the model as scenario-defining conditions, not as causal treatments.

Policies are normalized to the interval [0,1]

Policy variables are treated as endogenous and reactive

Only future policy paths are manipulated in scenarios

Historical epidemiological trajectories remain fixed and observed

Structural constraints applied during training, including lag gating, directional constraints, and effect capping, bias the model toward conservative, causality-favorable behavior. The system does not estimate or identify causal policy effects.

4. Training and Inference Separation

All substantive modeling assumptions, such as horizon-aware structure, constrained policy influence, and regularization choices, are implemented during model training and are reflected in the trained model artifacts.

At inference time, the application loads these pre-trained artifacts and applies them directly to generate probabilistic forecasts under user-defined scenarios. No additional constraint enforcement or assumption checking is performed during runtime. Interpretation of forecast outputs therefore relies on the documented training procedure and the provenance of the model artifacts used.

5. Backend API

The FastAPI backend performs probabilistic forecasting inference.

Pre-trained pre-Omicron and Omicron models are loaded

Horizon-aware quantile forecasts are generated

Regime-specific outputs are blended

Recent historical prevalence and policy context may be returned

No training, optimization, or causal analysis is performed at runtime.

6. Frontend (Streamlit)

The Streamlit frontend provides an interactive interface for scenario exploration.

Users can:

Select a country and historical forecast anchor week

Construct future policy scenarios using constant, linear, or manual paths

Request blended probabilistic forecasts from the backend

Visualize

historical prevalence and forecast trajectories

horizon-specific uncertainty bands

baseline versus scenario comparisons

The frontend performs no forecasting logic itself and serves purely as an exploration and visualization layer.

7. Data

The app uses exactly two processed training artifacts.

df_final_pre_omicron.csv

df_final_omicron.csv

These are regime-specific transformations of a harmonized cross-national COVID-19 risk and policy dataset. No additional external policy datasets are introduced at inference time.

8. Scope and Non-Goals

This system is not intended to:

estimate or identify causal policy effects

provide policy recommendations

support real-time surveillance or alerting

function as an operational decision-support system

All outputs should be interpreted as retrospective, conditional probabilistic forecasts under explicit scenario assumptions.

License

MIT License unless otherwise specified.

Citation

Christoph H. Maass (2025).
COVID-19 Prevalence Forecasting App: Policy-Sensitive Scenario Forecasts (2020-2023)
GitHub repository: https://github.com/chhmaass/covid_prevalence_forecasting_app