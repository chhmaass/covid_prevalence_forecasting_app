# COVID-19 Prevalence Forecasting App

A **horizon-aware quantile forecasting system** for retrospective COVID-19 **point prevalence** analysis under **policy as-if scenarios**.

This repository contains a **research forecasting prototype** that generates probabilistic, multi-horizon prevalence forecasts across countries.  
It is **not** a real-time surveillance system, early-warning system, or operational decision-support tool.

---

## Overview

The app produces **conditional quantile trajectories** (e.g. q10 / q50 / q90) for COVID-19 prevalence over multiple forecast horizons. Forecasts are **scenario-conditioned**, not causal: policy variables define *assumed future paths*, not interventions whose effects are identified.

The system is designed for:

- retrospective analysis  
- uncertainty-aware forecasting  
- controlled counterfactual-style scenario exploration  
- methodological transparency  

---

## Key Features

- 📈 **Probabilistic forecasting** via conditional quantile regression  
- ⏱️ **Horizon-aware global models** with direct multi-horizon prediction  
- 🦠 **Separate regime models** (Pre-Omicron / Omicron)  
- ⚙️ **FastAPI backend** for inference-only serving  
- 📊 **Streamlit frontend** for interactive scenario exploration  
- 📦 **Pre-trained model artifacts & processed data** for reproducible inference  
- ⚖️ **Explicit epistemic limits** (scenario conditioning, no causal claims)  

---

## Repository Structure

```
COVID_PREVALENCE_FORECASTING_APP
├── backend
│   ├── app
│   │   ├── inference
│   │   │   └── quantile_forecast.py
│   │   ├── models
│   │   │   ├── common.py
│   │   │   ├── omicron.py
│   │   │   └── pre_omicron.py
│   │   ├── config.py
│   │   ├── features.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── serving.py
│   │   └── utils.py
│   │
│   ├── artifacts
│   │   ├── omicron
│   │   │   ├── center_means_omicron.json
│   │   │   ├── country_index_map_omicron.json
│   │   │   ├── feature_contract_omicron.json
│   │   │   ├── feature_norm_stats_omicron.json
│   │   │   ├── serving_schema_omicron.json
│   │   │   └── val_countries_omicron.json
│   │   ├── pre_omicron
│   │   │   ├── center_means_pre_omicron.json
│   │   │   ├── country_index_map_pre_omicron.json
│   │   │   ├── feature_contract_pre_omicron.json
│   │   │   ├── feature_norm_stats_pre_omicron.json
│   │   │   ├── serving_schema_pre_omicron.json
│   │   │   └── val_countries_pre_omicron.json
│   │   └── shared
│   │
│   ├── data
│   │   ├── df_final_omicron.csv
│   │   └── df_final_pre_omicron.csv
│   │
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend
│   ├── api_client.py
│   ├── streamlit_app.py
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
│
├── training
│   ├── omicron
│   │   ├── inference_notebook.ipynb
│   │   └── training_notebook.ipynb
│   └── pre_omicron
│       ├── inference_notebook.ipynb
│       └── training_notebook.ipynb
│
├── docker-compose.yml
├── How_to_run_app.txt
├── README.md
└── .gitignore
```

---

## Modeling Approach

The forecasting system is implemented as a **global, horizon-aware neural quantile regressor**, trained jointly across:

- countries  
- forecast horizons  
- epidemiological regimes  

Two **regime-specific models** are used:

- **Pre-Omicron**
- **Omicron**

At inference time, regime-specific forecasts are:

- selected based on the forecast anchor week, or  
- manually overridden, and  
- **blended** into a single probabilistic forecast distribution per horizon.

### Forecast Outputs

Outputs are **conditional quantile trajectories** (e.g. q10 / q50 / q90), allowing:

- explicit uncertainty inspection  
- horizon-specific dispersion analysis  
- baseline vs. scenario comparison  

---

## Policy-Sensitive Scenario Forecasting

Policy variables enter the model **only as scenario-defining conditions**, not as causal treatments.

**Key design choices:**

- Policy inputs are normalized to `[0, 1]`
- Policy variables are treated as **endogenous and reactive**
- Only **future policy paths** are manipulated in scenarios
- Historical epidemiological trajectories remain fixed and observed

### Structural Training Constraints

During training, the model is biased toward **conservative, causality-favorable behavior** via:

- lag gating  
- directional constraints  
- effect capping  
- regularization  

⚠️ **Important:**  
The system **does not estimate, identify, or validate causal policy effects**.  
Forecasts are conditional projections under explicit assumptions.

---

## Training vs. Inference Separation

All substantive modeling assumptions are implemented **during training** and embedded in the **trained model artifacts**, including:

- horizon-aware architecture  
- constrained policy influence  
- regime separation  
- regularization structure  

### Inference-Time Behavior

At runtime:

- pre-trained artifacts are loaded  
- no retraining or optimization occurs  
- no causal checks or constraint enforcement is applied  

Interpretation of outputs therefore depends on:

- the documented training procedure  
- the provenance of the model artifacts used  

---

## Backend API (FastAPI)

The backend performs **inference only**.

It:

- loads pre-trained Pre-Omicron and Omicron models  
- generates horizon-aware quantile forecasts  
- blends regime-specific outputs  
- optionally returns recent historical context  

It **does not**:

- train models  
- optimize parameters  
- estimate causal effects  
- perform policy evaluation  

---

## Frontend (Streamlit)

The Streamlit frontend is a **pure exploration and visualization layer**.

Users can:

- select a country and forecast anchor week  
- define future policy scenarios:
  - constant paths  
  - linear ramps  
  - manual paths  
- request probabilistic forecasts from the backend  

Visualizations include:

- historical prevalence and forecast trajectories  
- horizon-specific uncertainty bands  
- baseline vs. scenario comparisons  

No forecasting logic is implemented in the frontend.

---

## Data

The application uses **exactly two processed training artifacts**:

- `df_final_pre_omicron.csv`
- `df_final_omicron.csv`

These are regime-specific transformations of a harmonized, cross-national COVID-19 risk and policy dataset.

No additional external policy datasets are introduced at inference time.

---

## Scope and Non-Goals

This system is **not intended to**:

- estimate or identify causal policy effects  
- provide policy recommendations  
- support real-time surveillance or alerting  
- function as an operational decision-support system  

All outputs should be interpreted as:

> **retrospective, conditional probabilistic forecasts under explicit scenario assumptions**

---

## License

MIT License (unless otherwise specified).

---

## Citation

Christoph H. Maass (2025).  
COVID-19 Prevalence Forecasting App:  
Policy-Sensitive Scenario Forecasts (2020–2023).  
GitHub repository:  
https://github.com/chhmaass/covid_prevalence_forecasting_app
