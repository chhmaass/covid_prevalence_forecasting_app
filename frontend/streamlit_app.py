# streamlit_app.py
# ---------------------------------------------------------------------
# COVID-19 Prevalence — Variant-Aware Scenario Forecast App
#
# Frontend for a BACKEND API that wraps TWO variant-specific models:
#   - pre-Omicron
#   - Omicron
#
# The backend endpoint /v1/quantile_forecast is expected to:
#   - run both models
#   - blend their outputs according to a variant blend
#   - return:
#       - blended quantile forecasts per horizon
#       - blended baseline vs scenario deltas
#       - raw model outputs for debugging
#       - (optionally) recent HISTORY (prevalence + policies)
#
# This app:
#   - collects country + week_id_or_idx
#   - builds a horizon-aware scenario path for 3 policies
#   - sends that + variant blend to the backend
#   - visualizes:
#         • HISTORY + FORECAST combined chart  (new!)
#         • blended quantile forecasts
#         • baseline vs scenario deltas
#         • raw model outputs
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from api_client import post_quantile_forecast

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

H_MODEL: int = 12

POLICY_COLS = [
    "covid_19_policy_stringency",
    "covid_19_face_covering_policy",
    "covid_19_testing_tracing_policy",
]

POLICY_RANGES = {
    "covid_19_policy_stringency": (0.0, 1.0, 0.5),
    "covid_19_face_covering_policy": (0.0, 1.0, 0.5),
    "covid_19_testing_tracing_policy": (0.0, 1.0, 0.5),
}

DEFAULT_QUANTILES = [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------
# Helpers (scenario construction)
# ---------------------------------------------------------------------
def build_constant_path(levels: Tuple[float, float, float],
                        horizon: int = H_MODEL) -> np.ndarray:
    v = np.asarray(levels, dtype=float).reshape(1, 3)
    return np.repeat(v, horizon, axis=0)


def build_linear_path(start_levels: Tuple[float, float, float],
                      end_levels: Tuple[float, float, float],
                      horizon: int = H_MODEL) -> np.ndarray:
    start = np.asarray(start_levels, dtype=float).reshape(1, 3)
    end = np.asarray(end_levels, dtype=float).reshape(1, 3)
    steps = np.linspace(0.0, 1.0, horizon).reshape(horizon, 1)
    return start + (end - start) * steps


def build_manual_path(df: pd.DataFrame,
                      horizon: int = H_MODEL) -> np.ndarray:
    df = df.copy()
    df = df[POLICY_COLS].astype(float)
    if len(df) != horizon:
        raise ValueError(
            f"Manual path length {len(df)} != horizon {horizon}; "
            "the table must have exactly H rows."
        )
    return df.to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Variant-Aware Scenario Forecasts",
    page_icon="🦠",
    layout="wide",
)

st.title("🦠 COVID-19 Prevalence — Variant-Aware Scenario Forecasts")

st.markdown("""
This app calls a **backend forecasting API** that wraps two horizon-aware
quantile regressors (**pre-Omicron** and **Omicron**).  
You can design 12-step policy scenarios and inspect **history + blended forecasts**.
""")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("Settings")

country_iso3 = st.sidebar.text_input(
    "Country ISO3", value="USA").strip().upper()

# week_id_or_idx = st.sidebar.number_input(
# "Anchor week_id_or_idx", min_value=0, max_value=9999, value=100
# )
week_id_or_idx = st.sidebar.number_input(
    "Global week index (week_id_or_idx)",
    min_value=0,
    max_value=178,
    value=100,
    help="Global pandemic week; backend maps this to pre- and omicron local weeks."
)


display_h = st.sidebar.slider(
    "Display horizon (steps ahead)", min_value=1, max_value=H_MODEL, value=H_MODEL
)

# w_omicron = st.sidebar.slider(
# "Omicron weight (variant blend)",
# min_value=0.0, max_value=1.0,
# value=1.0, step=0.05
# )

# st.sidebar.caption(
# "Policy sliders use **raw 0–1 policy levels** (matching model inputs)."
# )

auto_blend = st.sidebar.checkbox(
    "Auto variant blend (time-based)",
    value=True,
    help=(
        "If checked, the backend chooses pre/omicron weights based on the "
        "global week (transition zone around OMI_GLOBAL_START). "
        "If unchecked, use the slider below as a manual override."
    ),
)

w_omicron = st.sidebar.slider(
    "Omicron weight (manual variant blend; used only if auto is OFF)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
)

st.sidebar.caption(
    "Policy sliders use **raw 0–1 policy levels** "
    "(0 = minimal restrictions, 1 = max in training data)."
)


# ---------------------------------------------------------------------
# 1. Scenario Builder
# ---------------------------------------------------------------------
st.subheader("1. Design a future policy scenario")

scenario_type = st.radio(
    "Scenario type",
    ["Constant", "Linear trend", "Manual (per-step table)"],
    horizontal=True
)

scenario_path: Optional[np.ndarray] = None

if scenario_type in ("Constant", "Linear trend"):
    st.markdown("#### Policy sliders (0–1 normalized raw levels)")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("**Start levels (h = 1)**")
        s_start = st.number_input("Stringency start",
                                  *POLICY_RANGES["covid_19_policy_stringency"])
        f_start = st.number_input("Face covering start",
                                  *POLICY_RANGES["covid_19_face_covering_policy"])
        t_start = st.number_input("Testing & tracing start",
                                  *POLICY_RANGES["covid_19_testing_tracing_policy"])

    with c2:
        st.write("**End levels (h = H)**")
        s_end = st.number_input("Stringency end",
                                *POLICY_RANGES["covid_19_policy_stringency"],
                                disabled=(scenario_type != "Linear trend"))
        f_end = st.number_input("Face covering end",
                                *POLICY_RANGES["covid_19_face_covering_policy"],
                                disabled=(scenario_type != "Linear trend"))
        t_end = st.number_input("Testing & tracing end",
                                *POLICY_RANGES["covid_19_testing_tracing_policy"],
                                disabled=(scenario_type != "Linear trend"))

    with c3:
        st.metric("Model horizon H", H_MODEL)

    if scenario_type == "Constant":
        scenario_path = build_constant_path((s_start, f_start, t_start))
    else:
        scenario_path = build_linear_path(
            (s_start, f_start, t_start),
            (s_end, f_end, t_end)
        )

else:  # Manual
    st.markdown("#### Manual per-step policy table (must have 12 rows)")
    initial_df = pd.DataFrame({
        "step": np.arange(1, H_MODEL + 1),
        "covid_19_policy_stringency": 0.5,
        "covid_19_face_covering_policy": 0.5,
        "covid_19_testing_tracing_policy": 0.5,
    })
    edited_df = st.data_editor(
        initial_df, num_rows="fixed", use_container_width=True)
    try:
        scenario_path = build_manual_path(edited_df)
    except Exception as e:
        st.error(f"Invalid manual scenario: {e}")
        scenario_path = None

# Preview
if scenario_path is not None:
    st.markdown("##### Scenario preview")
    df_prev = pd.DataFrame(scenario_path, columns=POLICY_COLS)
    df_prev.insert(0, "step", np.arange(1, H_MODEL + 1))
    st.dataframe(df_prev.head(display_h), use_container_width=True)
else:
    st.info("Define a valid scenario path to continue.")

st.markdown("---")

# ---------------------------------------------------------------------
# 2. Run forecast
# ---------------------------------------------------------------------
st.subheader("2. Run blended forecast (pre-Omicron + Omicron)")
run = st.button("🚀 Run forecast", type="primary")

if run:
    if scenario_path is None:
        st.error("Scenario path invalid.")
        st.stop()

    # payload = {
        # "country_iso3": country_iso3,
        # "week_id_or_idx": int(week_id_or_idx),
        # "policy_sliders": scenario_path.tolist(),
        # "quantiles": DEFAULT_QUANTILES,
        # "variant_blend": {
        # "pre_omicron_weight": float(1.0 - w_omicron),
        # "omicron_weight": float(w_omicron),
        # },
    # }
    if auto_blend:
        # Let the backend compute time-based weights from global_week
        variant_blend = {
            "pre_omicron_weight": 0.0,
            "omicron_weight": 0.0,
        }
    else:
        # Manual override: use the sidebar slider
        variant_blend = {
            "pre_omicron_weight": float(1.0 - w_omicron),
            "omicron_weight": float(w_omicron),
        }

    payload = {
        "country_iso3": country_iso3,
        "week_id_or_idx": int(week_id_or_idx),
        "policy_sliders": scenario_path.tolist(),
        "quantiles": DEFAULT_QUANTILES,
        "variant_blend": variant_blend,
    }

    with st.spinner("Contacting backend..."):
        try:
            response = post_quantile_forecast(payload)
            # st.json(response)  # temporarily
        except Exception as e:
            st.error(f"Backend call failed: {e}")
            st.stop()

    preds = response.get("predictions", {}) or {}
    cmps = response.get("compare", {}) or {}
    history = response.get("history", {}) or {}

    def to_df(maybe_list_or_none):
        if maybe_list_or_none is None:
            return None
        df = pd.DataFrame(maybe_list_or_none)
        return df if not df.empty else None

    # Blended
    df_blend_pred = to_df(preds.get("blended"))
    df_blend_cmp = to_df(cmps.get("blended"))

    # Raw pre/omicron (for debug + extra charts)
    pre_pred = to_df(preds.get("pre"))
    omi_pred = to_df(preds.get("omicron"))
    pre_cmp = to_df(cmps.get("pre"))
    omi_cmp = to_df(cmps.get("omicron"))

    # Optional: show raw response structure for debugging
    # st.expander("Raw backend response").json(response)

    # -------------------
    # History block (NEW)
    # -------------------
    week_hist = history.get("week_id", [])
    prev_hist = history.get("prevalence", [])
    pol_hist = history.get("policies", {}) or {}

    if week_hist and prev_hist:
        df_hist = pd.DataFrame({
            "week_id": week_hist,
            "prevalence": prev_hist,
        })
    else:
        df_hist = pd.DataFrame()

    # Truncate forecast
    if df_blend_pred is not None and "h" in df_blend_pred.columns:
        df_blend_pred = df_blend_pred[df_blend_pred["h"] <= display_h]

    if df_blend_cmp is not None and "h" in df_blend_cmp.columns:
        df_blend_cmp = df_blend_cmp[df_blend_cmp["h"] <= display_h]

    # ---------------------------------------------------------------------
    # HISTORY + FORECAST combined chart (NEW)
    # ---------------------------------------------------------------------
    st.markdown("### 📈 History + Forecast (q10 / q50 / q90)")

    if df_blend_pred is None:
        st.info("No blended predictions returned.")
    else:
        df_f = df_blend_pred.copy().sort_values("week_id")

        layers = []

        if not df_hist.empty:
            anchor = int(df_hist["week_id"].iloc[-1])
            hist_line = (
                alt.Chart(df_hist)
                .mark_line(strokeWidth=2, color="steelblue")
                .encode(
                    x="week_id:Q",
                    y="prevalence:Q",
                    tooltip=["week_id", "prevalence"],
                )
            )
            layers.append(hist_line)
        else:
            anchor = int(df_f["week_id"].iloc[0])

        # Forecast band
        band = (
            alt.Chart(df_f)
            .mark_area(opacity=0.25)
            .encode(
                x="week_id:Q",
                y="q10:Q",
                y2="q90:Q",
            )
        )

        # q50 line
        line = (
            alt.Chart(df_f)
            .mark_line(color="black")
            .encode(
                x="week_id:Q",
                y="q50:Q",
                tooltip=["week_id", "q10", "q50", "q90"],
            )
        )

        layers.extend([band, line])

        # Anchor line
        vline = alt.Chart(pd.DataFrame({"week_id": [anchor]})).mark_rule(
            strokeDash=[4, 4], color="red"
        ).encode(x="week_id:Q")

        layers.append(vline)

        st.altair_chart(alt.layer(*layers).properties(height=350).interactive(),
                        use_container_width=True)

    # ---------------------------------------------------------------------
    # Optional policy history (NEW)
    # ---------------------------------------------------------------------
    if pol_hist:
        st.markdown("#### Recent policy history (normalized 0–1)")
        dfp = pd.DataFrame({
            "week_id": week_hist,
            "stringency": pol_hist.get("covid_19_policy_stringency", []),
            "face_covering": pol_hist.get("covid_19_face_covering_policy", []),
            "testing_tracing": pol_hist.get("covid_19_testing_tracing_policy", []),
        })
        melt = dfp.melt("week_id", var_name="policy", value_name="value")
        pol_chart = (
            alt.Chart(melt)
            .mark_line()
            .encode(
                x="week_id:Q",
                y="value:Q",
                color="policy:N",
                tooltip=["week_id", "policy", "value"],
            )
        )
        st.altair_chart(pol_chart.interactive(), use_container_width=True)

    st.markdown("---")

    # ---------------------------------------------------------------------
    # Blended future-only forecast
    # ---------------------------------------------------------------------
    st.markdown("### Blended forecast (future only)")
    if df_blend_pred is not None and not df_blend_pred.empty:
        chart_df = df_blend_pred.melt(
            id_vars="h",
            value_vars=["q10", "q50", "q90"],
            var_name="quantile",
            value_name="value",
        )
        q50 = chart_df[chart_df["quantile"] == "q50"]
        band_low = df_blend_pred[["h", "q10"]].rename(columns={"q10": "lower"})
        band_high = df_blend_pred[["h", "q90"]].rename(
            columns={"q90": "upper"})
        band2 = band_low.merge(band_high, on="h")

        band_chart = alt.Chart(band2).mark_area(opacity=0.25).encode(
            x="h:Q", y="lower:Q", y2="upper:Q"
        )
        line_chart = alt.Chart(q50).mark_line().encode(
            x="h:Q", y="value:Q"
        )
        st.altair_chart((band_chart + line_chart).properties(height=300),
                        use_container_width=True)

    # ---------------------------------------------------------------------
    # Baseline vs scenario
    # ---------------------------------------------------------------------
    st.markdown("### Baseline vs Scenario (q50 per horizon)")
    if df_blend_cmp is not None and not df_blend_cmp.empty:
        st.dataframe(df_blend_cmp, use_container_width=True)
        st.download_button(
            "Download baseline vs scenario CSV",
            df_blend_cmp.to_csv(index=False).encode("utf-8"),
            f"{country_iso3}_baseline_vs_scenario.csv",
            "text/csv"
        )
    else:
        st.info("No compare_horizons data returned.")

    # ---------------------------------------------------------------------
    # Debug tabs
    # ---------------------------------------------------------------------
    st.markdown("### Raw model outputs (debug)")
    t1, t2, t3, t4 = st.tabs([
        "pre: predict_horizon",
        "pre: compare_horizons",
        "omicron: predict_horizon",
        "omicron: compare_horizons",
    ])

    with t1:
        st.dataframe(pre_pred.head(display_h)
                     if pre_pred is not None else None)

    with t2:
        st.dataframe(pre_cmp.head(display_h) if pre_cmp is not None else None)

    with t3:
        st.dataframe(omi_pred.head(display_h)
                     if omi_pred is not None else None)

    with t4:
        st.dataframe(omi_cmp.head(display_h) if omi_cmp is not None else None)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.caption("""
Policy sliders: raw normalized levels 0–1.  
Horizon always 12.  
History returned by backend is shown on the combined chart.  
""")
