"""
Omicron model wrapper.

This module implements the public interface:

    - predict_horizon_omi(country_iso3, week_id_or_idx, policy_sliders)
    - compare_horizons_omi(country_iso3, week_id_or_idx, scenario_sliders)

It mirrors the behaviour of the Colab inference notebook:

    "omicron inference gated global horizon-aware quantile regressor.ipynb"

but uses local artifacts under backend/artifacts/omicron and the
shared helpers in app.models.common.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from app.config import H_MODEL
from app.models.common import get_variant_artifacts, run_variant_model

# ---------------------------------------------------------------------
# Variant + artifacts
# ---------------------------------------------------------------------

VARIANT = "omicron"

_artifacts = get_variant_artifacts(VARIANT)
_feature_contract = _artifacts.feature_contract
# dict: name -> {"mean": .., "std": ..}
_feature_norm = _artifacts.norm_stats.stats
_center_means = _artifacts.center_means.means        # dict: raw_name -> center
_feature_order = _feature_contract.feature_order     # list[str]
# must equal feature_contract.horizon
_H = H_MODEL

# ---------------------------------------------------------------------
# Data (history) – Omicron only
# ---------------------------------------------------------------------

# You can override this via env var if your data path differs.
_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]  # backend/
    / "data"
    / "df_final_omicron.csv"
)
DATA_PATH = Path(os.getenv("OMICRON_DATA_PATH", str(_DEFAULT_DATA_PATH)))

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Omicron history CSV not found at {DATA_PATH}. "
        f"Set OMICRON_DATA_PATH or place df_final_omicron.csv in backend/data/."
    )

df = pd.read_csv(DATA_PATH)
df = df.sort_values(["country_iso3", "week_id"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Configuration constants (must match training / notebook)
# ---------------------------------------------------------------------

# Koyck windows halflives for policy sliders
HALFLIVES_POLICY: Tuple[int, int] = (6, 12)
EWM_TAGS = [f"ewm_hl{hl}" for hl in HALFLIVES_POLICY]

# Discrete policy lags used in training
LAG_STEPS: Tuple[int, int, int, int] = (1, 2, 3, 4)

# Raw policy slider columns used in training/inference
POLICY_COLS: Tuple[str, str, str] = (
    "covid_19_policy_stringency",
    "covid_19_face_covering_policy",
    "covid_19_testing_tracing_policy",
)

# (Training has interactions OFF; keep hook for compatibility)
INT_MOD_RELU_NAMES: Tuple[str, ...] = tuple()

# Seasonality config
WEEKS_PER_YEAR: float = 52.1775  # consistent with training


# ---------------------------------------------------------------------
# Low-level helpers (copied/adapted from notebook)
# ---------------------------------------------------------------------


def _ewm_alpha(halflife: float) -> float:
    return 1.0 - math.exp(-math.log(2.0) / float(halflife))


def _series_ewm(series: Iterable[float], hl: float) -> np.ndarray:
    """Causal EWM (Koyck) for a 1D array-like; aligns with pandas ewm(adjust=False)."""
    a = _ewm_alpha(hl)
    arr = np.asarray(series, dtype=float)
    out = np.empty(len(arr), dtype=float)
    prev = None
    for i, xi in enumerate(arr):
        prev = xi if i == 0 else (a * float(xi) + (1.0 - a) * float(prev))
        out[i] = prev
    return out


def _resolve_week_id_for_country(
    df_country: pd.DataFrame,
    week_id_or_idx: int,
) -> int:
    """
    Accept a true week_id or a 0-based positional index into that country's history.
    """
    wk = int(week_id_or_idx)
    wkvals = df_country["week_id"].to_numpy()
    if (wk == wkvals).any():
        # It's a true week_id
        return wk
    if 0 <= wk < len(wkvals):
        # Treat as positional index
        return int(wkvals[wk])
    raise ValueError(
        f"{wk} not valid week_id or positional index (0..{len(wkvals) - 1})."
    )


def _normalize_scalar(name: str, raw_value: float, norm_stats: dict) -> float:
    s = norm_stats[name]
    mu = float(s["mean"])
    sd = float(s["std"]) or 1.0
    return (float(raw_value) - mu) / sd


def _policy_base_names() -> Sequence[str]:
    """
    Names of Koyck-smoothed policy windows used during training.
    """
    return [f"{feat}_{tag}" for tag in EWM_TAGS for feat in POLICY_COLS]


def _future_slider_path_from_data(
    df_country: pd.DataFrame,
    start_week_id: int,
    horizon: int,
) -> np.ndarray:
    """
    Baseline path from observed data: sliders (S, M, T) for weeks (t0+1 .. t0+H).
    Pads with the last available observed triple if shorter.
    """
    g = df_country[df_country["week_id"] > int(start_week_id)].copy().sort_values(
        "week_id"
    )
    fut = g[list(POLICY_COLS)].astype(float).to_numpy()
    if fut.size == 0:
        # No future rows; replicate the anchor's current policy triple
        row0 = df_country[df_country["week_id"] == int(start_week_id)].iloc[0]
        triple = np.array(
            [
                float(row0[POLICY_COLS[0]]),
                float(row0[POLICY_COLS[1]]),
                float(row0[POLICY_COLS[2]]),
            ],
            dtype=float,
        )
        triples = np.repeat(triple[None, :], horizon, axis=0)
        return triples

    triples = fut
    if len(triples) < horizon:
        pad = np.repeat(triples[-1][None, :], horizon - len(triples), axis=0)
        triples = np.vstack([triples, pad])
    return triples


def _build_future_policy_windows(
    history_df: pd.DataFrame,
    start_week_id: int,
    future_sliders: Union[Sequence[float], np.ndarray],
    horizon: int = _H,
) -> dict:
    """
    Build step-varying Koyck-smoothed RAW policy windows from history + future slider path.

    future_sliders: (3,) or (H,3) of policy levels (S, M, T) in model-native scale (0-1).
    Returns: dict mapping f"{raw}_ewm_hl{6|12}" -> np.ndarray of length H.
    """
    g = history_df[history_df["week_id"] <= int(start_week_id)].copy().sort_values(
        "week_id"
    )
    if g.empty:
        raise ValueError("No history available to build Koyck windows.")

    # History EWM states at t0
    last_ewm = {}
    for raw in POLICY_COLS:
        hist_vals = g[raw].astype(float).to_numpy()
        for hl in HALFLIVES_POLICY:
            sm = _series_ewm(hist_vals, hl)
            last_ewm[(raw, hl)] = float(sm[-1])

    fs = np.asarray(future_sliders, dtype=float)
    if fs.ndim == 1:
        if fs.shape[0] != 3:
            raise ValueError("Single future_sliders triple must be length 3.")
        future_triples = np.repeat(fs[None, :], horizon, axis=0)
    else:
        if fs.shape[1] != 3 or fs.shape[0] < horizon:
            raise ValueError(
                "future_sliders must be (H,3) or a single length-3 triple."
            )
        future_triples = fs

    out = {
        f"{raw}_ewm_hl{hl}": np.empty(horizon, dtype=float)
        for raw in POLICY_COLS
        for hl in HALFLIVES_POLICY
    }

    for h in range(horizon):
        s_raw, m_raw, t_raw = [float(x) for x in future_triples[h]]
        raw_vals = {
            "covid_19_policy_stringency": s_raw,
            "covid_19_face_covering_policy": m_raw,
            "covid_19_testing_tracing_policy": t_raw,
        }
        for raw in POLICY_COLS:
            xh_raw = raw_vals[raw]
            for hl in HALFLIVES_POLICY:
                a = _ewm_alpha(hl)
                prev = last_ewm[(raw, hl)]
                newv = a * xh_raw + (1.0 - a) * prev
                last_ewm[(raw, hl)] = newv
                out[f"{raw}_ewm_hl{hl}"][h] = newv

    return out


def _build_future_policy_lags(
    history_df: pd.DataFrame,
    start_week_id: int,
    future_sliders: Union[Sequence[float], np.ndarray],
    horizon: int = _H,
    lag_steps: Sequence[int] = LAG_STEPS,
) -> dict:
    """
    Build discrete lags of RAW policy sliders from history + future path.

    Uses the training convention:
        first k steps fall back to the *current* value (shift(k).fillna(current)).

    Returns: dict mapping f"{raw}_lag{k}" -> np.ndarray length H.
    """
    g = history_df.copy().sort_values("week_id")
    # Build a combined sequence of history + future path for each policy column
    hist = {p: g[p].astype(float).to_numpy() for p in POLICY_COLS}

    fs = np.asarray(future_sliders, dtype=float)
    if fs.ndim == 1:
        if fs.shape[0] != 3:
            raise ValueError("Single future_sliders triple must be length 3.")
        fut_seq = np.repeat(fs[None, :], horizon, axis=0)
    else:
        if fs.shape[1] != 3 or fs.shape[0] < horizon:
            raise ValueError(
                "future_sliders must be (H,3) or a single length-3 triple."
            )
        fut_seq = fs[:horizon]

    out = {
        f"{p}_lag{k}": np.empty(horizon, dtype=float)
        for p in POLICY_COLS
        for k in lag_steps
    }

    for p_idx, p in enumerate(POLICY_COLS):
        seq = np.concatenate(
            [hist[p], fut_seq[:, p_idx].astype(float)], axis=0
        )  # full history + future
        # Anchor index: history ends at index len(hist[p]) - 1
        # Future step h (1..H) corresponds to position (len(hist[p]) - 1 + h)
        for h in range(1, horizon + 1):
            cur_idx = len(hist[p]) - 1 + h
            for k in lag_steps:
                lag_idx = cur_idx - k
                if lag_idx < 0:
                    # Align with training's .shift(k).fillna(current)
                    val = seq[cur_idx]
                else:
                    val = seq[lag_idx]
                out[f"{p}_lag{k}"][h - 1] = float(val)

    return out


def _seasonal_for_week(week_id: int, latitude: float) -> dict:
    """
    Deterministic seasonality as a function of week_id and latitude.
    """
    angle = 2.0 * math.pi * (week_id / WEEKS_PER_YEAR)
    s = math.sin(angle)
    c = math.cos(angle)
    return {
        "sine_seasonality": s,
        "cosine_seasonality": c,
        "sine_seasonality_x_latitude": s * float(latitude),
        "cosine_seasonality_x_latitude": c * float(latitude),
    }


def _extract_knowns_rolling(
    df_country: pd.DataFrame,
    start_week_id: int,
    horizon: int,
) -> pd.DataFrame:
    """
    Build per-step known controls:
    - Seasonality varies with horizon step (week_id + h).
    - Latitude is held constant at the anchor row's latitude.
    """
    row0 = df_country[df_country["week_id"] == int(start_week_id)]
    if row0.empty:
        raise ValueError(f"week_id={start_week_id} not found.")
    lat = float(row0.iloc[0]["latitude"])

    rows = []
    for h in range(1, horizon + 1):
        wk = int(start_week_id) + h
        seas = _seasonal_for_week(wk, lat)
        rows.append(
            dict(
                week_id=wk,
                latitude=lat,
                **seas,
            )
        )
    return pd.DataFrame(rows)


def _compute_prev_anchors_at_t0(
    df_country: pd.DataFrame,
    week_id: int,
    trend_window: int = 4,
) -> Tuple[float, float]:
    """
    Compute prev_ma_4w and prev_slope_4w at anchor time (using history up to week_id, inclusive).
    """
    g = df_country[df_country["week_id"] <= int(week_id)].copy()
    y = g["covid_19_prevalence"].astype(float).to_numpy()
    ma = float(np.mean(y[-trend_window:])) if y.size >= 1 else 0.0
    yy = y[-trend_window:]
    k = len(yy)
    if k < 2:
        slope = 0.0
    else:
        x = np.arange(k, dtype=float)
        xm, ym = x.mean(), yy.mean()
        denom = np.sum((x - xm) ** 2)
        slope = (
            float(np.sum((x - xm) * (yy - ym)) / denom) if denom > 0 else 0.0
        )
    return ma, slope


def _recompute_interactions_inplace(
    step_raw_features: dict,
    norm_stats: dict,
    policy_base_names: Sequence[str],
) -> None:
    """
    Placeholder for interaction recomputation. Interactions are OFF in this config.
    Kept for API compatibility with earlier experiments.
    """
    return


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------


def predict_horizon_omi(
    country_iso3: str,
    week_id_or_idx: int,
    policy_sliders: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """
    Omicron horizon-aware forecast.

    Parameters
    ----------
    country_iso3:
        Country ISO3 code (e.g., "USA").

    week_id_or_idx:
        Either the actual week_id in the history, or a 0-based positional
        index into that country's rows.

    policy_sliders:
        Either:
        - a single length-3 triple (stringency, face coverings, testing/tracing),
          interpreted as constant over the horizon; OR
        - an array/list of shape (H, 3) specifying step-varying policy levels.

        These levels must be in the same (normalized) scale as the training
        data (0–1 in your current setup).

    Returns
    -------
    pandas.DataFrame
        Columns:
            country_iso3, week_id, q10, q50, q90, dq10, dq50, dq90
        where dq* are deltas relative to the first horizon step (h=1).
    """
    country = str(country_iso3)
    g = df[df["country_iso3"] == country].sort_values("week_id").reset_index(
        drop=True
    )
    if g.empty:
        raise ValueError(f"No data for {country}")

    # Resolve anchor week_id
    week_id = _resolve_week_id_for_country(g, week_id_or_idx)

    # ---- constants at anchor (kept across horizon) ----
    # centered mains (time_c uses train center "time" which equals week_id in training)
    time_center = float(_center_means.get("time", 0.0))
    time_c = float(week_id) - time_center

    # Known controls per step (seasonality ROLLS across H; latitude constant)
    fut_ctrl = _extract_knowns_rolling(g, week_id, _H)

    # Anchors from history at t0
    prev_ma, prev_slope = _compute_prev_anchors_at_t0(
        g, week_id, trend_window=4)
    prev_ma_c = prev_ma - float(_center_means.get("prev_ma_4w", prev_ma))
    prev_slope_c = prev_slope - float(
        _center_means.get("prev_slope_4w", prev_slope)
    )

    # Policy windows (RAW) & policy lags (RAW) from history + future path
    windows = _build_future_policy_windows(g, week_id, policy_sliders, _H)
    lags = _build_future_policy_lags(g, week_id, policy_sliders, _H, LAG_STEPS)

    # Build raw → normalized features per step (and interactions)
    policy_base_names = _policy_base_names()
    feature_order = list(_feature_order)
    step_rows = []

    for h in range(_H):
        wkrow = fut_ctrl.iloc[h]

        # RAW values first (for correct normalization)
        sr = {
            # ---- anchor mains (centered where *_c) ----
            "prev_ma_4w_c": prev_ma_c,
            "prev_slope_4w_c": prev_slope_c,
            "time_c": time_c,
            # step-known controls (seasonality varies with week; latitude constant)
            "latitude": float(wkrow["latitude"]),
            "sine_seasonality": float(wkrow["sine_seasonality"]),
            "cosine_seasonality": float(wkrow["cosine_seasonality"]),
            "sine_seasonality_x_latitude": float(
                wkrow["sine_seasonality_x_latitude"]
            ),
            "cosine_seasonality_x_latitude": float(
                wkrow["cosine_seasonality_x_latitude"]
            ),
        }

        # ------------------ (A) GATE: add lag_gate raw (will be z-scored) ------------------
        sr["lag_gate"] = 0.0  # serve with gate OFF

        # ------------------ (B) policy Koyck windows ------------------
        for name in policy_base_names:
            sr[name] = float(windows[name][h])

        # ------------------ (C) discrete lag features (RAW) ------------------
        for p in POLICY_COLS:
            for k in LAG_STEPS:
                lname = f"{p}_lag{k}"
                # In practice all lag names exist in lags
                sr[lname] = float(lags[lname][h]) if lname in lags else _feature_norm[
                    lname
                ]["mean"]

        # --- Z-score everything present ---
        z = {
            k: _normalize_scalar(k, v, _feature_norm)
            for k, v in sr.items()
            if k in _feature_norm
        }

        # ------------------ (D) GATE in Z-space: zero all lag features ------------------
        for p in POLICY_COLS:
            for k in LAG_STEPS:
                lname = f"{p}_lag{k}"
                if lname in z:
                    z[lname] = 0.0

        # Optionally, recompute interactions (no-op here)
        _recompute_interactions_inplace(sr, _feature_norm, policy_base_names)

        # Ensure any (future) normalized interaction features present (no-ops here)
        for k in feature_order:
            if "_x_" in k and k not in z and k in sr:
                z[k] = float(sr[k])

        # Order to match training
        step_rows.append([z.get(n, 0.0) for n in feature_order])

    # [B=1, H, D] – run the scripted model via the common helper
    decoder_cont = np.asarray(step_rows, dtype=np.float32).reshape(1, _H, -1)
    y = run_variant_model(VARIANT, decoder_cont).squeeze(0)  # [H, 3]

    weeks = [int(week_id) + h for h in range(1, _H + 1)]
    out = pd.DataFrame(
        {
            "country_iso3": country,
            "week_id": weeks,
            "q10": y[:, 0],
            "q50": y[:, 1],
            "q90": y[:, 2],
        }
    )

    # === Δ over horizon relative to first step (h=1) ===
    base_q = out.loc[out.index.min(), ["q10", "q50", "q90"]].to_numpy(
        dtype=float
    )
    deltas = out[["q10", "q50", "q90"]].to_numpy(dtype=float) - base_q
    out["dq10"] = deltas[:, 0]
    out["dq50"] = deltas[:, 1]
    out["dq90"] = deltas[:, 2]

    return out


def compare_horizons_omi(
    country_iso3: str,
    week_id_or_idx: int,
    scenario_sliders: Union[Sequence[float], np.ndarray],
) -> pd.DataFrame:
    """
    Compare baseline vs scenario for the q50 trajectory.

    Baseline = observed policy path from the data.
    Scenario = user-specified policy sliders.

    Parameters
    ----------
    country_iso3:
        Country ISO3 code.

    week_id_or_idx:
        Either a true week_id or a 0-based positional index for that country.

    scenario_sliders:
        Scenario path (3,) or (H,3) specifying future policy levels in
        model-native scale (0–1).

    Returns
    -------
    pandas.DataFrame
        Columns:
            country_iso3, week_id, q50_baseline, q50_scenario, pct_change_q50
    """
    country = str(country_iso3)
    g = df[df["country_iso3"] == country].sort_values("week_id").reset_index(
        drop=True
    )
    if g.empty:
        raise ValueError(f"No data for {country}")

    week_id = _resolve_week_id_for_country(g, week_id_or_idx)

    # Observed baseline policy path (t0+1..t0+H)
    baseline_path = _future_slider_path_from_data(g, week_id, _H)

    preds_baseline = predict_horizon_omi(country_iso3, week_id, baseline_path)
    preds_scenario = predict_horizon_omi(
        country_iso3, week_id, scenario_sliders)

    df_cmp = preds_baseline.merge(
        preds_scenario,
        on=["country_iso3", "week_id"],
        suffixes=("_baseline", "_scenario"),
    )

    # % change in the q50 level (scenario vs baseline)
    denom = df_cmp["q50_baseline"].replace({0.0: np.nan})
    df_cmp["pct_change_q50"] = (
        100.0 * (df_cmp["q50_scenario"] - df_cmp["q50_baseline"]) / denom
    )

    return df_cmp[
        [
            "country_iso3",
            "week_id",
            "q50_baseline",
            "q50_scenario",
            "pct_change_q50",
        ]
    ]
