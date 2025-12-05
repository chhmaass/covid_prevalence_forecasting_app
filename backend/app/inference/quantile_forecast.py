"""
Quantile forecast inference endpoint.

This module exposes:

    POST /v1/quantile_forecast

It:
- parses a ForecastRequest payload,
- runs both pre-Omicron and Omicron models,
- blends their outputs according to variant_blend weights,
- returns:
    - blended quantile forecasts per horizon,
    - blended baseline vs scenario deltas,
    - raw pre-/Omicron outputs (for debug tabs in the UI),
    - recent history block (prevalence + policies) for the UI.

NOTE:
    - week_id_or_idx in the request is interpreted as a GLOBAL week index.
    - pre-Omicron sees local_week = global_week (clamped inside its wrapper).
    - Omicron sees local_week = global_week - OMI_GLOBAL_START.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.config import (
    H_MODEL,
    DEFAULT_QUANTILES,
    OMI_GLOBAL_START,
    TRANSITION_WIDTH,
)
from app.schemas import ForecastRequest
from app.models.pre_omicron import (
    predict_horizon_pre,
    compare_horizons_pre,
    df as df_pre,
)
from app.models.omicron import (
    predict_horizon_omi,
    compare_horizons_omi,
    df as df_omi,
)

router = APIRouter()


# ---------------------------------------------------------------------
# Global history table: pre + omicron with GLOBAL week index
# ---------------------------------------------------------------------

# pre-Omicron: assume its week_id is already global 0..95
_pre_hist = df_pre.copy()
_pre_hist["global_week"] = _pre_hist["week_id"].astype(int)

# Omicron: local 0..82 → global (OMI_GLOBAL_START..OMI_GLOBAL_START+82)
_omi_hist = df_omi.copy()
_omi_hist["global_week"] = _omi_hist["week_id"].astype(
    int) + int(OMI_GLOBAL_START)

# Unified history df used for the UI history block
DF_HISTORY = pd.concat([_pre_hist, _omi_hist], ignore_index=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _normalize_blend_weights(
    w_pre: float, w_omi: float
) -> tuple[float, float]:
    total = float(w_pre) + float(w_omi)
    if total <= 0.0:
        # sensible default: equal weights if user gave both 0
        return 0.5, 0.5
    return float(w_pre) / total, float(w_omi) / total


def _default_time_based_weights(global_week: int) -> tuple[float, float]:
    """
    Compute pre/omicron blend weights as a function of global_week.

    - Before the transition window: all pre
    - After the window: all omicron
    - Inside the window: linear blend
    """
    g = float(global_week)
    start = float(OMI_GLOBAL_START) - TRANSITION_WIDTH / 2.0
    end = float(OMI_GLOBAL_START) + TRANSITION_WIDTH / 2.0

    if g <= start:
        return 1.0, 0.0
    if g >= end:
        return 0.0, 1.0

    # Linear interpolation in [start, end]
    t = (g - start) / (end - start)  # 0 → 1 across the transition band
    w_omi = t
    w_pre = 1.0 - t
    return w_pre, w_omi


def _validate_policy_sliders(policy_sliders: list[list[float]]) -> None:
    """
    Basic shape validation for policy sliders: expect H_MODEL x 3.
    """
    if not isinstance(policy_sliders, list) or len(policy_sliders) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="policy_sliders must be a non-empty list.",
        )

    h = len(policy_sliders)
    if h != H_MODEL:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"policy_sliders must have length H={H_MODEL}; "
                f"got {h}."
            ),
        )

    for row in policy_sliders:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Each policy_sliders row must be a list/tuple of length 3 "
                    "(stringency, face coverings, testing/tracing)."
                ),
            )


def _blend_predictions(
    pre_pred: Optional[pd.DataFrame],
    omi_pred: Optional[pd.DataFrame],
    w_omi: float,
) -> Optional[pd.DataFrame]:
    """
    Blend q10/q50/q90 from pre- and Omicron model outputs.
    Returns DataFrame with columns: h, week_id, q10, q50, q90.

    week_id is always on the GLOBAL axis after omicron shifting.
    """
    if pre_pred is None and omi_pred is None:
        return None

    # Case 1: only one variant has predictions -> just pass it through
    if (pre_pred is None or pre_pred.empty) and (omi_pred is not None and not omi_pred.empty):
        d = omi_pred.sort_values("week_id").reset_index(drop=True)
        h = len(d)
        if h == 0:
            return None
        return pd.DataFrame(
            {
                "h": np.arange(1, h + 1, dtype=int),
                "week_id": d["week_id"].astype(int).to_numpy(),
                "q10": d["q10"].to_numpy(dtype=float),
                "q50": d["q50"].to_numpy(dtype=float),
                "q90": d["q90"].to_numpy(dtype=float),
            }
        )

    if (omi_pred is None or omi_pred.empty) and (pre_pred is not None and not pre_pred.empty):
        d = pre_pred.sort_values("week_id").reset_index(drop=True)
        h = len(d)
        if h == 0:
            return None
        return pd.DataFrame(
            {
                "h": np.arange(1, h + 1, dtype=int),
                "week_id": d["week_id"].astype(int).to_numpy(),
                "q10": d["q10"].to_numpy(dtype=float),
                "q50": d["q50"].to_numpy(dtype=float),
                "q90": d["q90"].to_numpy(dtype=float),
            }
        )

    # Case 2: both present -> do the original weighted blend
    w_omi = float(w_omi)
    w_pre = max(0.0, 1.0 - w_omi)

    # Sort and determine horizon
    if pre_pred is not None and not pre_pred.empty:
        pre_sorted = pre_pred.sort_values("week_id").reset_index(drop=True)
        h_pre = len(pre_sorted)
    else:
        pre_sorted = None
        h_pre = 0

    if omi_pred is not None and not omi_pred.empty:
        omi_sorted = omi_pred.sort_values("week_id").reset_index(drop=True)
        h_omi = len(omi_sorted)
    else:
        omi_sorted = None
        h_omi = 0

    horizon = max(h_pre, h_omi)
    if horizon == 0:
        return None

    # Choose week_ids on the GLOBAL axis (prefer Omicron if available)
    if omi_sorted is not None and "week_id" in omi_sorted.columns:
        week_ids = omi_sorted["week_id"].to_numpy(dtype=int)
    elif pre_sorted is not None and "week_id" in pre_sorted.columns:
        week_ids = pre_sorted["week_id"].to_numpy(dtype=int)
    else:
        week_ids = np.arange(1, horizon + 1, dtype=int)

    def extract_q(df: Optional[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if df is None or df.empty:
            return (
                np.full(horizon, np.nan, dtype=float),
                np.full(horizon, np.nan, dtype=float),
                np.full(horizon, np.nan, dtype=float),
            )
        d = df.sort_values("week_id").reset_index(drop=True)
        q10 = d["q10"].to_numpy(dtype=float)
        q50 = d["q50"].to_numpy(dtype=float)
        q90 = d["q90"].to_numpy(dtype=float)
        if len(q10) < horizon:
            pad = horizon - len(q10)
            q10 = np.concatenate([q10, np.full(pad, np.nan)])
            q50 = np.concatenate([q50, np.full(pad, np.nan)])
            q90 = np.concatenate([q90, np.full(pad, np.nan)])
        return q10, q50, q90

    pre_q10, pre_q50, pre_q90 = extract_q(pre_sorted)
    omi_q10, omi_q50, omi_q90 = extract_q(omi_sorted)

    # Weighted blend (NaNs propagate if both are NaN; but we already handled the
    # "only one variant present" case above)
    q10_blend = w_pre * pre_q10 + w_omi * omi_q10
    q50_blend = w_pre * pre_q50 + w_omi * omi_q50
    q90_blend = w_pre * pre_q90 + w_omi * omi_q90

    df_blend = pd.DataFrame(
        {
            "h": np.arange(1, horizon + 1, dtype=int),
            "week_id": week_ids[:horizon],
            "q10": q10_blend,
            "q50": q50_blend,
            "q90": q90_blend,
        }
    )
    return df_blend


def _blend_compare(
    pre_cmp: Optional[pd.DataFrame],
    omi_cmp: Optional[pd.DataFrame],
    w_omi: float,
) -> Optional[pd.DataFrame]:
    """
    Blend baseline vs scenario q50 and pct_change_q50 across variants.

    Returns DataFrame with columns:
      h, week_id (GLOBAL),
      q50_baseline_pre, q50_scenario_pre, pct_change_q50_pre,
      q50_baseline_omi, q50_scenario_omi, pct_change_q50_omi,
      q50_baseline_blend, q50_scenario_blend, pct_change_q50_blend
    """
    if pre_cmp is None and omi_cmp is None:
        return None

    # Case 1: only Omicron present -> use it for blended, pre = NaN
    if (pre_cmp is None or pre_cmp.empty) and (omi_cmp is not None and not omi_cmp.empty):
        d = omi_cmp.sort_values("week_id").reset_index(drop=True)
        h = len(d)
        if h == 0:
            return None
        b = d["q50_baseline"].to_numpy(dtype=float)
        s = d["q50_scenario"].to_numpy(dtype=float)
        p = d["pct_change_q50"].to_numpy(dtype=float)
        week_ids = d["week_id"].astype(int).to_numpy()
        return pd.DataFrame(
            {
                "h": np.arange(1, h + 1, dtype=int),
                "week_id": week_ids,
                "q50_baseline_pre": np.full(h, np.nan, dtype=float),
                "q50_scenario_pre": np.full(h, np.nan, dtype=float),
                "pct_change_q50_pre": np.full(h, np.nan, dtype=float),
                "q50_baseline_omi": b,
                "q50_scenario_omi": s,
                "pct_change_q50_omi": p,
                "q50_baseline_blend": b,
                "q50_scenario_blend": s,
                "pct_change_q50_blend": p,
            }
        )

    # Case 2: only Pre present -> use it for blended, omicron = NaN
    if (omi_cmp is None or omi_cmp.empty) and (pre_cmp is not None and not pre_cmp.empty):
        d = pre_cmp.sort_values("week_id").reset_index(drop=True)
        h = len(d)
        if h == 0:
            return None
        b = d["q50_baseline"].to_numpy(dtype=float)
        s = d["q50_scenario"].to_numpy(dtype=float)
        p = d["pct_change_q50"].to_numpy(dtype=float)
        week_ids = d["week_id"].astype(int).to_numpy()
        return pd.DataFrame(
            {
                "h": np.arange(1, h + 1, dtype=int),
                "week_id": week_ids,
                "q50_baseline_pre": b,
                "q50_scenario_pre": s,
                "pct_change_q50_pre": p,
                "q50_baseline_omi": np.full(h, np.nan, dtype=float),
                "q50_scenario_omi": np.full(h, np.nan, dtype=float),
                "pct_change_q50_omi": np.full(h, np.nan, dtype=float),
                "q50_baseline_blend": b,
                "q50_scenario_blend": s,
                "pct_change_q50_blend": p,
            }
        )

    # Case 3: both present -> original weighted blend
    w_omi = float(w_omi)
    w_pre = max(0.0, 1.0 - w_omi)

    if pre_cmp is not None and not pre_cmp.empty:
        pre_sorted = pre_cmp.sort_values("week_id").reset_index(drop=True)
        h_pre = len(pre_sorted)
    else:
        pre_sorted = None
        h_pre = 0

    if omi_cmp is not None and not omi_cmp.empty:
        omi_sorted = omi_cmp.sort_values("week_id").reset_index(drop=True)
        h_omi = len(omi_sorted)
    else:
        omi_sorted = None
        h_omi = 0

    horizon = max(h_pre, h_omi)
    if horizon == 0:
        return None

    # Choose week_ids on GLOBAL axis (prefer Omicron if available)
    if omi_sorted is not None and "week_id" in omi_sorted.columns:
        week_ids = omi_sorted["week_id"].to_numpy(dtype=int)
    elif pre_sorted is not None and "week_id" in pre_sorted.columns:
        week_ids = pre_sorted["week_id"].to_numpy(dtype=int)
    else:
        week_ids = np.arange(1, horizon + 1, dtype=int)

    def extract(
        df: Optional[pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if df is None or df.empty:
            return (
                np.full(horizon, np.nan, dtype=float),
                np.full(horizon, np.nan, dtype=float),
                np.full(horizon, np.nan, dtype=float),
            )
        d = df.sort_values("week_id").reset_index(drop=True)
        b = d["q50_baseline"].to_numpy(dtype=float)
        s = d["q50_scenario"].to_numpy(dtype=float)
        p = d["pct_change_q50"].to_numpy(dtype=float)
        if len(b) < horizon:
            pad = horizon - len(b)
            b = np.concatenate([b, np.full(pad, np.nan)])
            s = np.concatenate([s, np.full(pad, np.nan)])
            p = np.concatenate([p, np.full(pad, np.nan)])
        return b, s, p

    pre_b, pre_s, pre_pct = extract(pre_sorted)
    omi_b, omi_s, omi_pct = extract(omi_sorted)

    # Blended baseline/scenario
    b_blend = w_pre * pre_b + w_omi * omi_b
    s_blend = w_pre * pre_s + w_omi * omi_s

    # Recompute pct_change from blended levels
    denom = np.where(b_blend == 0.0, np.nan, b_blend)
    pct_blend = 100.0 * (s_blend - b_blend) / denom

    df_blend = pd.DataFrame(
        {
            "h": np.arange(1, horizon + 1, dtype=int),
            "week_id": week_ids[:horizon],
            "q50_baseline_pre": pre_b,
            "q50_scenario_pre": pre_s,
            "pct_change_q50_pre": pre_pct,
            "q50_baseline_omi": omi_b,
            "q50_scenario_omi": omi_s,
            "pct_change_q50_omi": omi_pct,
            "q50_baseline_blend": b_blend,
            "q50_scenario_blend": s_blend,
            "pct_change_q50_blend": pct_blend,
        }
    )
    return df_blend


def _df_to_records(df: Optional[pd.DataFrame]) -> Optional[list[Dict[str, Any]]]:
    if df is None:
        return None
    if df.empty:
        return []
    # Convert NaNs to None so JSON is cleaner
    df_clean = df.replace({np.nan: None})
    return df_clean.to_dict(orient="records")


# ---------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------


@router.post("/quantile_forecast")
def quantile_forecast(req: ForecastRequest) -> Dict[str, Any]:
    """
    Main inference endpoint.

    Request (ForecastRequest):
        country_iso3: str
        week_id_or_idx: int (interpreted as GLOBAL week index)
        policy_sliders: List[List[float]] of shape (H,3)
        quantiles: List[float]
        variant_blend: { pre_omicron_weight: float, omicron_weight: float }

    Response:
        {
          "country_iso3": ...,
          "week_id_or_idx": ...,
          "horizon": H,
          "quantiles": [0.1,0.5,0.9],
          "weights": { "pre_omicron": w_pre, "omicron": w_omi },
          "predictions": {
              "blended": [...],
              "pre": [...],
              "omicron": [...],
          },
          "compare": {
              "blended": [...],
              "pre": [...],
              "omicron": [...],
          },
          "history": {
              "week_id": [...],          # GLOBAL weeks
              "prevalence": [...],
              "policies": {...},
          }
        }
    """
    # 1) Validate policy sliders
    _validate_policy_sliders(req.policy_sliders)

    # 2) Interpret week_id_or_idx as GLOBAL week index
    country = req.country_iso3.strip().upper()
    global_week = int(req.week_id_or_idx)

    # 3) Compute or override blend weights
    w_pre_raw = float(req.variant_blend.pre_omicron_weight)
    w_omi_raw = float(req.variant_blend.omicron_weight)

    if w_pre_raw == 0.0 and w_omi_raw == 0.0:
        # If UI sends both 0 → use time-based weights from global_week
        w_pre, w_omi = _default_time_based_weights(global_week)
    else:
        # If user sets explicit weights → normalize them
        w_pre, w_omi = _normalize_blend_weights(w_pre_raw, w_omi_raw)

    # 4) Quantiles sanity check (currently informational only)
    q_req = list(req.quantiles or [])
    if sorted(q_req) != sorted(DEFAULT_QUANTILES):
        # For now we still run with model's native quantiles; the request field
        # is primarily informational to keep the API extensible.
        pass

    # Convert sliders to numpy array for convenience
    scenario_path = np.asarray(req.policy_sliders, dtype=float)

    # 5) Map GLOBAL week to LOCAL week indices per regime
    pre_local_week = global_week
    omi_local_week = global_week - OMI_GLOBAL_START

    pre_pred = pre_cmp = omi_pred = omi_cmp = None

    # We isolate each call so one failing variant doesn't crash the whole endpoint
    try:
        pre_pred = predict_horizon_pre(
            country_iso3=country,
            week_id_or_idx=pre_local_week,
            policy_sliders=scenario_path,
        )
    except Exception:
        pre_pred = None

    try:
        pre_cmp = compare_horizons_pre(
            country_iso3=country,
            week_id_or_idx=pre_local_week,
            scenario_sliders=scenario_path,
        )
    except Exception:
        pre_cmp = None

    # Only call Omicron model when we are at or after its global start
    if omi_local_week >= 0:
        try:
            omi_pred = predict_horizon_omi(
                country_iso3=country,
                week_id_or_idx=omi_local_week,
                policy_sliders=scenario_path,
            )
        except Exception:
            omi_pred = None

        try:
            omi_cmp = compare_horizons_omi(
                country_iso3=country,
                week_id_or_idx=omi_local_week,
                scenario_sliders=scenario_path,
            )
        except Exception:
            omi_cmp = None
    else:
        omi_pred = None
        omi_cmp = None

    if (
        pre_pred is None
        and pre_cmp is None
        and omi_pred is None
        and omi_cmp is None
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Both pre-Omicron and Omicron models failed.",
        )

    # ------------------------------------------------------------------
    # Shift Omicron outputs from LOCAL week axis to GLOBAL axis
    # ------------------------------------------------------------------
    if omi_pred is not None and not omi_pred.empty:
        omi_pred = omi_pred.copy()
        omi_pred["week_id"] = omi_pred["week_id"].astype(
            int) + int(OMI_GLOBAL_START)

    if omi_cmp is not None and not omi_cmp.empty:
        omi_cmp = omi_cmp.copy()
        omi_cmp["week_id"] = omi_cmp["week_id"].astype(
            int) + int(OMI_GLOBAL_START)

    # 6) Blend outputs (on GLOBAL week axis)
    df_blend_pred = _blend_predictions(pre_pred, omi_pred, w_omi)
    df_blend_cmp = _blend_compare(pre_cmp, omi_cmp, w_omi)

    # -----------------------------------------------------
    # 7) Build history block for the UI (from unified DF_HISTORY)
    # -----------------------------------------------------
    g_hist = DF_HISTORY[DF_HISTORY["country_iso3"] == country].sort_values(
        "global_week"
    )

    if df_blend_pred is not None and not df_blend_pred.empty:
        # df_blend_pred['week_id'] is GLOBAL after omicron shift
        anchor_global = int(df_blend_pred["week_id"].iloc[0])
    else:
        # fall back to the requested GLOBAL week index
        anchor_global = global_week

    # Last 32 observations up to anchor global week
    g_hist = g_hist[g_hist["global_week"] <= anchor_global].tail(32)

    if g_hist.empty:
        history_block = {
            "week_id": [],
            "prevalence": [],
            "policies": {
                "covid_19_policy_stringency": [],
                "covid_19_face_covering_policy": [],
                "covid_19_testing_tracing_policy": [],
            },
        }
    else:
        history_block = {
            # expose GLOBAL week index to the frontend
            "week_id": g_hist["global_week"].tolist(),
            "prevalence": g_hist["covid_19_prevalence"].tolist(),
            "policies": {
                "covid_19_policy_stringency": g_hist[
                    "covid_19_policy_stringency"
                ].tolist(),
                "covid_19_face_covering_policy": g_hist[
                    "covid_19_face_covering_policy"
                ].tolist(),
                "covid_19_testing_tracing_policy": g_hist[
                    "covid_19_testing_tracing_policy"
                ].tolist(),
            },
        }

    # 8) Build response JSON
    resp: Dict[str, Any] = {
        "country_iso3": country,
        "week_id_or_idx": global_week,
        "horizon": H_MODEL,
        "quantiles": DEFAULT_QUANTILES,
        "weights": {
            "pre_omicron": w_pre,
            "omicron": w_omi,
        },
        "predictions": {
            "blended": _df_to_records(df_blend_pred),
            "pre": _df_to_records(pre_pred),
            "omicron": _df_to_records(omi_pred),
        },
        "compare": {
            "blended": _df_to_records(df_blend_cmp),
            "pre": _df_to_records(pre_cmp),
            "omicron": _df_to_records(omi_cmp),
        },
        "history": history_block,
    }

    return resp
