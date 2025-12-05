"""
Common utilities for loading GGHAQR artifacts and preparing inputs.

This module is deliberately variant-agnostic and is shared by the
pre-Omicron and Omicron model wrappers.

Responsibilities:
- Load all per-variant artifacts (scripted model + JSON configs).
- Provide typed, cached accessors to those artifacts.
- Offer helpers to:
    * encode country_iso3 using the label map,
    * construct feature arrays in the canonical feature order,
    * apply Z-normalization using training statistics,
    * run the scripted model safely.

The per-variant wrappers (pre_omicron.py / omicron.py) are expected to:
- Use these helpers to build decoder_cont tensors of shape [B, H, D].
- Implement high-level functions like predict_horizon_* and
  compare_horizons_* building on these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Literal

import json

import numpy as np
import pandas as pd
import torch

from app.config import PRE_DIR, OMI_DIR, SHARED_DIR, H_MODEL, DEFAULT_QUANTILES

Variant = Literal["pre_omicron", "omicron"]


# ---------------------------------------------------------------------
# Dataclasses describing the artifact bundle
# ---------------------------------------------------------------------


@dataclass
class FeatureContract:
    decoder_cont_lastdim: int
    horizon: int
    quantiles: List[float]
    feature_order: List[str]
    raw: Mapping[str, Any]


@dataclass
class ServingSchema:
    time_idx: str
    target: str
    group_ids: List[str]
    max_prediction_length: int
    static_categoricals: List[str]
    time_varying_known_reals: List[str]
    time_varying_unknown_reals: List[str]
    normalization_type: str
    categorical_encoders: Mapping[str, Any]
    constants: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass
class NormStats:
    """
    Z-normalization parameters per feature.

    stats[feature_name] = {"mean": float, "std": float}
    """

    stats: Mapping[str, Mapping[str, float]]


@dataclass
class CenterMeans:
    """
    Means for "centered" features (e.g. prev_ma_4w_c = prev_ma_4w - mean).

    keys are raw feature names (without _c suffix).
    """

    means: Mapping[str, float]


@dataclass
class CountryIndexMap:
    """
    Label map for country_iso3 -> integer index used during training.
    """

    mapping: Mapping[str, int]


@dataclass
class VariantArtifacts:
    """
    Bundle of all artifacts needed to serve a single variant model.
    """

    variant: Variant
    model: torch.jit.ScriptModule
    feature_contract: FeatureContract
    serving_schema: ServingSchema
    norm_stats: NormStats
    center_means: CenterMeans
    country_index_map: CountryIndexMap


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _variant_dir(variant: Variant) -> Path:
    if variant == "pre_omicron":
        return PRE_DIR
    if variant == "omicron":
        return OMI_DIR
    raise ValueError(f"Unknown variant: {variant!r}")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Expected JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_scripted_model(path: Path) -> torch.jit.ScriptModule:
    if not path.is_file():
        raise FileNotFoundError(f"Expected scripted model not found: {path}")
    # Map to CPU by default; the caller can .to(device) if needed.
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------
# Public artifact accessors (cached per variant)
# ---------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_feature_contract(variant: Variant) -> FeatureContract:
    vdir = _variant_dir(variant)
    path = vdir / f"feature_contract_{variant}.json"
    raw = _load_json(path)

    feature_order = raw.get("feature_order", [])
    decoder_cont_lastdim = int(raw["decoder_cont_lastdim"])
    horizon = int(raw["horizon"])
    quantiles = list(raw["quantiles"])

    if len(feature_order) != decoder_cont_lastdim:
        raise ValueError(
            f"Decoder feature dimension mismatch for {variant}: "
            f"decoder_cont_lastdim={decoder_cont_lastdim}, "
            f"len(feature_order)={len(feature_order)}"
        )

    if horizon != H_MODEL:
        # Not fatal, but almost certainly a bug in the setup.
        raise ValueError(
            f"Horizon mismatch for {variant}: "
            f"feature_contract.horizon={horizon}, expected H_MODEL={H_MODEL}"
        )

    # Optional sanity check vs DEFAULT_QUANTILES
    if sorted(quantiles) != sorted(DEFAULT_QUANTILES):
        # Again, not strictly fatal, but keep it strict for safety.
        raise ValueError(
            f"Quantile mismatch for {variant}: "
            f"feature_contract.quantiles={quantiles}, "
            f"expected DEFAULT_QUANTILES={DEFAULT_QUANTILES}"
        )

    return FeatureContract(
        decoder_cont_lastdim=decoder_cont_lastdim,
        horizon=horizon,
        quantiles=quantiles,
        feature_order=feature_order,
        raw=raw,
    )


@lru_cache(maxsize=None)
def get_serving_schema(variant: Variant) -> ServingSchema:
    vdir = _variant_dir(variant)
    path = vdir / f"serving_schema_{variant}.json"
    raw = _load_json(path)

    time_idx = raw["time_idx"]
    target = raw["target"]
    group_ids = list(raw.get("group_ids", []))
    max_prediction_length = int(raw["max_prediction_length"])
    static_categoricals = list(raw.get("static_categoricals", []))
    time_varying_known_reals = list(raw.get("time_varying_known_reals", []))
    time_varying_unknown_reals = list(
        raw.get("time_varying_unknown_reals", []))
    normalization_type = raw.get("normalization", {}).get("type", "")
    categorical_encoders = raw.get("categorical_encoders", {})
    constants = raw.get("constants", {})

    if max_prediction_length != H_MODEL:
        raise ValueError(
            f"max_prediction_length mismatch for {variant}: "
            f"{max_prediction_length} vs H_MODEL={H_MODEL}"
        )

    fc = get_feature_contract(variant)
    if time_varying_known_reals != fc.feature_order:
        raise ValueError(
            f"time_varying_known_reals does not match feature_order for {variant}. "
            f"Check feature_contract vs serving_schema."
        )

    return ServingSchema(
        time_idx=time_idx,
        target=target,
        group_ids=group_ids,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        normalization_type=normalization_type,
        categorical_encoders=categorical_encoders,
        constants=constants,
        raw=raw,
    )


@lru_cache(maxsize=None)
def get_norm_stats(variant: Variant) -> NormStats:
    vdir = _variant_dir(variant)
    path = vdir / f"feature_norm_stats_{variant}.json"
    raw = _load_json(path)

    fc = get_feature_contract(variant)
    missing = [f for f in fc.feature_order if f not in raw]
    if missing:
        raise ValueError(
            f"Normalization stats missing entries for {variant}: {missing}"
        )

    return NormStats(stats=raw)


@lru_cache(maxsize=None)
def get_center_means(variant: Variant) -> CenterMeans:
    vdir = _variant_dir(variant)
    path = vdir / f"center_means_{variant}.json"
    raw = _load_json(path)
    return CenterMeans(means=raw)


@lru_cache(maxsize=None)
def get_country_index_map(variant: Variant) -> CountryIndexMap:
    vdir = _variant_dir(variant)
    path = vdir / f"country_index_map_{variant}.json"
    raw = _load_json(path)
    return CountryIndexMap(mapping=raw)


@lru_cache(maxsize=None)
def get_scripted_model(variant: Variant) -> torch.jit.ScriptModule:
    vdir = _variant_dir(variant)
    path = vdir / f"model_scripted_{variant}.pt"
    return _load_scripted_model(path)


@lru_cache(maxsize=None)
def get_variant_artifacts(variant: Variant) -> VariantArtifacts:
    """
    Convenience accessor that returns all per-variant artifacts as a bundle.
    """
    return VariantArtifacts(
        variant=variant,
        model=get_scripted_model(variant),
        feature_contract=get_feature_contract(variant),
        serving_schema=get_serving_schema(variant),
        norm_stats=get_norm_stats(variant),
        center_means=get_center_means(variant),
        country_index_map=get_country_index_map(variant),
    )


# ---------------------------------------------------------------------
# Country encoding
# ---------------------------------------------------------------------


def encode_country_iso3(
    country_iso3: str,
    variant: Variant,
    *,
    allow_fallback_to_shared: bool = False,
) -> int:
    """
    Map ISO3 code to the integer index used during training.

    Parameters
    ----------
    country_iso3:
        Country code as passed by the client (e.g. "USA").

    variant:
        "pre_omicron" or "omicron".

    allow_fallback_to_shared:
        If True and the country is not found in the variant-specific map,
        attempt to look up a shared mapping in SHARED_DIR (e.g. for future
        extensions). Currently, this is mostly a placeholder for possible
        future behaviour.

    Returns
    -------
    int
        Integer label index as used by the model.

    Raises
    ------
    KeyError
        If the code cannot be mapped.
    """
    code = country_iso3.strip().upper()
    vimap = get_country_index_map(variant).mapping

    if code in vimap:
        return int(vimap[code])

    if allow_fallback_to_shared:
        shared_path = SHARED_DIR / "country_index_map_shared.json"
        if shared_path.is_file():
            shared_map = _load_json(shared_path)
            if code in shared_map:
                return int(shared_map[code])

    raise KeyError(
        f"Unknown country_iso3 {code!r} for variant {variant}. "
        f"Make sure it exists in country_index_map_{variant}.json."
    )


# ---------------------------------------------------------------------
# Feature preparation & normalization
# ---------------------------------------------------------------------


def z_normalize_features(
    features: np.ndarray,
    feature_order: List[str],
    norm_stats: NormStats,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Apply Z-normalization to features in-place order [*, D].

    Parameters
    ----------
    features:
        Array of shape (..., D) where D == len(feature_order).
        This is assumed to already be in the canonical feature order.

    feature_order:
        Ordered list of feature names.

    norm_stats:
        Normalization statistics per feature.

    eps:
        Small constant to avoid division by zero if std ≈ 0.

    Returns
    -------
    np.ndarray
        Normalized features with the same shape and dtype float32.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.shape[-1] != len(feature_order):
        raise ValueError(
            f"Feature dimension mismatch: last dim={x.shape[-1]}, "
            f"expected {len(feature_order)}"
        )

    stats = norm_stats.stats
    for j, fname in enumerate(feature_order):
        s = stats.get(fname)
        if s is None:
            raise KeyError(
                f"Missing normalization stats for feature {fname!r}")
        mean = float(s.get("mean", 0.0))
        std = float(s.get("std", 1.0))
        if std == 0.0:
            std = eps
        x[..., j] = (x[..., j] - mean) / std

    return x.astype(np.float32, copy=False)


def build_feature_array_from_frame(
    df: pd.DataFrame,
    variant: Variant,
) -> np.ndarray:
    """
    Build a normalized feature array [N, D] from a pandas DataFrame.

    This helper expects that `df` already contains all the *model-ready*
    continuous features as columns, i.e. the names in feature_order.

    It will:
    - select columns in canonical feature_order,
    - cast to float32,
    - apply Z-normalization using training stats.

    Parameters
    ----------
    df:
        DataFrame with at least the columns in `feature_order`.

    variant:
        "pre_omicron" or "omicron".

    Returns
    -------
    np.ndarray
        Normalized features of shape [N, D] ready to be reshaped to [B, H, D].
    """
    artifacts = get_variant_artifacts(variant)
    fc = artifacts.feature_contract
    stats = artifacts.norm_stats

    missing_cols = [c for c in fc.feature_order if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required feature columns for {variant}: "
            f"{missing_cols}"
        )

    # Select in the canonical order
    arr = df[fc.feature_order].to_numpy(dtype=np.float32)
    arr = z_normalize_features(arr, fc.feature_order, stats)
    return arr


def reshape_features_to_decoder_cont(
    features: np.ndarray,
    *,
    horizon: Optional[int] = None,
) -> np.ndarray:
    """
    Reshape a flat [N, D] feature matrix into [B, H, D] decoder_cont.

    This helper assumes that:
    - N == B * H
    - rows are ordered by horizon within batch.

    For most serving use-cases here, B == 1 and N == H (a single
    (H, D) path), so this function will simply add the batch dimension.

    Parameters
    ----------
    features:
        Array of shape [N, D].

    horizon:
        Horizon length H. If None, defaults to H_MODEL.

    Returns
    -------
    np.ndarray
        Decoder_cont array of shape [B, H, D].
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D features [N, D], got shape {x.shape}"
        )

    N, D = x.shape
    H = int(horizon or H_MODEL)
    if N % H != 0:
        raise ValueError(
            f"Cannot reshape features of length N={N} into batches of horizon H={H}."
        )

    B = N // H
    return x.reshape(B, H, D)


# ---------------------------------------------------------------------
# Model invocation
# ---------------------------------------------------------------------


def run_variant_model(
    variant: Variant,
    decoder_cont: np.ndarray,
    *,
    device: Optional[torch.device] = None,
    no_grad: bool = True,
) -> np.ndarray:
    """
    Run the scripted model for a given variant on decoder_cont.

    Parameters
    ----------
    variant:
        "pre_omicron" or "omicron".

    decoder_cont:
        Feature tensor as numpy array of shape [B, H, D] or [H, D].
        If 2D, a batch dimension B=1 will be added automatically.

    device:
        Optional torch.device to move the model + inputs to. If None,
        uses CPU.

    no_grad:
        Whether to disable gradient tracking (recommended for inference).

    Returns
    -------
    np.ndarray
        Model outputs as a numpy array, typically [B, H, Q] where
        Q == len(quantiles) from the feature contract.

        The exact semantics (e.g. whether these are already in prevalence
        space vs logits) are determined by the training/export code.
    """
    artifacts = get_variant_artifacts(variant)
    model = artifacts.model

    x = np.asarray(decoder_cont, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, ...]  # [H, D] -> [1, H, D]
    elif x.ndim != 3:
        raise ValueError(
            f"decoder_cont must be 2D or 3D, got shape {x.shape}"
        )

    # Torch tensor
    t = torch.from_numpy(x)
    if device is not None:
        model = model.to(device)
        t = t.to(device)

    with torch.no_grad() if no_grad else torch.enable_grad():
        out = model(t)

    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    # If the scripted model returns a tuple or dict, the wrapper calling
    # run_variant_model is responsible for interpreting that structure.
    raise TypeError(
        "Scripted model returned a non-tensor output. "
        "Adapt run_variant_model to handle the custom output structure."
    )
