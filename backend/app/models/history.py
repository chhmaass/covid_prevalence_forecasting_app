"""Shared global history for both regime-specific inference models."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import BASE_DIR, OMI_GLOBAL_START


@lru_cache(maxsize=1)
def load_global_history() -> pd.DataFrame:
    """Load both processed regimes on one continuous global week axis."""
    pre_path = Path(
        os.getenv(
            "PRE_OMICRON_DATA_PATH",
            str(BASE_DIR / "data" / "df_final_pre_omicron.csv"),
        )
    )
    omi_path = Path(
        os.getenv(
            "OMICRON_DATA_PATH",
            str(BASE_DIR / "data" / "df_final_omicron.csv"),
        )
    )

    missing = [str(path) for path in (pre_path, omi_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"History CSV file(s) not found: {', '.join(missing)}")

    pre = pd.read_csv(pre_path)
    omi = pd.read_csv(omi_path)
    omi = omi.copy()
    omi["week_id"] = omi["week_id"].astype(int) + int(OMI_GLOBAL_START)

    history = pd.concat([pre, omi], ignore_index=True)
    duplicates = history.duplicated(["country_iso3", "week_id"])
    if duplicates.any():
        raise ValueError("Global history contains duplicate country/week rows.")

    return history.sort_values(["country_iso3", "week_id"]).reset_index(drop=True)
