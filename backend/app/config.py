from pathlib import Path

# Base directory = backend/ (where this app/ folder lives)
BASE_DIR = Path(__file__).resolve().parent.parent

# Forecast horizon length
H_MODEL = 12

# Variant timeline globals (global week index)
OMI_GLOBAL_START = 96          # pre covers global weeks 0..95, omicron starts at 96
TRANSITION_WIDTH = 12          # smooth transition width (in global weeks)
MAX_GLOBAL_WEEK = 178          # 0..178 inclusive = 179 weeks = 96 + 83

# Artifacts and shared resources live under backend/artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PRE_DIR = ARTIFACTS_DIR / "pre_omicron"
OMI_DIR = ARTIFACTS_DIR / "omicron"

# Optional shared dir under backend
SHARED_DIR = BASE_DIR / "shared"

# Default quantiles used throughout
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
