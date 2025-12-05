from pydantic import BaseModel, Field
from typing import List

class VariantBlend(BaseModel):
    pre_omicron_weight: float = Field(..., ge=0, le=1)
    omicron_weight: float = Field(..., ge=0, le=1)

class ForecastRequest(BaseModel):
    country_iso3: str
    week_id_or_idx: int
    policy_sliders: List[List[float]]
    quantiles: List[float] = [0.1, 0.5, 0.9]
    variant_blend: VariantBlend
