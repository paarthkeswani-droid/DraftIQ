"""Wide receiver feature engineering for DraftIQ."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "age", "height_in", "weight_lb", "rec_per_game", "yards_per_game",
    "yards_per_reception", "yard_share", "td_share", "forty", "vertical",
    "broad_jump", "bmi", "speed_score",
]


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float).div(den.replace(0, np.nan).astype(float))


def add_wr_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rec_per_game"] = safe_div(out["receptions"], out["games"])
    out["yards_per_game"] = safe_div(out["rec_yards"], out["games"])
    out["yards_per_reception"] = safe_div(out["rec_yards"], out["receptions"])
    out["yard_share"] = safe_div(out["rec_yards"], out["team_pass_yards"])
    out["td_share"] = safe_div(out["rec_tds"], out["team_pass_tds"])
    out["bmi"] = safe_div(out["weight_lb"] * 703, out["height_in"] ** 2)
    out["speed_score"] = safe_div(out["weight_lb"] * 200, out["forty"] ** 4)
    return out
