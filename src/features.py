"""Validated wide receiver feature engineering for DraftIQ."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "age", "height_in", "weight_lb", "rec_per_game", "yards_per_game",
    "yards_per_reception", "yard_share", "td_share", "forty", "vertical",
    "broad_jump", "bmi", "speed_score",
]

REQUIRED_COLLEGE_COLUMNS = {
    "player", "draft_year", "age", "height_in", "weight_lb", "games",
    "receptions", "rec_yards", "rec_tds", "team_pass_yards",
    "team_pass_tds", "forty", "vertical", "broad_jump",
}


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Vectorized division with stable zero-denominator behavior."""
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    numerator = pd.to_numeric(numerator, errors="coerce")
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def validate_college_data(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLLEGE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"College WR data is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("College WR data contains no rows")
    for column in ("games", "height_in", "weight_lb", "forty"):
        if (pd.to_numeric(frame[column], errors="coerce").fillna(0) < 0).any():
            raise ValueError(f"{column} cannot be negative")


def add_wr_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with production-share and athletic features."""
    validate_college_data(frame)
    out = frame.copy()
    numeric = sorted(REQUIRED_COLLEGE_COLUMNS - {"player"})
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out["rec_per_game"] = safe_div(out["receptions"], out["games"])
    out["yards_per_game"] = safe_div(out["rec_yards"], out["games"])
    out["yards_per_reception"] = safe_div(out["rec_yards"], out["receptions"])
    out["yard_share"] = safe_div(out["rec_yards"], out["team_pass_yards"])
    out["td_share"] = safe_div(out["rec_tds"], out["team_pass_tds"])
    out["bmi"] = safe_div(out["weight_lb"] * 703, out["height_in"] ** 2)
    out["speed_score"] = safe_div(out["weight_lb"] * 200, out["forty"] ** 4)
    return out

