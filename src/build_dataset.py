"""Join draft records and three-year NFL outcomes to college WR features."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import pandas as pd

from features import add_wr_features, FEATURES


def normalize_name(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.casefold().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def first_existing(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Expected one of these columns: {', '.join(names)}")


def build_dataset(draft_path: Path, seasonal_path: Path, college_path: Path) -> pd.DataFrame:
    for path in (draft_path, seasonal_path, college_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
    draft, seasonal, college = map(pd.read_csv, (draft_path, seasonal_path, college_path))
    name_col = first_existing(draft, ["pfr_player_name", "player_name", "name"])
    pos_col = first_existing(draft, ["position", "pos"])
    year_col = first_existing(draft, ["season", "draft_year", "year"])
    pick_col = first_existing(draft, ["pick", "overall"])
    draft = draft[draft[pos_col].astype(str).str.upper().eq("WR")].copy()
    if draft.empty:
        raise ValueError("Draft input contains no wide receivers")
    draft["player"] = draft[name_col]
    draft["draft_year"] = pd.to_numeric(draft[year_col], errors="coerce")
    draft["draft_pick"] = pd.to_numeric(draft[pick_col], errors="coerce")
    draft["player_key"] = normalize_name(draft["player"])

    sname = first_existing(seasonal, ["player_display_name", "player_name", "name"])
    season_col = first_existing(seasonal, ["season", "year"])
    rec_col = first_existing(seasonal, ["receptions"])
    yards_col = first_existing(seasonal, ["receiving_yards", "rec_yards"])
    td_col = first_existing(seasonal, ["receiving_tds", "rec_tds"])
    seasonal["player_key"] = normalize_name(seasonal[sname])
    seasonal["season"] = pd.to_numeric(seasonal[season_col], errors="coerce")
    joined = seasonal.merge(draft[["player_key", "player", "draft_year", "draft_pick"]], on="player_key", how="inner")
    joined = joined[(joined.season >= joined.draft_year) & (joined.season <= joined.draft_year + 2)]
    outcomes = joined.groupby(["player_key", "player", "draft_year", "draft_pick"], as_index=False).agg(
        nfl_receptions=(rec_col, "sum"), nfl_rec_yards=(yards_col, "sum"), nfl_rec_tds=(td_col, "sum"),
    )
    if outcomes.empty:
        raise ValueError("No draft records matched the first three NFL seasons")
    outcomes["nfl_value"] = outcomes.nfl_rec_yards + 20 * outcomes.nfl_receptions + 150 * outcomes.nfl_rec_tds

    college["player_key"] = normalize_name(college["player"])
    college = add_wr_features(college)
    model = college.merge(outcomes, on=["player_key", "draft_year"], how="inner", suffixes=("", "_draft"))
    if model.empty:
        raise ValueError("No college prospect names matched draft/NFL outcome records")
    if "player_draft" in model:
        model["player"] = model["player"].fillna(model["player_draft"])
    identity = ["player", "draft_year", "draft_pick"] + [c for c in ["college", "conference"] if c in model]
    return model[identity + FEATURES + ["nfl_value"]].replace([float("inf"), float("-inf")], pd.NA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=Path("data/raw/draft_picks.csv"))
    parser.add_argument("--seasonal", type=Path, default=Path("data/raw/seasonal.csv"))
    parser.add_argument("--college", type=Path, default=Path("data/raw/college_wr.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/wr_model_data.csv"))
    parser.add_argument("--database", type=Path, default=Path("data/processed/draftiq.sqlite"))
    args = parser.parse_args()
    data = build_dataset(args.draft, args.seasonal, args.college)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    with sqlite3.connect(args.database) as connection:
        data.to_sql("receivers", connection, if_exists="replace", index=False)
    print(f"Wrote {len(data):,} matched WR prospects to {args.output}")


if __name__ == "__main__":
    main()

