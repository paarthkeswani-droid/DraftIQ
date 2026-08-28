"""Join public draft/NFL outcomes to optional college WR prospect features."""
from pathlib import Path
import pandas as pd
from features import add_wr_features, FEATURES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def normalize_name(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def first_existing(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Expected one of these columns: {names}")


def main() -> None:
    draft = pd.read_csv(RAW / "draft_picks.csv")
    seasonal = pd.read_csv(RAW / "seasonal.csv")
    name_col = first_existing(draft, ["pfr_player_name", "player_name", "name"])
    pos_col = first_existing(draft, ["position", "pos"])
    year_col = first_existing(draft, ["season", "draft_year", "year"])
    pick_col = first_existing(draft, ["pick", "overall"])
    draft = draft[draft[pos_col].astype(str).str.upper().eq("WR")].copy()
    draft["player"] = draft[name_col]
    draft["draft_year"] = pd.to_numeric(draft[year_col], errors="coerce")
    draft["draft_pick"] = pd.to_numeric(draft[pick_col], errors="coerce")
    draft["player_key"] = normalize_name(draft["player"])

    # nflverse seasonal files generally include player_display_name and season.
    sname = first_existing(seasonal, ["player_display_name", "player_name", "name"])
    seasonal["player_key"] = normalize_name(seasonal[sname])
    seasonal["season"] = pd.to_numeric(seasonal["season"], errors="coerce")
    merged = seasonal.merge(draft[["player_key", "player", "draft_year", "draft_pick"]], on="player_key", how="inner")
    merged = merged[(merged["season"] >= merged["draft_year"]) & (merged["season"] <= merged["draft_year"] + 2)]

    rec_col = first_existing(merged, ["receptions"])
    yards_col = first_existing(merged, ["receiving_yards"])
    td_col = first_existing(merged, ["receiving_tds"])
    outcomes = merged.groupby(["player_key", "player", "draft_year", "draft_pick"], as_index=False).agg(
        nfl_receptions=(rec_col, "sum"), nfl_rec_yards=(yards_col, "sum"), nfl_rec_tds=(td_col, "sum")
    )
    outcomes["nfl_value"] = outcomes["nfl_rec_yards"] + 20 * outcomes["nfl_receptions"] + 150 * outcomes["nfl_rec_tds"]

    college_path = RAW / "college_wr.csv"
    if college_path.exists():
        college = pd.read_csv(college_path)
        college["player_key"] = normalize_name(college["player"])
        college = add_wr_features(college)
        model = college.merge(outcomes, on=["player_key", "draft_year"], how="inner", suffixes=("", "_draft"))
        if "player_draft" in model:
            model["player"] = model["player"].fillna(model["player_draft"])
        keep = ["player", "draft_year", "draft_pick"] + [c for c in ["college", "conference"] if c in model] + FEATURES + ["nfl_value"]
        model = model[keep]
    else:
        # Market baseline: useful immediately; add college_wr.csv for the full independent sleeper model.
        model = outcomes[["player", "draft_year", "draft_pick", "nfl_value"]].copy()
        model["market_expectation"] = 1 / model["draft_pick"].clip(lower=1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    model.to_csv(PROCESSED / "wr_model_data.csv", index=False)
    print(f"Wrote {len(model):,} WR rows. Full prospect features: {college_path.exists()}")

if __name__ == "__main__":
    main()
