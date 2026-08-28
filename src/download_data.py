"""Download public nflverse draft/player/seasonal data used by DraftIQ."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

URLS = {
    "draft_picks": "https://github.com/nflverse/nfldata/raw/master/data/drafts/draft_picks.csv",
    "players": "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
    "seasonal": "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        print(f"Downloading {name}...")
        df = pd.read_csv(url)
        df.to_csv(RAW / f"{name}.csv", index=False)
        print(f"  {len(df):,} rows")


if __name__ == "__main__":
    main()
