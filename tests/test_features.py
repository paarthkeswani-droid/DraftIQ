import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from features import add_wr_features


def test_wr_feature_math():
    df=pd.DataFrame([{"age":21,"height_in":72,"weight_lb":200,"games":10,"receptions":50,"rec_yards":800,"rec_tds":8,"team_pass_yards":3200,"team_pass_tds":32,"forty":4.45,"vertical":38,"broad_jump":125}])
    out=add_wr_features(df)
    assert round(out.loc[0,"yards_per_game"],1)==80.0
    assert round(out.loc[0,"yards_per_reception"],1)==16.0
    assert round(out.loc[0,"yard_share"],2)==0.25
    assert round(out.loc[0,"td_share"],2)==0.25
