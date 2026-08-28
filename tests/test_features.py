import pathlib,sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).parents[1]/"src"))
from features import add_wr_features,safe_div
from similarities import find_similar_players
from model import build_rankings


def sample():
    return pd.DataFrame([{"player":"A","draft_year":2024,"age":21,"height_in":72,"weight_lb":200,"games":10,"receptions":50,"rec_yards":800,"rec_tds":8,"team_pass_yards":3200,"team_pass_tds":32,"forty":4.45,"vertical":38,"broad_jump":125}])


def test_wr_feature_math():
    row=add_wr_features(sample()).iloc[0]; assert row.yards_per_game==pytest.approx(80); assert row.yards_per_reception==pytest.approx(16); assert row.yard_share==pytest.approx(.25); assert row.td_share==pytest.approx(.25)


def test_zero_denominators_stay_finite():
    data=sample(); data.loc[0,["games","receptions","team_pass_yards","team_pass_tds","height_in","forty"]]=0; result=add_wr_features(data); assert np.isfinite(result.select_dtypes("number")).all().all()


def test_missing_columns_raise_helpful_error():
    with pytest.raises(ValueError,match="missing required columns"): add_wr_features(pd.DataFrame({"player":["A"]}))


def test_similarity_excludes_selected_player():
    rows=[]
    for i in range(4): row=sample().iloc[0].to_dict(); row["player"]=chr(65+i); row["rec_yards"]+=i*40; row["nfl_value"]=1000+i*100; row["draft_pick"]=20+i; rows.append(row)
    data=add_wr_features(pd.DataFrame(rows)); result=find_similar_players(data,"A",2); assert len(result)==2 and "A" not in result.player.tolist()


def test_sleeper_rank_is_within_draft_class():
    data=pd.DataFrame({"player":["A","B","C","D"],"draft_year":[2023,2023,2024,2024],"draft_pick":[10,30,5,20]}); ranked=build_rankings(data,np.array([1,3,4,2])); a=ranked[ranked.player=="A"].iloc[0]; assert a.market_rank==1 and a.model_rank==2 and a.sleeper_score==-1

