"""Find closest historical WR profiles."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from features import FEATURES

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"processed"/"wr_model_data.csv"; OUT=ROOT/"outputs"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--player",required=True); p.add_argument("--top",type=int,default=5); a=p.parse_args()
    df=pd.read_csv(DATA).reset_index(drop=True)
    if not all(c in df for c in FEATURES): raise ValueError("Add data/raw/college_wr.csv and rebuild to enable production/athletic comparisons.")
    hits=df.index[df.player.str.lower()==a.player.lower()].tolist()
    if not hits: raise ValueError(f"Player not found: {a.player}")
    idx=hits[0]; X=StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[FEATURES])); sims=cosine_similarity(X[idx:idx+1],X)[0]
    order=[i for i in sims.argsort()[::-1] if i!=idx][:a.top]; cols=["player","draft_year","draft_pick","nfl_value"]
    comps=df.loc[order,cols].copy(); comps["similarity"]=[round(float(sims[i]),3) for i in order]; OUT.mkdir(exist_ok=True); comps.to_csv(OUT/"similar_players.csv",index=False); print(comps.to_string(index=False))

if __name__=="__main__": main()
