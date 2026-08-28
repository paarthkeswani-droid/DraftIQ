"""Generate a deterministic fictional WR dataset for the complete DraftIQ demo."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data"/"raw"


def main(rows_per_class: int=30, seed: int=14):
    rng=np.random.default_rng(seed); years=np.repeat(np.arange(2018,2025),rows_per_class); n=len(years); idx=np.arange(n)
    age=rng.uniform(19.4,23.4,n).round(1); height=rng.normal(73.2,2.3,n).clip(68,79).round(1); weight=(height*2.25+rng.normal(35,13,n)).clip(165,235).round(); games=rng.integers(9,15,n)
    targets=rng.uniform(5,11,n)*games; catch_rate=rng.normal(.66,.07,n).clip(.45,.82); receptions=(targets*catch_rate).round(); ypr=rng.normal(14.2,2.6,n).clip(8.5,21); yards=(receptions*ypr).round(); tds=np.maximum(1,(yards/rng.uniform(95,180,n)).round()); team_yards=rng.normal(3600,500,n).clip(2400,5000).round(); team_tds=rng.normal(30,6,n).clip(16,48).round()
    forty=(4.62-(height-72)*.006-(weight-195)*.001+rng.normal(0,.105,n)).clip(4.25,4.85).round(2); vertical=rng.normal(35,3,n).clip(27,44).round(1); broad=rng.normal(121,7,n).clip(103,139).round(0)
    production=yards/games; athletic=weight*200/(forty**4); talent=(production-55)/15+(athletic-98)/14+(22-age)*.45+(yards/team_yards-.2)*3+rng.normal(0,.7,n)
    market_signal=talent+rng.normal(0,1.1,n); picks=np.empty(n,dtype=int)
    for year in np.unique(years):
        loc=np.where(years==year)[0]; order=loc[np.argsort(market_signal[loc])[::-1]]; slots=np.sort(rng.choice(np.arange(8,250),size=len(loc),replace=False)); picks[order]=slots
    college=pd.DataFrame({"player":[f"Demo Receiver {i+1:03d}" for i in idx],"draft_year":years,"age":age,"height_in":height,"weight_lb":weight,"games":games,"receptions":receptions,"rec_yards":yards,"rec_tds":tds,"team_pass_yards":team_yards,"team_pass_tds":team_tds,"forty":forty,"vertical":vertical,"broad_jump":broad,"college":[f"Demo State {(i%24)+1}" for i in idx],"conference":[f"Conference {chr(65+i%5)}" for i in idx]})
    draft=pd.DataFrame({"pfr_player_name":college.player,"position":"WR","season":years,"pick":picks})
    rows=[]
    for i,row in college.iterrows():
        career=max(0,2200+talent[i]*1150+rng.normal(0,900)); shares=np.array([.28,.34,.38])+rng.normal(0,.025,3); shares=np.maximum(.1,shares); shares/=shares.sum()
        for yr,share in enumerate(shares):
            rec_yards=max(0,career*share); rec=max(0,rec_yards/rng.uniform(11,15)); rec_tds=max(0,round(rec_yards/rng.uniform(110,190)))
            rows.append({"player_display_name":row.player,"season":int(row.draft_year+yr),"receptions":round(rec),"receiving_yards":round(rec_yards),"receiving_tds":rec_tds})
    RAW.mkdir(parents=True,exist_ok=True); college.to_csv(RAW/"college_wr.csv",index=False); draft.to_csv(RAW/"draft_picks.csv",index=False); pd.DataFrame(rows).to_csv(RAW/"seasonal.csv",index=False)
    print(f"Generated {n} fictional WR prospects across {len(np.unique(years))} draft classes")


if __name__=="__main__": main()

