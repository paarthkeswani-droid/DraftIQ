"""Interactive DraftIQ recruiter demo."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent; OUT=ROOT/"outputs"
st.set_page_config(page_title="DraftIQ",page_icon="🏈",layout="wide")
st.markdown("""<style>.stApp{background:#f4f0e7;color:#172536}[data-testid="stSidebar"]{background:#102d46}[data-testid="stSidebar"] *{color:#f5f0e7!important}.hero{background:linear-gradient(120deg,#102d46,#1e5278);color:white;padding:28px 32px;border-radius:14px;margin-bottom:22px}.hero small{color:#f2b24a;letter-spacing:.14em;font-weight:700}.hero h1{margin:.25rem 0}.hero p{color:#c9d7e2;margin:0}[data-testid="stMetric"]{background:#fffdf8;border:1px solid #ddd8ce;padding:15px;border-radius:10px}.card{background:#fffdf8;border:1px solid #ddd8ce;padding:18px;border-radius:10px}</style>""",unsafe_allow_html=True)
needed=[OUT/"wr_rankings.csv",OUT/"model_metrics.json",OUT/"feature_importance.csv"]
if not all(x.exists() for x in needed): st.error("Demo outputs are missing. Run `make demo` first."); st.stop()
rankings=pd.read_csv(needed[0]); metrics=json.loads(needed[1].read_text()); importance=pd.read_csv(needed[2]); selected=metrics["selected_model"]; score=metrics["models"][selected]
st.sidebar.markdown("## 🏈 DRAFTIQ"); st.sidebar.caption("WR prospect intelligence"); st.sidebar.divider(); page=st.sidebar.radio("Workspace",["Model overview","Sleeper board","Prospect explorer","Feature intelligence"]); st.sidebar.divider(); st.sidebar.info("All demo players and outcomes are fictional. Results demonstrate the workflow, not real player evaluations.")
st.markdown("""<div class="hero"><small>RECRUITER DEMO • WR ANALYTICS</small><h1>Find value before the market does.</h1><p>Independent prospect grades, model-vs-market gaps, and transparent statistical profiles.</p></div>""",unsafe_allow_html=True)
if page=="Model overview":
    a,b,c,d=st.columns(4); a.metric("Selected model",selected.replace("_"," ").title()); b.metric("Holdout MAE",f"{score['mae']:.0f}"); c.metric("Holdout R²",f"{score['r2']:.2f}"); d.metric("Prospect histories",metrics["n_rows"]); st.caption(metrics["validation_split"]+" • synthetic demonstration")
    left,right=st.columns(2); left.image(str(OUT/"predicted_vs_actual.png"),width="stretch",caption="Held-out draft class evaluation"); right.image(str(OUT/"sleeper_board.png"),width="stretch",caption="Largest model-vs-market gaps")
elif page=="Sleeper board":
    year=st.selectbox("Draft class",sorted(rankings.draft_year.unique(),reverse=True)); board=rankings[rankings.draft_year==year].sort_values("sleeper_score",ascending=False); st.dataframe(board[["player","college","draft_pick","model_rank","market_rank","sleeper_score","model_grade"]],hide_index=True,width="stretch",column_config={"model_grade":st.column_config.ProgressColumn("Grade",min_value=0,max_value=100,format="%.1f"),"sleeper_score":st.column_config.NumberColumn("Sleeper score",format="%+d")})
elif page=="Prospect explorer":
    chosen=st.selectbox("Select a receiver",rankings.sort_values("model_grade",ascending=False).player.tolist()); p=rankings[rankings.player==chosen].iloc[0]; a,b,c,d,e=st.columns(5); a.metric("DraftIQ grade",f"{p.model_grade:.1f}"); b.metric("Sleeper score",f"{p.sleeper_score:+.0f}"); c.metric("Draft pick",f"#{int(p.draft_pick)}"); d.metric("Yards/game",f"{p.yards_per_game:.1f}"); e.metric("Speed score",f"{p.speed_score:.1f}")
    left,right=st.columns([1.3,1]); profile=pd.DataFrame({"Metric":["Yards / game","Receptions / game","Yards / reception","Yard share x100","TD share x100","Speed score"],"Value":[p.yards_per_game,p.rec_per_game,p.yards_per_reception,p.yard_share*100,p.td_share*100,p.speed_score]}).set_index("Metric"); left.bar_chart(profile,horizontal=True,color="#1e5278"); right.markdown(f"<div class='card'><b>{p.player}</b><br>{p.college} • {int(p.draft_year)}<hr>Age: <b>{p.age:.1f}</b><br>Size: <b>{p.height_in:.1f} in / {p.weight_lb:.0f} lb</b><br>40-yard dash: <b>{p.forty:.2f}s</b><br>Model rank: <b>#{int(p.model_rank)}</b><br>Market rank: <b>#{int(p.market_rank)}</b></div>",unsafe_allow_html=True); right.info("A model flag starts a film question; it does not end the evaluation.")
else:
    st.subheader("What drives the model?"); st.write("The chart exposes the selected model's learned relationships so a scout can challenge the signal."); st.bar_chart(importance.sort_values("importance").set_index("feature"),horizontal=True,color="#1e5278"); st.dataframe(importance,hide_index=True,width="stretch"); st.warning("Feature importance is descriptive, not causal. Route detail, hands, health, role, and competition context remain essential.")
st.divider(); st.caption("DraftIQ • Educational portfolio project • Synthetic demo data")

