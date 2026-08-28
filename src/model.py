"""Train DraftIQ and produce an explainable model-vs-market sleeper board."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES


def candidates(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "elastic_net": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ElasticNet(alpha=.05, l1_ratio=.35, max_iter=20000, random_state=random_state))]),
        "gradient_boosting": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=300, learning_rate=.03, max_depth=2, loss="huber", random_state=random_state))]),
    }


def train_and_evaluate(data: pd.DataFrame, random_state: int = 42):
    missing = sorted(set(FEATURES + ["nfl_value", "draft_pick", "draft_year"]) - set(data.columns))
    if missing:
        raise ValueError(f"Model data is missing columns: {', '.join(missing)}")
    if len(data) < 40:
        raise ValueError("At least 40 prospect rows are required for evaluation")
    X, y = data[FEATURES], data.nfl_value
    years = sorted(data.draft_year.dropna().unique())
    if len(years) >= 3 and (data.draft_year == years[-1]).sum() >= 8:
        test_mask = data.draft_year == years[-1]
        Xtr, Xte, ytr, yte = X.loc[~test_mask], X.loc[test_mask], y.loc[~test_mask], y.loc[test_mask]
        split = f"time-based holdout: {int(years[-1])} draft class"
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=random_state)
        split = "random 80/20 holdout"
    cv = KFold(n_splits=min(5, max(2, len(Xtr)//20)), shuffle=True, random_state=random_state)
    models, metrics = candidates(random_state), {}
    for name, model in models.items():
        cv_mae = -cross_val_score(model, Xtr, ytr, scoring="neg_mean_absolute_error", cv=cv).mean()
        model.fit(Xtr, ytr); predicted = model.predict(Xte)
        metrics[name] = {"cv_mae":float(cv_mae), "mae":float(mean_absolute_error(yte,predicted)), "rmse":float(mean_squared_error(yte,predicted)**.5), "r2":float(r2_score(yte,predicted))}
    winner = min(metrics, key=lambda name: metrics[name]["cv_mae"])
    selected = models[winner].fit(X, y)
    return selected, winner, metrics, split, (yte.to_numpy(), models[winner].predict(Xte))


def build_rankings(data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    ranked = data.copy(); ranked["predicted_nfl_value"] = predictions
    ranked["model_grade"] = ranked.predicted_nfl_value.rank(pct=True).mul(100).round(1)
    ranked["model_rank"] = ranked.groupby("draft_year").predicted_nfl_value.rank(ascending=False, method="min").astype(int)
    ranked["market_rank"] = ranked.groupby("draft_year").draft_pick.rank(ascending=True, method="min").astype(int)
    ranked["sleeper_score"] = ranked.market_rank - ranked.model_rank
    return ranked.sort_values(["sleeper_score", "model_grade"], ascending=[False, False])


def importance_table(model: Pipeline) -> pd.DataFrame:
    fitted = model.named_steps["model"]
    values = fitted.feature_importances_ if hasattr(fitted, "feature_importances_") else fitted.coef_
    return pd.DataFrame({"feature":FEATURES,"importance":values}).assign(magnitude=lambda x:x.importance.abs()).sort_values("magnitude",ascending=False).drop(columns="magnitude")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--data",type=Path,default=Path("data/processed/wr_model_data.csv")); parser.add_argument("--outputs",type=Path,default=Path("outputs")); args=parser.parse_args()
    data=pd.read_csv(args.data); model,winner,metrics,split,holdout=train_and_evaluate(data); args.outputs.mkdir(parents=True,exist_ok=True)
    ranked=build_rankings(data,model.predict(data[FEATURES])); ranked.to_csv(args.outputs/"wr_rankings.csv",index=False); importance_table(model).to_csv(args.outputs/"feature_importance.csv",index=False)
    payload={"selected_model":winner,"validation_split":split,"models":metrics,"n_rows":len(data)}; (args.outputs/"model_metrics.json").write_text(json.dumps(payload,indent=2))
    actual,predicted=holdout; fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(actual,predicted,alpha=.72,color="#173f63"); lo,hi=min(actual.min(),predicted.min()),max(actual.max(),predicted.max()); ax.plot([lo,hi],[lo,hi],"--",color="#ef6b3b"); ax.set(title=f"DraftIQ holdout - {winner}",xlabel="Actual 3-year NFL value",ylabel="Predicted NFL value"); fig.tight_layout(); fig.savefig(args.outputs/"predicted_vs_actual.png",dpi=180); plt.close(fig)
    top=ranked.head(15).sort_values("sleeper_score"); fig,ax=plt.subplots(figsize=(9,7)); ax.barh(top.player,top.sleeper_score,color="#173f63"); ax.axvline(0,color="#ef6b3b",lw=1); ax.set(title="DraftIQ model-vs-market sleeper board",xlabel="Sleeper score (market rank - model rank)"); fig.tight_layout(); fig.savefig(args.outputs/"sleeper_board.png",dpi=180); plt.close(fig)
    print(f"Selected {winner} using {split}; wrote outputs to {args.outputs}")


if __name__=="__main__": main()

