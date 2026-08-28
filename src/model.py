"""Train DraftIQ and generate a model-vs-market sleeper board."""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "wr_model_data.csv"
OUT = ROOT / "outputs"


def main() -> None:
    df = pd.read_csv(DATA)
    full = all(c in df.columns for c in FEATURES)
    feature_cols = FEATURES if full else ["market_expectation"]
    candidates = {
        "elastic_net": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ElasticNet(alpha=0.05, l1_ratio=0.35, max_iter=20000))]),
        "gradient_boosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", GradientBoostingRegressor(n_estimators=250, learning_rate=0.03, max_depth=2, loss="huber", random_state=42))]),
    }
    X, y = df[feature_cols], df["nfl_value"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = {name: float((-cross_val_score(m, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error")).mean()) for name, m in candidates.items()}
    best_name = min(cv_mae, key=cv_mae.get); best = candidates[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)
    metrics = {"selected_model": best_name, "full_college_feature_model": full, "cv_mae": cv_mae,
               "test_mae": float(mean_absolute_error(yte, pred)), "test_rmse": float(mean_squared_error(yte, pred) ** .5),
               "test_r2": float(r2_score(yte, pred)), "n_rows": int(len(df))}
    best.fit(X, y); df["predicted_nfl_value"] = best.predict(X)
    df["model_grade"] = (df["predicted_nfl_value"].rank(pct=True) * 100).round(1)
    df["model_rank"] = df["predicted_nfl_value"].rank(ascending=False, method="min").astype(int)
    df["market_rank"] = df["draft_pick"].rank(ascending=True, method="min").astype(int)
    df["sleeper_score"] = df["market_rank"] - df["model_rank"]
    df = df.sort_values("sleeper_score", ascending=False)
    OUT.mkdir(exist_ok=True)
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2)); df.to_csv(OUT / "wr_rankings.csv", index=False); joblib.dump(best, OUT / "draftiq_model.joblib")
    fitted = best.named_steps["model"]; vals = fitted.feature_importances_ if hasattr(fitted, "feature_importances_") else np.abs(fitted.coef_)
    pd.DataFrame({"feature": feature_cols, "importance": vals}).sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    plt.figure(figsize=(7,6)); plt.scatter(yte, pred, alpha=.7); lo, hi=min(yte.min(),pred.min()),max(yte.max(),pred.max()); plt.plot([lo,hi],[lo,hi],linestyle="--"); plt.xlabel("Actual 3-year NFL value"); plt.ylabel("Predicted value"); plt.title(f"DraftIQ holdout evaluation — {best_name}"); plt.tight_layout(); plt.savefig(OUT/"predicted_vs_actual.png",dpi=180); plt.close()
    top=df.head(15).sort_values("sleeper_score"); plt.figure(figsize=(9,7)); plt.barh(top["player"],top["sleeper_score"]); plt.xlabel("Sleeper score (market rank − model rank)"); plt.title("DraftIQ model-vs-market sleeper board"); plt.tight_layout(); plt.savefig(OUT/"sleeper_board.png",dpi=180); plt.close()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
