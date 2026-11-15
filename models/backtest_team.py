"""
models/backtest_team.py
=======================
Out-of-sample backtest (2025-26 holdout) of the game predictor logic used in
predict_oct20_offline.py: ratings formula + XGBoost blend + normal win prob.

Reports winner accuracy, Brier, log-loss, spread/total MAE, and searches the
XGBoost blend weight and win-probability sigma that minimise log-loss.

    python models/backtest_team.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor

df = pd.read_parquet("data/team_features.parquet")
meta = json.load(open("models/team_model_meta.json"))
val_season = meta["val_season"]
val = df[df["season"] == val_season]

home = val[val["is_home"] == 1].copy()
away = val[val["is_home"] == 0].copy()
g = home.merge(away, on="game_id", suffixes=("_h", "_a"))
print(f"Holdout season {val_season}: {len(g)} games")

# league averages from *training* seasons only
tr = df[df["season"] != val_season]
lg_rtg = tr["roll_ortg_10"].mean()
lg_pace = tr["roll_pace_10"].mean()
HCA = 2.3

# --- ratings predictor (same formula as production) ---
pace_rt = g["roll_pace_10_h"] * g["roll_pace_10_a"] / lg_pace
h_rt = g["roll_ortg_10_h"] * g["roll_drtg_10_a"] / lg_rtg * pace_rt / 100 + HCA / 2
a_rt = g["roll_ortg_10_a"] * g["roll_drtg_10_h"] / lg_rtg * pace_rt / 100 - HCA / 2
h_rt = h_rt - 1.5 * (g["rest_days_h"] == 1)
a_rt = a_rt - 1.5 * (g["rest_days_a"] == 1)

# --- XGBoost predictions ---
def load(name):
    m = XGBRegressor(); m.load_model(f"models/{name}.json"); return m

pcols = meta["points_features"]
def feats(suffix_self, suffix_opp):
    X = {}
    for c in pcols:
        if c == "is_home":
            X[c] = 1.0 if suffix_self == "_h" else 0.0
        elif c.endswith("_opp_pregame"):
            X[c] = g[c.replace("_opp_pregame", "") + suffix_opp]
        else:
            X[c] = g[c + suffix_self]
    return pd.DataFrame(X)[pcols].fillna(0)

h_x = load("team_points_home_model").predict(feats("_h", "_a"))
a_x = load("team_points_away_model").predict(feats("_a", "_h"))

actual_margin = (g["team_pts_h"] - g["team_pts_a"]).to_numpy()
actual_total = (g["team_pts_h"] + g["team_pts_a"]).to_numpy()
home_win = (actual_margin > 0).astype(int)

def evaluate(w_xgb, sigma):
    hp = (1 - w_xgb) * h_rt + w_xgb * h_x
    ap = (1 - w_xgb) * a_rt + w_xgb * a_x
    spread = (hp - ap).to_numpy() if hasattr(hp, "to_numpy") else hp - ap
    p = np.clip(norm.cdf(spread / sigma), 1e-4, 1 - 1e-4)
    return {
        "acc": float(((p > 0.5) == home_win).mean()),
        "brier": float(np.mean((p - home_win) ** 2)),
        "logloss": float(-np.mean(home_win * np.log(p) + (1 - home_win) * np.log(1 - p))),
        "spread_mae": float(np.mean(np.abs(spread - actual_margin))),
        "total_mae": float(np.mean(np.abs((hp + ap) - actual_total))),
    }

print(f"\nBaselines: always-home acc = {home_win.mean():.3f}, "
      f"coin-flip Brier = 0.2500, coin-flip logloss = {np.log(2):.4f}")

print("\nCurrent production setting (w_xgb=0.20, sigma=12.0):")
for k, v in evaluate(0.20, 12.0).items():
    print(f"   {k:<11}: {v:.4f}")

best = None
for w in np.arange(0, 1.01, 0.1):
    for s in np.arange(10, 16.1, 0.5):
        r = evaluate(w, s)
        if best is None or r["logloss"] < best[2]["logloss"]:
            best = (w, s, r)
print(f"\nBest in-sample grid (w_xgb={best[0]:.1f}, sigma={best[1]:.1f}):")
for k, v in best[2].items():
    print(f"   {k:<11}: {v:.4f}")

# calibration table for production setting
hp = 0.8 * h_rt + 0.2 * h_x
ap = 0.8 * a_rt + 0.2 * a_x
p = norm.cdf((hp - ap).to_numpy() / 12.0)
bins = pd.cut(p, [0, .3, .4, .5, .6, .7, .8, 1.0])
cal = pd.DataFrame({"bin": bins, "p": p, "y": home_win}).groupby("bin", observed=True).agg(
    n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
print("\nCalibration (production setting):")
print(cal.round(3).to_string())
