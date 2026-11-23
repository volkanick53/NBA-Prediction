"""
models/train.py
===============
Trains and validates the whole prediction stack on REAL OUTCOMES only.

Data   : BigQuery fact_player_boxscores (2021-22 .. 2025-26) + current season (ESPN).
Protocol (walk-forward, every feature uses only information from before the game):
    burn-in     2021-22                      (player histories / ratings warm up)
    select      train 2022-23..2023-24  ->  validate 2024-25   (hyper-parameter choice)
    final fit   2022-23..2024-25
    test        2025-26 (full season)  and  2026-27 to date   (never used for any choice)

Outputs: models/game_model.json, models/props_<stat>.json, models/evaluation_report.txt

    python models/train.py            # full grid
    python models/train.py --quick    # small grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import data  # noqa: E402
from engine.features import (TEAM_FEATS, FeatureParams, game_table, low_minute_prior,  # noqa: E402
                             player_states, team_features_from_states)
from engine.game_model import (GAME_MODEL_FILE, MARGIN_FEATS, MODEL_DIR, PROP_FEATS,  # noqa: E402
                               PROP_RATE_STATS, PROP_STATS, TOTAL_FEATS, Linear,
                               game_dataset, total_level)
from engine.team_ratings import TUNED, walk_ratings  # noqa: E402

SELECT_TRAIN = ["2022-23", "2023-24"]
SELECT_VAL = ["2024-25"]
FINAL_TRAIN = ["2022-23", "2023-24", "2024-25"]
TESTS = [["2025-26"], [data.CURRENT_SEASON]]
REPORT = MODEL_DIR / "evaluation_report.txt"
XGB_PARAMS = dict(n_estimators=500, learning_rate=0.04, max_depth=5, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=30, reg_lambda=5.0, n_jobs=-1)

_lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    _lines.append(s)


# ---------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------
def prop_dataset(states: pd.DataFrame, m: np.ndarray, games: pd.DataFrame, rfeat: pd.DataFrame,
                 TF: pd.DataFrame) -> pd.DataFrame:
    P = states.copy()
    P["m"] = m
    for s in PROP_RATE_STATS:
        P[f"x_{s}"] = P["m"] * P[f"pre_{s}"]
    g = games.merge(rfeat, on="game_id")
    own = TF.add_prefix("team_").rename(columns={"team_game_id": "game_id", "team_team": "team"})
    opp = TF.add_prefix("opp_").rename(columns={"opp_game_id": "game_id", "opp_team": "opp"})
    P = P.merge(own, on=["game_id", "team"]).merge(opp, on=["game_id", "opp"])
    P = P.merge(g[["game_id", "home", "r_pace", "r_home_off", "r_home_def", "r_away_off",
                   "r_away_def", "home_rest", "away_rest"]], on="game_id")
    hm = P["team"] == P["home"]
    P["own_off"] = np.where(hm, P["r_home_off"], P["r_away_off"])
    P["own_def"] = np.where(hm, P["r_home_def"], P["r_away_def"])
    P["opp_off"] = np.where(hm, P["r_away_off"], P["r_home_off"])
    P["opp_def"] = np.where(hm, P["r_away_def"], P["r_home_def"])
    P["rest"] = np.where(hm, P["home_rest"], P["away_rest"])
    P["is_home"] = P["is_home"].astype(float)
    P["share_pts"] = P["x_pts"] / P["team_F_pts"].clip(lower=1e-6)
    P["share_fga"] = P["x_fga"] / P["team_F_fga"].clip(lower=1e-6)
    P["share_ast"] = P["x_ast"] / P["team_F_ast"].clip(lower=1e-6)
    P["share_trb"] = P["x_trb"] / (P["team_F_orb"] + P["team_F_drb"]).clip(lower=1e-6)
    return P


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def game_metrics(D: pd.DataFrame, m: np.ndarray, t: np.ndarray, sigma: float) -> dict:
    y = (D["margin"].to_numpy() > 0).astype(float)
    p = np.clip(norm.cdf(m / sigma), 1e-4, 1 - 1e-4)
    return {"games": len(D),
            "acc": float(((p > .5) == y).mean()),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)),
            "margin_mae": float(np.mean(np.abs(m - D["margin"]))),
            "margin_rmse": float(np.sqrt(np.mean((m - D["margin"]) ** 2))),
            "total_mae": float(np.mean(np.abs(t - D["total"]))),
            "total_bias": float(np.mean(t - D["total"]))}


def fmt(r: dict) -> str:
    return (f"n={r['games']:>5} acc {r['acc']:.3f} | logloss {r['logloss']:.4f} | brier {r['brier']:.4f}"
            f" | margin MAE {r['margin_mae']:.2f} RMSE {r['margin_rmse']:.2f}"
            f" | total MAE {r['total_mae']:.2f} (bias {r['total_bias']:+.2f})")


def fit_game(D: pd.DataFrame, seasons: list[str], alpha: float, feats_m=MARGIN_FEATS, feats_t=TOTAL_FEATS):
    T = D[D["season"].isin(seasons)]
    lm = Linear.fit(T, T["margin"].to_numpy(float), feats_m, alpha)
    lt = Linear.fit(T, T["total"].to_numpy(float), feats_t, alpha)
    sig = float(np.std(T["margin"] - lm.predict(T)))
    tsig = float(np.std(T["total"] - lt.predict(T)))
    return lm, lt, sig, tsig


def evaluate(D, seasons, lm, lt, sig, n0: float = np.inf) -> dict:
    corr = total_level(D, lt, n0)
    mask = D["season"].isin(seasons).to_numpy()
    V = D[mask]
    return game_metrics(V, lm.predict(V), lt.predict(V) + corr[mask], sig)


def calibration(D, lm, sig) -> pd.DataFrame:
    p = norm.cdf(lm.predict(D) / sig)
    y = (D["margin"] > 0).astype(float)
    b = pd.cut(p, [0, .2, .3, .4, .5, .6, .7, .8, 1.0])
    return pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(3)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main(quick: bool) -> None:
    t0 = time.time()
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York")).date()
    pg, _, _ = data.player_games(today, push_bq=False)
    say("=" * 100)
    say(f"NBA prediction model - training & validation report   ({datetime.now():%Y-%m-%d %H:%M})")
    say("=" * 100)
    say(f"Player-games: {len(pg):,} | seasons: "
        + ", ".join(f"{s} ({n:,})" for s, n in pg.groupby('season').size().items()))
    say("Sources: BigQuery fact_player_boxscores (completed seasons), ESPN box scores (current season).")
    say("No betting lines or external projections are used anywhere.\n")

    prior = low_minute_prior(pg[pg["season"] <= "2024-25"])
    games = game_table(pg)
    rfeat, _ = walk_ratings(games, TUNED)
    say(f"Games: {len(games):,} | team ratings walked day-by-day ({time.time() - t0:.0f}s)")

    # ---------- hyper-parameter selection (validation season only) ----------
    grid_fp = ([(40, 250)] if quick else list(itertools.product([20, 40, 80], [100, 250, 500])))
    alphas = [1.0, 10.0, 100.0]
    say(f"\n[1] Selection: train {SELECT_TRAIN} -> validate {SELECT_VAL}")
    best, cache = None, {}
    for hl, K in grid_fp:
        fp = FeatureParams(halflife_games=hl, prior_minutes=K)
        st = player_states(pg, prior, fp)
        TF, m = team_features_from_states(st)
        D = game_dataset(games, rfeat, TF)
        cache[(hl, K)] = (st, TF, m, D)
        for a in alphas:
            lm, lt, sig, _ = fit_game(D, SELECT_TRAIN, a)
            r = evaluate(D, SELECT_VAL, lm, lt, sig)
            say(f"   halflife {hl:>3} g | prior {K:>3} min | alpha {a:>5}: {fmt(r)}")
            if best is None or r["logloss"] < best[0]["logloss"]:
                best = (r, hl, K, a)
    _, hl, K, alpha = best
    say(f"   -> chosen: halflife {hl} games, prior {K} minutes, alpha {alpha}")
    st, TF, m_all, D = cache[(hl, K)]
    cache.clear()
    lm_s, lt_s, sig_s, _ = fit_game(D, SELECT_TRAIN, alpha)
    best_n0 = None
    for n0 in [np.inf, 400.0, 200.0, 100.0, 50.0]:
        r = evaluate(D, SELECT_VAL, lm_s, lt_s, sig_s, n0)
        say(f"   total level correction n0={n0:>5}: total MAE {r['total_mae']:.3f} (bias {r['total_bias']:+.2f})")
        if best_n0 is None or r["total_mae"] < best_n0[0]:
            best_n0 = (r["total_mae"], n0)
    n0 = best_n0[1]
    say(f"   -> chosen n0 = {n0}")

    # ---------- final game model ----------
    lm, lt, sig, tsig = fit_game(D, FINAL_TRAIN, alpha)
    say(f"\n[2] Game model fitted on {FINAL_TRAIN}: margin sigma {sig:.2f}, total sigma {tsig:.2f}")
    say("    Margin coefficients (points per unit):")
    for f, c in lm.raw_coefs().items():
        say(f"      {f:<16} {c:+.4f}")
    say(f"      home edge (intercept at average features) {lm.intercept:+.2f}")

    say("\n[3] TEST - never used for any choice")
    abl = {
        "full model": (MARGIN_FEATS, TOTAL_FEATS),
        "ratings only": (["r_margin", "b2b_diff"], ["r_total", "r_pace", "b2b_n"]),
        "players only": ([f for f in MARGIN_FEATS if not f.startswith("r_")],
                         [f for f in TOTAL_FEATS if not f.startswith("r_")]),
    }
    test_metrics = {}
    for seasons in TESTS:
        V = D[D["season"].isin(seasons)]
        if V.empty:
            continue
        say(f"   {seasons[0]}:")
        for name, (fm, ft) in abl.items():
            a_lm, a_lt, a_sig, _ = fit_game(D, FINAL_TRAIN, alpha, fm, ft)
            r = evaluate(D, seasons, a_lm, a_lt, a_sig, n0)
            say(f"     {name:<13} {fmt(r)}")
            if name == "full model":
                test_metrics[seasons[0]] = r
        say(f"     {'home always':<13} acc {(V['margin'] > 0).mean():.3f}  (baseline)")
        say("     calibration (full model):")
        for line in calibration(V, lm, sig).to_string().splitlines():
            say("       " + line)

    # ---------- player props ----------
    say("\n[4] Player props (players who played; minutes are projected, not actual)")
    P = prop_dataset(st, m_all, games, rfeat, TF)
    tr_s = P[P["season"].isin(SELECT_TRAIN)]
    va_s = P[P["season"].isin(SELECT_VAL)]
    tr_f = P[P["season"].isin(FINAL_TRAIN)]
    prop_sd, prop_metrics, prop_method = {}, {}, {}
    say(f"   minutes MAE (projection vs actual): validation {np.mean(np.abs(va_s['m'] - va_s['minutes'])):.2f}")
    for s in PROP_STATS:
        mdl = XGBRegressor(**XGB_PARAMS)
        mdl.fit(tr_s[PROP_FEATS], tr_s[s])
        v_x = np.mean(np.abs(np.clip(mdl.predict(va_s[PROP_FEATS]), 0, None) - va_s[s]))
        v_b = np.mean(np.abs(va_s[f"x_{s}"] - va_s[s]))
        use_model = v_x < v_b                      # decided on the validation season only
        prop_method[s] = "xgboost" if use_model else "rate_x_minutes"
        final = XGBRegressor(**XGB_PARAMS)
        final.fit(tr_f[PROP_FEATS], tr_f[s])
        final.save_model(MODEL_DIR / f"props_{s}.json")

        def pred(X):
            return np.clip(final.predict(X[PROP_FEATS]), 0, None) if use_model else X[f"x_{s}"].to_numpy()
        pred_tr = pred(tr_f)
        c = float(np.sqrt(np.mean((tr_f[s] - pred_tr) ** 2 / (pred_tr + 1.0))))
        prop_sd[s] = c                                   # sd ~ c * sqrt(pred + 1)
        line = (f"   {s:<5} validation MAE: xgb {v_x:.3f} vs rate*minutes {v_b:.3f} -> {prop_method[s]:<14}")
        for seasons in TESTS:
            T = P[P["season"].isin(seasons)]
            if T.empty:
                continue
            pt = pred(T)
            mae, base = np.mean(np.abs(pt - T[s])), np.mean(np.abs(T[f"x_{s}"] - T[s]))
            corr = np.corrcoef(pt, T[s])[0, 1]
            prop_metrics.setdefault(seasons[0], {})[s] = {"mae": float(mae), "baseline_mae": float(base),
                                                         "r": float(corr)}
            line += f" | test {seasons[0]} MAE {mae:.3f} (baseline {base:.3f}, r {corr:.2f})"
        say(line)

    # ---------- save ----------
    meta = {
        "version": "v3-player-lineup+ratings",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_seasons": FINAL_TRAIN,
        "feature_params": asdict(FeatureParams(halflife_games=hl, prior_minutes=K)),
        "rating_params": asdict(TUNED),
        "alpha": alpha,
        "prior_rates": prior,
        "margin_model": lm.to_dict(), "total_model": lt.to_dict(),
        "margin_sigma": sig, "total_sigma": tsig, "prop_sd": prop_sd, "prop_method": prop_method,
        "total_level_n0": None if not np.isfinite(n0) else n0,
        "test_metrics": test_metrics, "prop_test_metrics": prop_metrics,
    }
    GAME_MODEL_FILE.write_text(json.dumps(meta, indent=2))
    say(f"\nSaved {GAME_MODEL_FILE} and props_<stat>.json ({time.time() - t0:.0f}s total)")
    REPORT.write_text("\n".join(_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
