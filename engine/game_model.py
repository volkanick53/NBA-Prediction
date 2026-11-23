"""
engine/game_model.py
====================
Game and player-prop models. Training: models/train.py. Same code path is used
by training, back-testing and the live engine, so what is validated is what runs.

Game level (two ridge regressions on standardised features, fitted on real outcomes):
    margin = f( home-minus-away projected box score of tonight's lineups,
                opponent-adjusted rating margin, back-to-back difference )   + home edge (intercept)
    total  = f( summed projected box scores, rating total & pace, back-to-backs )
    P(home win) = Phi(margin / sigma),   sigma = residual sd on training seasons

Player props: one gradient-boosted model per stat on the player's shrunk per-minute
rates x projected minutes, team/opponent context and ratings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine.features import TEAM_FEATS

MODEL_DIR = Path("models")
GAME_MODEL_FILE = MODEL_DIR / "game_model.json"

MARGIN_FEATS = [f"d_{s}" for s in TEAM_FEATS] + ["r_margin", "b2b_diff"]
TOTAL_FEATS = [f"s_{s}" for s in TEAM_FEATS] + ["r_total", "r_pace", "b2b_n"]

PROP_STATS = ["pts", "trb", "ast", "fg3m", "stl", "blk"]
PROP_RATE_STATS = ["pts", "fga", "fta", "fg3m", "fg3a", "orb", "drb", "trb", "ast", "stl", "blk",
                   "tov", "pf", "plus_minus"]
PROP_FEATS = (["m", "pre_min", "pre_games"] + [f"x_{s}" for s in PROP_RATE_STATS] +
              ["share_pts", "share_fga", "share_ast", "share_trb",
               "team_F_pts", "team_F_fga", "team_F_ast", "team_F_orb", "team_F_drb",
               "opp_F_pts", "opp_F_stl", "opp_F_blk", "opp_F_drb", "opp_F_orb", "opp_F_pf",
               "opp_F_plus_minus", "r_pace", "own_off", "own_def", "opp_off", "opp_def",
               "is_home", "rest"])


# ---------------------------------------------------------------
# Game rows
# ---------------------------------------------------------------
def game_row(hf: dict, af: dict, rf: dict, home_rest: float, away_rest: float) -> dict:
    """hf/af: team_vector() dicts; rf: rating_features()."""
    r = {}
    for s in TEAM_FEATS:
        r[f"d_{s}"] = hf[f"F_{s}"] - af[f"F_{s}"]
        r[f"s_{s}"] = hf[f"F_{s}"] + af[f"F_{s}"]
    b2b_h, b2b_a = float(home_rest <= 1), float(away_rest <= 1)
    r.update({"r_margin": rf["r_margin"], "r_total": rf["r_total"], "r_pace": rf["r_pace"],
              "b2b_diff": b2b_a - b2b_h, "b2b_n": b2b_h + b2b_a})
    return r


def game_dataset(games: pd.DataFrame, rfeat: pd.DataFrame, TF: pd.DataFrame) -> pd.DataFrame:
    D = games.merge(rfeat, on="game_id")
    h = TF.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home"})
    a = TF.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away"})
    D = D.merge(h, on=["game_id", "home"]).merge(a, on=["game_id", "away"])
    for s in TEAM_FEATS:
        D[f"d_{s}"] = D[f"h_F_{s}"] - D[f"a_F_{s}"]
        D[f"s_{s}"] = D[f"h_F_{s}"] + D[f"a_F_{s}"]
    D["b2b_diff"] = (D["away_rest"] <= 1).astype(float) - (D["home_rest"] <= 1).astype(float)
    D["b2b_n"] = (D["away_rest"] <= 1).astype(float) + (D["home_rest"] <= 1).astype(float)
    D["margin"] = D["home_pts"] - D["away_pts"]
    D["total"] = D["home_pts"] + D["away_pts"]
    return D



def total_level(D: pd.DataFrame, lt: Linear, n0: float) -> np.ndarray:
    """
    In-season scoring-level correction for totals, walk-forward: for a game on day d,
    mean residual (actual - predicted total) of that season's games BEFORE d, shrunk
    with n0 pseudo-games. Player rates carry last season's scoring environment; this
    lets the model track league-wide scoring drift using only past games.
    """
    if not np.isfinite(n0):
        return np.zeros(len(D))
    res = D["total"].to_numpy() - lt.predict(D)
    out = np.zeros(len(D))
    for _, idx in D.groupby("season").indices.items():
        sub = D.iloc[idx]
        r = res[idx]
        order = np.argsort(sub["game_date"].to_numpy(), kind="stable")
        dates = sub["game_date"].to_numpy()[order]
        csum, ccnt = np.cumsum(r[order]), np.arange(1, len(order) + 1)
        first = np.searchsorted(dates, dates, side="left")      # games strictly before the day
        prev_sum = np.where(first > 0, csum[first - 1], 0.0)
        prev_cnt = np.where(first > 0, ccnt[first - 1], 0)
        out[idx[order]] = prev_sum / (prev_cnt + n0)
    return out



# ---------------------------------------------------------------
# Ridge with standardisation (intercept unpenalised)
# ---------------------------------------------------------------
@dataclass
class Linear:
    feats: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, X: pd.DataFrame, y: np.ndarray, feats: list[str], alpha: float) -> "Linear":
        A = X[feats].to_numpy(float)
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        Z = (A - mu) / sd
        yc = y - y.mean()
        coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ yc)
        return cls(feats, mu, sd, coef, float(y.mean()))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.feats].to_numpy(float) - self.mean) / self.scale
        return Z @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {"feats": self.feats, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "Linear":
        return cls(d["feats"], np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]),
                   d["intercept"])

    def raw_coefs(self) -> dict[str, float]:
        """Coefficients in original feature units (for interpretation)."""
        return {f: float(c / s) for f, c, s in zip(self.feats, self.coef, self.scale)}


class GameModel:
    def __init__(self, d: dict):
        self.meta = d
        self.margin = Linear.from_dict(d["margin_model"])
        self.total = Linear.from_dict(d["total_model"])
        self.sigma = d["margin_sigma"]
        self.total_sigma = d["total_sigma"]
        self.prop_sd = d.get("prop_sd", {})
        self.prop_dist = {s: PropDistribution.from_dict(v) for s, v in d.get("prop_dist", {}).items()}
        self._props = {}

    @classmethod
    def load(cls) -> "GameModel":
        return cls(json.loads(GAME_MODEL_FILE.read_text()))

    def predict_games(self, X: pd.DataFrame) -> pd.DataFrame:
        m = self.margin.predict(X)
        t = self.total.predict(X)
        return pd.DataFrame({"margin": m, "total": t,
                             "home_pts": t / 2 + m / 2, "away_pts": t / 2 - m / 2,
                             "home_win_prob": norm.cdf(m / self.sigma)}, index=X.index)

    def prop_model(self, stat: str):
        if stat not in self._props:
            from xgboost import XGBRegressor
            mdl = XGBRegressor()
            mdl.load_model(MODEL_DIR / f"props_{stat}.json")
            self._props[stat] = mdl
        return self._props[stat]

    def predict_props(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for s in PROP_STATS:
            out[s] = np.clip(self.prop_model(s).predict(X[PROP_FEATS]), 0, None)
        return out


# ---------------------------------------------------------------
# Predictive distribution of a player stat (skewed, data-driven)
# ---------------------------------------------------------------
QLEVELS = np.linspace(0.0, 1.0, 101)


class PropDistribution:
    """
    Conditional empirical distribution of  y - mu  given the model's prediction mu.

    Training player-games are grouped into bins by mu (quantile bins); each bin keeps
    the empirical quantiles of the residual y - mu. For a new prediction the two nearest
    bins are blended. This captures the right skew of NBA box-score stats (a role
    player can have a 25-point night, he cannot have -15) and any systematic bias of mu
    in a given range, calibrated on real outcomes rather than an assumed shape.
    """

    def __init__(self, centers: np.ndarray, rq: np.ndarray):
        self.centers = np.asarray(centers, float)   # (B,)
        self.rq = np.asarray(rq, float)             # (B, 101) residual quantiles

    @classmethod
    def fit(cls, mu: np.ndarray, y: np.ndarray, n_bins: int = 30) -> "PropDistribution":
        mu, y = np.asarray(mu, float), np.asarray(y, float)
        edges = np.unique(np.quantile(mu, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, len(edges) - 2)
        centers, rq = [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() < 50:
                continue
            centers.append(mu[sel].mean())
            rq.append(np.quantile(y[sel] - mu[sel], QLEVELS))
        return cls(np.array(centers), np.array(rq))

    def _resid_q(self, mu: np.ndarray) -> np.ndarray:
        """(N, 101) residual quantiles, linear blend of the two nearest bins."""
        mu = np.atleast_1d(np.asarray(mu, float))
        c = self.centers
        j = np.clip(np.searchsorted(c, mu) - 1, 0, len(c) - 2)
        w = np.clip((mu - c[j]) / (c[j + 1] - c[j]), 0.0, 1.0)[:, None]
        return (1 - w) * self.rq[j] + w * self.rq[j + 1]

    def quantile(self, mu, q: float) -> np.ndarray:
        r = self._resid_q(mu)
        return np.maximum(np.atleast_1d(mu) + np.array([np.interp(q, QLEVELS, row) for row in r]), 0.0)

    def prob_at_least(self, mu, k: float) -> np.ndarray:
        """P(y >= k) for integer-valued stats (continuity: y >= k  <=>  y > k - 0.5)."""
        mu = np.atleast_1d(np.asarray(mu, float))
        r = self._resid_q(mu)
        thr = k - 0.5 - mu
        cdf = np.array([np.interp(t, row, QLEVELS, left=0.0, right=1.0) for t, row in zip(thr, r)])
        return 1.0 - cdf

    def to_dict(self) -> dict:
        return {"centers": self.centers.round(4).tolist(), "rq": self.rq.round(3).tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PropDistribution":
        return cls(np.array(d["centers"]), np.array(d["rq"]))


# thresholds reported per stat (and the "breakout" definition used by the live engine)
PROP_THRESHOLDS = {"pts": [10, 15, 20, 25, 30, 40], "trb": [5, 8, 10, 15], "ast": [5, 8, 10],
                   "fg3m": [2, 3, 4, 5], "stl": [2, 3], "blk": [2, 3]}


# ---------------------------------------------------------------
# Player prop rows
# ---------------------------------------------------------------
def prop_rows(players: pd.DataFrame, team_f: dict, opp_f: dict, rf_own: dict, rf_opp: dict,
              r_pace: float, is_home: bool, rest: float) -> pd.DataFrame:
    """
    players: columns m (scaled minutes), pre_min, pre_games, pre_<s> (per-minute rates).
    rf_own/rf_opp: {"off","def"} rating deviations of the player's team / opponent.
    """
    X = pd.DataFrame(index=players.index)
    X["m"] = players["m"]
    X["pre_min"] = players["pre_min"]
    X["pre_games"] = players["pre_games"]
    for s in PROP_RATE_STATS:
        X[f"x_{s}"] = players["m"] * players[f"pre_{s}"]
    for s, F in (("pts", "F_pts"), ("fga", "F_fga"), ("ast", "F_ast")):
        X[f"share_{s}"] = X[f"x_{s}"] / max(team_f[F], 1e-6)
    X["share_trb"] = X["x_trb"] / max(team_f["F_orb"] + team_f["F_drb"], 1e-6)
    for k in ("pts", "fga", "ast", "orb", "drb"):
        X[f"team_F_{k}"] = team_f[f"F_{k}"]
    for k in ("pts", "stl", "blk", "drb", "orb", "pf", "plus_minus"):
        X[f"opp_F_{k}"] = opp_f[f"F_{k}"]
    X["r_pace"] = r_pace
    X["own_off"], X["own_def"] = rf_own["off"], rf_own["def"]
    X["opp_off"], X["opp_def"] = rf_opp["off"], rf_opp["def"]
    X["is_home"] = float(is_home)
    X["rest"] = float(rest)
    return X
