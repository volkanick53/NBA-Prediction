"""
engine/features.py
==================
Leakage-free player and team features. Every "pre_" value of a player-game is
computed from that player's EARLIER games only.

Player state (per player, decayed over his own games, across seasons):
    rate_s      shrunk per-minute rate of box stat s:
                    (sum_g d^age * stat_g + K * prior_s) / (sum_g d^age * min_g + K)
                d = 0.5 ** (1 / halflife_games); prior = rates of low-minute players
                (what an unknown player looks like), K = pseudo-minutes of prior.
    proj_min    expected minutes when he plays: EWMA of minutes, half-life MIN_HALFLIFE games.

Team features for one game (from the players expected to play):
    minutes  m_i = proj_min_i scaled so the team sums to 240
    F_s      = sum_i m_i * rate_i,s        ("projected team box score")
    F_pm     = sum_i m_i * pm_rate_i / 5    (on-court plus-minus -> team margin units)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

STATS = ["pts", "fga", "fta", "fg3m", "fg3a", "orb", "drb", "trb", "ast", "stl", "blk", "tov",
         "pf", "plus_minus"]
TEAM_FEATS = ["pts", "fga", "fta", "fg3m", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "plus_minus"]
MIN_HALFLIFE = 6.0
DEFAULT_MIN = 12.0          # first career game, no minutes history
TEAM_MINUTES = 240.0
MAX_MIN = 42.0
ROTATION_PROTECT = 20.0     # players projected above this keep their minutes when trimming


@dataclass
class FeatureParams:
    halflife_games: float = 40.0
    prior_minutes: float = 250.0


def low_minute_prior(pg: pd.DataFrame) -> dict[str, float]:
    """Per-minute rates of players in their first 300 career minutes (an 'unknown' player)."""
    d = pg.sort_values("game_date").copy()
    d["cum_min"] = d.groupby("player_id")["minutes"].cumsum() - d["minutes"]
    low = d[d["cum_min"] < 300]
    return {s: float(low[s].sum() / low["minutes"].sum()) for s in STATS}


def _decayed_cumsum(x: np.ndarray, d: float) -> np.ndarray:
    """y_t = d*y_{t-1} + x_t  (value AFTER game t)."""
    return lfilter([1.0], [1.0, -d], x)


def player_states(pg: pd.DataFrame, prior: dict[str, float], fp: FeatureParams) -> pd.DataFrame:
    """
    Adds pre-game columns  pre_<s>  (shrunk per-minute rates), pre_min (projected minutes),
    pre_games, plus post-game columns post_<s>, post_min used for the next, unseen game.
    """
    d = 0.5 ** (1.0 / fp.halflife_games)
    a = 0.5 ** (1.0 / MIN_HALFLIFE)
    K = fp.prior_minutes
    df = pg.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    n = len(df)
    out = {f"pre_{s}": np.empty(n) for s in STATS}
    out.update({f"post_{s}": np.empty(n) for s in STATS})
    pre_min, post_min, pre_games = np.empty(n), np.empty(n), np.empty(n)

    mins = df["minutes"].to_numpy(float)
    X = {s: df[s].to_numpy(float) for s in STATS}
    starts = np.flatnonzero(np.r_[True, df["player_id"].to_numpy()[1:] != df["player_id"].to_numpy()[:-1]])
    ends = np.r_[starts[1:], n]
    for lo, hi in zip(starts, ends):
        m = mins[lo:hi]
        M = _decayed_cumsum(m, d)                      # after each game
        M_pre = np.r_[0.0, M[:-1]]
        for s in STATS:
            S = _decayed_cumsum(X[s][lo:hi], d)
            S_pre = np.r_[0.0, S[:-1]]
            out[f"pre_{s}"][lo:hi] = (S_pre + K * prior[s]) / (M_pre + K)
            out[f"post_{s}"][lo:hi] = (S + K * prior[s]) / (M + K)
        # EWMA of minutes (adjusted, so early values are plain means)
        num = _decayed_cumsum(m, a)
        den = _decayed_cumsum(np.ones_like(m), a)
        post = num / den
        post_min[lo:hi] = post
        pre_min[lo:hi] = np.r_[DEFAULT_MIN, post[:-1]]
        pre_games[lo:hi] = np.arange(hi - lo)
    for k, v in out.items():
        df[k] = v
    df["pre_min"], df["post_min"], df["pre_games"] = pre_min, post_min, pre_games
    return df


def latest_states(states: pd.DataFrame) -> pd.DataFrame:
    """State of every player after his most recent game (for predicting a new game)."""
    last = states.sort_values(["game_date", "game_id"]).groupby("player_id").tail(1)
    out = pd.DataFrame({"player_id": last["player_id"].to_numpy(),
                        "player_name": last["player_name"].to_numpy(),
                        "last_team": last["team"].to_numpy(),
                        "last_date": last["game_date"].to_numpy(),
                        "proj_min": last["post_min"].to_numpy(),
                        "games": last["pre_games"].to_numpy() + 1})
    for s in STATS:
        out[f"rate_{s}"] = last[f"post_{s}"].to_numpy()
    return out.set_index("player_id")


def scale_minutes(m: np.ndarray) -> np.ndarray:
    """
    Fit projected minutes to the 240-minute team total.
    Too many minutes -> the excess is taken from bench players (< ROTATION_PROTECT min)
    first, as coaches do; only if the bench cannot absorb it is everyone scaled.
    Too few minutes -> everyone scaled up. Cap MAX_MIN per player.
    """
    m = np.asarray(m, float).copy()
    if m.sum() <= 0:
        return m
    excess = m.sum() - TEAM_MINUTES
    bench = m < ROTATION_PROTECT
    if excess > 0 and bench.any():
        take = min(excess, 0.8 * m[bench].sum())   # bench keeps at least 20% of its minutes
        m[bench] *= 1.0 - take / m[bench].sum()
    for _ in range(10):
        m = m * TEAM_MINUTES / m.sum()
        over = m > MAX_MIN
        if not over.any():
            break
        m[over] = MAX_MIN
        free = ~over
        rest = TEAM_MINUTES - m[over].sum()
        if m[free].sum() <= 0:
            break
        m[free] = m[free] * rest / m[free].sum()
        if (m <= MAX_MIN + 1e-9).all():
            break
    return m


def team_vector(minutes: np.ndarray, rates: dict[str, np.ndarray]) -> dict[str, float]:
    f = {f"F_{s}": float(np.sum(minutes * rates[s])) for s in TEAM_FEATS}
    f["F_plus_minus"] /= 5.0
    return f


def team_features_from_states(states: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Historical team features per (game_id, team). The lineup = players who appeared
    (stand-in for the pre-game injury report); their minutes are PROJECTED from earlier
    games, never the minutes they actually played. Also returns scaled minutes per row.
    """
    rows = []
    m_all = np.zeros(len(states))
    pos = {k: v for k, v in zip(states.index, range(len(states)))}
    for (gid, team), g in states.groupby(["game_id", "team"], sort=False):
        m = scale_minutes(g["pre_min"].to_numpy())
        m_all[[pos[i] for i in g.index]] = m
        f = team_vector(m, {s: g[f"pre_{s}"].to_numpy() for s in TEAM_FEATS})
        f.update({"game_id": gid, "team": team})
        rows.append(f)
    return pd.DataFrame(rows), m_all


# ---------------------------------------------------------------
# Game table (possessions, efficiency) from player box scores
# ---------------------------------------------------------------
def game_table(pg: pd.DataFrame) -> pd.DataFrame:
    t = pg.groupby(["game_id", "game_date", "season", "team", "is_home"], as_index=False)[
        ["minutes", "fga", "fta", "orb", "tov", "pts"]].sum()
    t["poss"] = t["fga"] - t["orb"] + t["tov"] + 0.44 * t["fta"]
    h = t[t["is_home"]].set_index("game_id")
    a = t[~t["is_home"]].set_index("game_id")
    g = h.join(a, lsuffix="_h", rsuffix="_a", how="inner")
    poss = (g["poss_h"] + g["poss_a"]) / 2
    mins = (g["minutes_h"] + g["minutes_a"]) / 2 / 5
    out = pd.DataFrame({
        "game_id": g.index, "game_date": g["game_date_h"], "season": g["season_h"],
        "home": g["team_h"], "away": g["team_a"], "home_pts": g["pts_h"], "away_pts": g["pts_a"],
        "home_ortg": 100 * g["pts_h"] / poss, "away_ortg": 100 * g["pts_a"] / poss,
        "pace": 48 * poss / mins,
    }).reset_index(drop=True).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    long = pd.concat([out[["game_id", "game_date", "season", "home"]].rename(columns={"home": "team"}),
                      out[["game_id", "game_date", "season", "away"]].rename(columns={"away": "team"})])
    long = long.sort_values(["team", "game_date"])
    long["rest"] = long.groupby(["team", "season"])["game_date"].diff().dt.days.fillna(7).clip(1, 7)
    rest = long.set_index(["game_id", "team"])["rest"].to_dict()
    out["home_rest"] = [rest[(gi, t)] for gi, t in zip(out["game_id"], out["home"])]
    out["away_rest"] = [rest[(gi, t)] for gi, t in zip(out["game_id"], out["away"])]
    return out
