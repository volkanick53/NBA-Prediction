"""
engine/team_ratings.py
======================
Opponent-adjusted team ratings (offense / defense / pace) fitted by
recency-weighted ridge regression shrunk toward a pre-season prior.

This is the standard "adjusted efficiency" approach (KenPom / SRS family):

    ORtg(team A vs B, game g) = mu + off_A - def_B + hca*[A home] - hca*[A away] + e
    Pace(game g)              = mu_p + pace_A + pace_B + e

* off_i  : points per 100 possessions above league average the team scores
* def_i  : points per 100 the team PREVENTS (higher = better defence)
* pace_i : possessions per 48 above average the team contributes (half of game pace)

Each team's parameters are pulled toward a prior (pre-season projection) with
strength `prior_games` - i.e. the prior is worth that many games of evidence,
so early-season ratings are mostly prior and late-season mostly data. Game rows
are weighted by an exponential recency decay (half-life in days).

The pre-season prior is the team's previous-season final rating regressed toward
the mean (carryover) - computed from our own data, no external projections.
Hyper-parameters were chosen by walk-forward validation on 2022-23..2024-25
(flat optimum: prior 15 games, half-life 60 days, carryover 0.5).

These ratings are ONE input of the game model (engine/game_model.py); the other is
the player-based projected box score of tonight's lineup (engine/features.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class RatingParams:
    prior_games: float = 15.0      # ridge strength on off/def toward the prior (games of evidence)
    pace_prior_games: float = 15.0 # ridge strength on pace toward the prior
    halflife_days: float = 60.0    # recency weighting of games within the season
    hca_eff: float = 1.0           # home bonus in pts/100 poss for each side, removed before fitting
    carryover: float = 0.5         # share of last season's final rating kept as next season's prior


TUNED = RatingParams()


@dataclass
class Ratings:
    teams: list[str]
    off: dict[str, float]
    deff: dict[str, float]
    pace: dict[str, float]
    mu: float
    mu_pace: float
    n_games: dict[str, int] = field(default_factory=dict)

    def net(self, t: str) -> float:
        return self.off[t] + self.deff[t]


def fit_ratings(
    games: pd.DataFrame,
    teams: list[str],
    prior: dict[str, dict],
    mu_prior: float,
    mu_pace_prior: float,
    asof: pd.Timestamp,
    p: RatingParams = TUNED,
) -> Ratings:
    """
    games : one row per game with columns
            game_date, home, away, home_ortg, away_ortg, pace
            (only games strictly before `asof` should be passed)
    prior : team -> {"off", "def", "pace"} (deviations from league average)
    """
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    off0 = np.array([prior.get(t, {}).get("off", 0.0) for t in teams])
    def0 = np.array([prior.get(t, {}).get("def", 0.0) for t in teams])
    pac0 = np.array([prior.get(t, {}).get("pace", 0.0) for t in teams])

    n_games = {t: 0 for t in teams}
    if games is None or games.empty:
        return Ratings(teams, dict(zip(teams, off0)), dict(zip(teams, def0)),
                       dict(zip(teams, pac0)), mu_prior, mu_pace_prior, n_games)

    g = games[games["home"].isin(idx) & games["away"].isin(idx)]
    h = g["home"].map(idx).to_numpy()
    a = g["away"].map(idx).to_numpy()
    for t, c in pd.concat([g["home"], g["away"]]).value_counts().items():
        n_games[t] = int(c)

    age = (asof - pd.to_datetime(g["game_date"])).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (age / p.halflife_days)
    m = len(g)

    # ---- efficiency system: params = [mu, off(n), def(n)] ----
    k = 1 + 2 * n
    X = np.zeros((2 * m, k))
    y = np.empty(2 * m)
    r = np.arange(m)
    X[r, 0] = 1;      X[r, 1 + h] = 1;      X[r, 1 + n + a] = -1
    y[:m] = g["home_ortg"].to_numpy() - p.hca_eff
    X[m + r, 0] = 1;  X[m + r, 1 + a] = 1;  X[m + r, 1 + n + h] = -1
    y[m:] = g["away_ortg"].to_numpy() + p.hca_eff
    W = np.concatenate([w, w])

    lam = np.concatenate([[2.0], np.full(2 * n, p.prior_games)])
    th0 = np.concatenate([[mu_prior], off0, def0])
    A = (X * W[:, None]).T @ X + np.diag(lam)
    b = (X * W[:, None]).T @ y + lam * th0
    th = np.linalg.solve(A, b)
    mu, off, deff = th[0], th[1:1 + n], th[1 + n:]

    # ---- pace system: params = [mu_p, pace(n)] ----
    Xp = np.zeros((m, 1 + n))
    Xp[r, 0] = 1; Xp[r, 1 + h] += 1; Xp[r, 1 + a] += 1
    yp = g["pace"].to_numpy()
    lam_p = np.concatenate([[2.0], np.full(n, p.pace_prior_games)])
    th0p = np.concatenate([[mu_pace_prior], pac0])
    Ap = (Xp * w[:, None]).T @ Xp + np.diag(lam_p)
    bp = (Xp * w[:, None]).T @ yp + lam_p * th0p
    thp = np.linalg.solve(Ap, bp)

    # identifiability: keep team effects centred, fold the mean into mu
    off_c, def_c, pac_c = off - off.mean(), deff - deff.mean(), thp[1:] - thp[1:].mean()
    mu = mu + off.mean() - deff.mean()
    mu_p = thp[0] + 2 * thp[1:].mean()
    return Ratings(teams, dict(zip(teams, off_c)), dict(zip(teams, def_c)),
                   dict(zip(teams, pac_c)), float(mu), float(mu_p), n_games)


def season_prior(R_prev: "Ratings | None", p: RatingParams = TUNED) -> tuple[dict, float | None, float | None]:
    if R_prev is None:
        return {}, None, None
    prior = {t: {"off": p.carryover * R_prev.off[t], "def": p.carryover * R_prev.deff[t],
                 "pace": p.carryover * R_prev.pace[t]} for t in R_prev.teams}
    return prior, R_prev.mu, R_prev.mu_pace


def rating_features(R: Ratings, home: str, away: str) -> dict[str, float]:
    """Neutral-court expectations from the ratings (home edge is learnt by the game model)."""
    pace = R.mu_pace + R.pace[home] + R.pace[away]
    he = R.mu + R.off[home] - R.deff[away]
    ae = R.mu + R.off[away] - R.deff[home]
    return {"r_margin": (he - ae) * pace / 100, "r_total": (he + ae) * pace / 100, "r_pace": pace,
            "r_home_off": R.off[home], "r_home_def": R.deff[home],
            "r_away_off": R.off[away], "r_away_def": R.deff[away]}


def walk_ratings(games: pd.DataFrame, p: RatingParams = TUNED) -> tuple[pd.DataFrame, dict[str, Ratings]]:
    """
    Walk every season day by day: ratings fitted on games strictly before the day.
    Returns per-game rating features and each season's final ratings (next season's prior).
    """
    rows, finals = [], {}
    R_prev = None
    for season in sorted(games["season"].unique()):
        S = games[games["season"] == season]
        teams = sorted(set(S["home"]) | set(S["away"]))
        prior, mu0, mup0 = season_prior(R_prev, p)
        mu0 = mu0 if mu0 is not None else float(S["home_ortg"].mean())
        mup0 = mup0 if mup0 is not None else float(S["pace"].mean())
        for day, today in S.groupby("game_date", sort=True):
            R = fit_ratings(S[S["game_date"] < day], teams, prior, mu0, mup0, day, p)
            for g in today.itertuples():
                rows.append({"game_id": g.game_id, **rating_features(R, g.home, g.away),
                             "n_home": R.n_games[g.home], "n_away": R.n_games[g.away]})
        R_prev = fit_ratings(S, teams, prior, mu0, mup0, S["game_date"].max() + pd.Timedelta(days=1), p)
        finals[season] = R_prev
    return pd.DataFrame(rows), finals
