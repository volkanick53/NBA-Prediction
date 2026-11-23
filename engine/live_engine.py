"""
engine/live_engine.py
=====================
Game-day prediction engine (model v3).

    python -m engine.live_engine                  # today's slate (US Eastern date)
    python -m engine.live_engine --days-ahead 1   # tomorrow's slate
    python -m engine.live_engine --date 20261124  # a specific date
    python -m engine.live_engine --no-store       # JSON only, no BigQuery writes

Pipeline (same functions as models/train.py, so what was validated is what runs)
--------
1. Data       BigQuery fact_player_boxscores (5 completed seasons) + current season
              from ESPN, pushed to BigQuery (espn_* tables, v_player_games view).
2. Players    leakage-free shrunk per-minute rates and projected minutes for every
              player (engine/features.py).
3. Lineups    ESPN roster minus Out/Doubtful players on the ESPN injury report;
              projected minutes scaled to 240 -> projected team box score.
4. Ratings    opponent-adjusted team offense/defense/pace (engine/team_ratings.py),
              prior = last season's final rating regressed to the mean.
5. Game model ridge regressions fitted on real outcomes (engine/game_model.py):
              margin, total (+ in-season scoring-level correction), win probability.
6. Props      gradient-boosted per-stat models on rates x projected minutes + context.

No betting lines or third-party projections are used.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from engine import data, espn_client as espn, espn_history as hist
from engine.features import (TEAM_FEATS, FeatureParams, game_table, latest_states, player_states,
                             scale_minutes, team_features_from_states, team_vector)
from engine.game_model import (PROP_FEATS, PROP_RATE_STATS, PROP_STATS, PROP_THRESHOLDS, GameModel,
                               game_dataset, game_row, prop_rows)
from engine.team_ratings import RatingParams, fit_ratings, rating_features, season_prior, walk_ratings

log = logging.getLogger("live_engine")

MODEL_VERSION = "v3"
OUT_STATUSES = {"Out", "Doubtful", "Injured Reserve", "Suspended", "Out For Season"}
RECENT_TEAM_GAMES = 10      # "in the rotation" = appeared in one of the team's last N games
REGULAR_MIN = 15.0          # ... or a regular (e.g. returning from an absence)
MIN_PROP_MINUTES = 5.0
ROLE_PTS = 12.0             # "role player" for breakout candidates = projected under 12 points

TEAM_FULL_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
    "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}
TO_ESPN = {"NYK": "NY", "SAS": "SA", "NOP": "NO", "UTA": "UTAH", "BRK": "BKN", "CHO": "CHA",
           "GSW": "GS", "PHO": "PHX", "WAS": "WSH"}
LOGO_TPL = "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr}.png"


def espn_abbr(t: str) -> str:
    return TO_ESPN.get(t, t)


def us_today() -> date_cls:
    """NBA slates are keyed by US Eastern date (a TR-night game belongs to the previous US day)."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def to_tr_time(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        return (dt + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return ""


def rest_days(day: date_cls, teams: set[str]) -> dict[str, int]:
    """Days since each team's previous game (schedule-based: includes games not yet final)."""
    rest = {}
    for back in range(1, 8):
        for g in hist.scoreboard(day - timedelta(days=back), day):
            for t in (hist.bbref(g["home_team_abbr"]), hist.bbref(g["away_team_abbr"])):
                if t in teams and t not in rest:
                    rest[t] = back
        if len(rest) == len(teams):
            break
    return {t: rest.get(t, 7) for t in teams}


# ---------------------------------------------------------------
# Lineup for one team
# ---------------------------------------------------------------
def build_lineup(team: str, espn_team_id: int, latest: pd.DataFrame, pg: pd.DataFrame,
                 id_map: dict, name_idx: dict, injuries: dict) -> dict:
    sg = pg[(pg["team"] == team) & (pg["season"] == data.CURRENT_SEASON)]
    recent = (sg[["game_id", "game_date"]].drop_duplicates().sort_values("game_date")
              ["game_id"].tail(RECENT_TEAM_GAMES))
    appear = sg[sg["game_id"].isin(recent)].groupby("player_id").size() / max(len(recent), 1)
    recent_players = set(appear.index)

    rows, report, no_history, not_in_rotation = [], [], [], []
    for a in espn.get_roster(espn_team_id):
        eid = str(a["player_id"])
        pid = id_map.get(eid) or data.map_espn(a["name"], team, name_idx) or f"espn_{eid}"
        inj = injuries.get(eid)
        out = bool(inj and inj["status"] in OUT_STATUSES)
        if inj:
            report.append({"player": a["name"], "status": inj["status"], "excluded": out,
                           "note": (inj.get("description") or "")[:240]})
        if pid not in latest.index:
            no_history.append(a["name"])
            continue
        st = latest.loc[pid]
        if out:
            continue
        if pid not in recent_players and st["proj_min"] < REGULAR_MIN:
            not_in_rotation.append(a["name"])
            continue
        # P(plays): a healthy regular plays (covers returns from injury and new arrivals);
        # anyone else = share of the team's recent games he appeared in.
        if st["proj_min"] >= REGULAR_MIN and not inj:
            p_play = 1.0
        else:
            p_play = float(appear.get(pid, 0.0)) or 1.0
        rows.append({"player_id": pid, "espn_id": eid, "name": a["name"],
                     "position": a.get("position", ""),
                     "injury_status": inj["status"] if inj else None, "p_play": p_play,
                     "pre_min": float(st["proj_min"]), "pre_games": float(st["games"]),
                     **{f"pre_{s}": float(st[f"rate_{s}"]) for s in PROP_RATE_STATS}})
    P = pd.DataFrame(rows)
    if P.empty:
        return {"players": P, "F": None, "report": report, "no_history": no_history}
    # expected minutes = P(plays) x minutes when playing, scaled to the 240-minute team total
    P["m"] = scale_minutes((P["p_play"] * P["pre_min"]).to_numpy())
    F = team_vector(P["m"].to_numpy(), {s: P[f"pre_{s}"].to_numpy() for s in TEAM_FEATS})
    return {"players": P, "F": F, "report": report, "no_history": no_history,
            "not_in_rotation": not_in_rotation}


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run(date_str: str, use_injuries: bool = True, store: bool = True,
        out_dir: Path = Path("output")) -> dict | None:
    day = datetime.strptime(date_str, "%Y%m%d").date()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("Live engine %s | date %s", MODEL_VERSION, date_str)

    slate = espn.get_scoreboard(date_str)
    if not slate:
        log.warning("No games on %s", date_str)
        return None

    model = GameModel.load()
    fp = FeatureParams(**model.meta["feature_params"])
    rp = RatingParams(**model.meta["rating_params"])
    prior = model.meta["prior_rates"]

    # 1-2. data + player states (everything strictly before `day`)
    pg, _, id_map = data.player_games(day, push_bq=store)
    states = player_states(pg, prior, fp)
    latest = latest_states(states)
    name_idx = data.build_name_index(pg[~pg["player_id"].str.startswith("espn_")])

    # 4. ratings: walk all seasons (prior chain + current-season features for the level correction)
    games = game_table(pg)
    rfeat, finals = walk_ratings(games, rp)
    cur = games[games["season"] == data.CURRENT_SEASON]
    prev_season = max(s for s in finals if s < data.CURRENT_SEASON)
    pri, mu0, mup0 = season_prior(finals[prev_season], rp)
    R = fit_ratings(cur, sorted(finals[prev_season].teams), pri, mu0, mup0, pd.Timestamp(day), rp)

    # in-season scoring level: shrunk mean residual of this season's games (all before `day`)
    level = 0.0
    n0 = model.meta.get("total_level_n0")
    if n0 and not cur.empty:
        TF, _ = team_features_from_states(states[states["season"] == data.CURRENT_SEASON])
        Dc = game_dataset(cur, rfeat, TF)
        res = Dc["total"].to_numpy() - model.total.predict(Dc)
        level = float(res.sum() / (len(res) + n0))
    log.info("Season games so far: %d | total-level correction %+.2f", len(cur), level)

    # 3. injuries
    injuries: dict[str, dict] = {}
    if use_injuries:
        try:
            for lst in espn.get_injuries().values():
                for i in lst:
                    injuries[str(i["player_id"])] = i
        except Exception as exc:
            log.warning("Injury report unavailable (%s) - all rostered players assumed available.", exc)

    teams_today = {hist.bbref(g[k]) for g in slate for k in ("home_team_abbr", "away_team_abbr")}
    rest = rest_days(day, teams_today)

    all_games, all_props, data_gaps = [], [], {}
    for g in slate:
        home, away = hist.bbref(g["home_team_abbr"]), hist.bbref(g["away_team_abbr"])
        if home not in R.off or away not in R.off:
            log.warning("Skipping %s @ %s (unknown team)", away, home)
            continue
        hl = build_lineup(home, g["home_team_id"], latest, pg, id_map, name_idx, injuries)
        al = build_lineup(away, g["away_team_id"], latest, pg, id_map, name_idx, injuries)
        if hl["F"] is None or al["F"] is None:
            log.warning("Skipping %s @ %s (no lineup)", away, home)
            continue
        for t, L in ((home, hl), (away, al)):
            if L["no_history"]:
                data_gaps[espn_abbr(t)] = L["no_history"]

        rf = rating_features(R, home, away)
        X = pd.DataFrame([game_row(hl["F"], al["F"], rf, rest[home], rest[away])])
        pr = model.predict_games(X).iloc[0]
        margin = float(pr["margin"])
        total = float(pr["total"]) + level
        hwp = float(norm.cdf(margin / model.sigma)) * 100
        h_pts, a_pts = total / 2 + margin / 2, total / 2 - margin / 2

        props = []
        for L, team, is_home, Fo, own, oth in ((hl, home, True, al["F"], home, away),
                                               (al, away, False, hl["F"], away, home)):
            P = L["players"]
            Xp = prop_rows(P, L["F"], Fo, {"off": R.off[own], "def": R.deff[own]},
                           {"off": R.off[oth], "def": R.deff[oth]}, rf["r_pace"], is_home, rest[team])
            preds = pd.DataFrame(index=P.index)
            for s in PROP_STATS:
                if model.meta["prop_method"].get(s) == "xgboost":
                    preds[s] = np.clip(model.prop_model(s).predict(Xp[PROP_FEATS]), 0, None)
                else:
                    preds[s] = Xp[f"x_{s}"]
            for rank, i in enumerate(P["m"].sort_values(ascending=False).index):
                if P.at[i, "m"] < MIN_PROP_MINUTES:
                    continue
                row = {"game_id": str(g["event_id"]), "player_id": P.at[i, "player_id"],
                       "espn_id": P.at[i, "espn_id"], "player_name": P.at[i, "name"],
                       "team_id": espn_abbr(team), "position": P.at[i, "position"],
                       "role": "starter" if rank < 5 else "bench",
                       "projected_mins": round(float(P.at[i, "m"]), 1),
                       "injury_status": P.at[i, "injury_status"]
                       if isinstance(P.at[i, "injury_status"], str) else None}
                probs = {}
                for s in PROP_STATS:
                    key = "fg3" if s == "fg3m" else s
                    v = float(preds.at[i, s])
                    row[f"proj_{key}"] = round(v, 1)
                    dist = model.prop_dist.get(s)
                    if dist is None:      # model trained before distributions existed
                        row[f"sd_{key}"] = round(model.prop_sd.get(s, 1.0) * np.sqrt(v + 1.0), 2)
                        continue
                    p10, p90 = float(dist.quantile(v, 0.1)[0]), float(dist.quantile(v, 0.9)[0])
                    row[f"{key}_p10"], row[f"{key}_p90"] = round(p10, 1), round(p90, 1)
                    row[f"sd_{key}"] = round((p90 - p10) / 2.563, 2)
                    for k in PROP_THRESHOLDS[s]:
                        probs[f"{key}_{k}+"] = round(float(dist.prob_at_least(v, k)[0]), 3)
                row["probabilities"] = probs
                props.append(row)

        # role players with the highest chance of a 20-point night (calibrated on real games)
        role = [p for p in props if p["proj_pts"] < ROLE_PTS and p.get("probabilities")]
        breakouts = [{"player": p["player_name"], "team": p["team_id"], "proj_pts": p["proj_pts"],
                      "pts_p90": p.get("pts_p90"), "prob_20_plus": p["probabilities"].get("pts_20+")}
                     for p in sorted(role, key=lambda p: -p["probabilities"].get("pts_20+", 0))[:3]]

        game = {
            "game_id": str(g["event_id"]), "game_date": date_str,
            "match_time_tr": to_tr_time(g["game_date"]), "status": g.get("status", ""),
            "home_team": espn_abbr(home), "away_team": espn_abbr(away),
            "home_team_full": TEAM_FULL_NAMES.get(home, home),
            "away_team_full": TEAM_FULL_NAMES.get(away, away),
            "home_logo": LOGO_TPL.format(abbr=espn_abbr(home).lower()),
            "away_logo": LOGO_TPL.format(abbr=espn_abbr(away).lower()),
            "proj_home_score": round(h_pts, 1), "proj_away_score": round(a_pts, 1),
            "proj_spread": round(margin, 1), "proj_total": round(total, 1),
            "spread_sd": round(model.sigma, 1), "total_sd": round(model.total_sigma, 1),
            "pred_game_pace": round(rf["r_pace"], 1),
            "home_win_prob": round(hwp, 1), "away_win_prob": round(100 - hwp, 1),
            "predicted_winner": espn_abbr(home if hwp >= 50 else away),
            "home_rest_days": rest[home], "away_rest_days": rest[away],
            "home_net_rating": round(R.off[home] + R.deff[home], 2),
            "away_net_rating": round(R.off[away] + R.deff[away], 2),
            "injury_report": {espn_abbr(home): hl["report"], espn_abbr(away): al["report"]},
            "breakout_candidates": breakouts,
            "model_breakdown": {
                "rating_margin": round(rf["r_margin"], 2),
                "home_projected_box": {k: round(v, 1) for k, v in hl["F"].items()},
                "away_projected_box": {k: round(v, 1) for k, v in al["F"].items()},
                "home_lineup_size": len(hl["players"]), "away_lineup_size": len(al["players"]),
                "games_played": [R.n_games.get(home, 0), R.n_games.get(away, 0)],
                "total_level_correction": round(level, 2),
            },
        }
        all_games.append(game)
        all_props.extend(props)
        log.info("  %s @ %s -> %.1f - %.1f | spread %+.1f | total %.1f | home %.1f%% | rest %d/%d",
                 away, home, a_pts, h_pts, margin, total, hwp, rest[away], rest[home])

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date_str, "season": data.CURRENT_SEASON, "model_version": MODEL_VERSION,
        "season_games_used": int(len(cur)),
        "methodology": ("Player-lineup projected box score (5 seasons BigQuery + current ESPN) + "
                        "opponent-adjusted team ratings -> ridge models fitted on real outcomes; "
                        "props: gradient boosting on rates x projected minutes. No betting data."),
        "players_without_history": data_gaps,
        "games": all_games, "player_props": all_props,
    }
    for name in (f"predictions_{date_str}.json", "predictions_latest.json"):
        (out_dir / name).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Done: %d games, %d player lines", len(all_games), len(all_props))
    if store:
        try:
            from engine import prediction_store
            prediction_store.save_predictions(output, MODEL_VERSION)
        except Exception as exc:   # JSON output is already on disk
            log.error("BigQuery store failed (%s); predictions kept in %s", exc, out_dir)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description="NBA live prediction engine")
    ap.add_argument("--date", default=None, help="YYYYMMDD (US game date); default = today in US Eastern")
    ap.add_argument("--days-ahead", type=int, default=0, help="offset from today, e.g. 1 = tomorrow")
    ap.add_argument("--no-injuries", action="store_true")
    ap.add_argument("--no-store", action="store_true", help="do not write to BigQuery")
    ap.add_argument("--out-dir", default="output")
    a = ap.parse_args()
    d = a.date or (us_today() + timedelta(days=a.days_ahead)).strftime("%Y%m%d")
    run(d, not a.no_injuries, not a.no_store, Path(a.out_dir))
