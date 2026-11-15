"""
engine/live_engine.py
=====================
Game-day prediction engine (replaces the ID-mismatched engine/predict_daily.py).

    python -m engine.live_engine --date 20261021
    python -m engine.live_engine --date 20261021 --write-bq     # also persist to BigQuery

Pipeline
--------
1. ESPN scoreboard  -> the day's games (authoritative fixture list).
2. ESPN rosters     -> who is on each team *today* (trades/signings handled).
3. ESPN injuries    -> Out/Doubtful players removed from rotations (+ team-strength penalty).
4. BigQuery history -> this season's box scores (fact_player_boxscores) and game
                       summaries (fact_game_summaries) read *before* game day.
5. Blend            -> pre-season prior (MARCELS / roster net rating) and in-season
                       form via Bayesian weight  w = n / (n + K), recency-weighted
                       (EWMA, half-life 10 games). Zero games -> pure prior.
6. Predict          -> the same calibrated matchup logic as predict_oct20_offline.py
                       (ratings + XGBoost blend, 240-minute rotation, props, SDs).

ESPN ids never touch our history tables: players are matched by normalised name
to Basketball-Reference ids, teams by abbreviation (core.bbref).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path

import db_dtypes  # noqa: F401  (BigQuery DATE -> pandas)
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

import predict_oct20_offline as core
from engine import espn_client as espn

log = logging.getLogger("live_engine")

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT = "nba-analytics-503718"
DATASET = f"{PROJECT}.nba_analytics"
HISTORY_TABLE = f"{DATASET}.prediction_history"

EWMA_HALFLIFE = 10
K_TEAM = core.K_TEAM_PRIOR       # games of pseudo-evidence behind the team prior
K_PLAYER = core.K_PLAYER_PRIOR   # ... behind the player prior

# prior-dict key -> box score column
PLAYER_FIELDS = {"mpg": "minutes", "ppg": "pts", "rpg": "trb", "apg": "ast",
                 "spg": "stl", "bpg": "blk", "fg3m": "fg3", "usg": "usg_pct", "ts": "ts_pct"}


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def season_for(d: date_cls) -> str:
    y = d.year if d.month >= 9 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def ewma(values: np.ndarray, halflife: float = EWMA_HALFLIFE) -> float:
    """Recency-weighted mean; the last element has weight 1."""
    n = len(values)
    w = 0.5 ** (np.arange(n)[::-1] / halflife)
    return float((np.asarray(values, dtype=float) * w).sum() / w.sum())


def bq_client() -> bigquery.Client:
    creds = service_account.Credentials.from_service_account_file(KEY_PATH)
    return bigquery.Client(credentials=creds, project=PROJECT)


# ---------------------------------------------------------------
# BigQuery history
# ---------------------------------------------------------------
def fetch_player_history(client, season: str, day: date_cls, ids: list[str]) -> dict[str, pd.DataFrame]:
    if not ids:
        return {}
    sql = f"""
        SELECT player_id, game_date, minutes, pts, trb, ast, fg3, stl, blk, usg_pct, ts_pct
        FROM `{DATASET}.fact_player_boxscores`
        WHERE season = @season AND game_date < @d AND player_id IN UNNEST(@ids)
        ORDER BY player_id, game_date
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("season", "STRING", season),
        bigquery.ScalarQueryParameter("d", "DATE", day),
        bigquery.ArrayQueryParameter("ids", "STRING", ids),
    ])
    df = client.query(sql, job_config=cfg).to_dataframe()
    for c in ["minutes", "pts", "trb", "ast", "fg3", "stl", "blk", "usg_pct", "ts_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["minutes"].fillna(0) > 0]
    return {pid: g.reset_index(drop=True) for pid, g in df.groupby("player_id")}


def fetch_team_history(client, season: str, day: date_cls) -> dict[str, pd.DataFrame]:
    """Per team (BBRef abbr): own ORtg, DRtg (= opponent ORtg), pace, date per game."""
    sql = f"""
        SELECT game_date, home_team, away_team, pace, home_off_rtg, away_off_rtg
        FROM `{DATASET}.fact_game_summaries`
        WHERE season = @season AND game_date < @d
        ORDER BY game_date
    """
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("season", "STRING", season),
        bigquery.ScalarQueryParameter("d", "DATE", day),
    ])
    df = client.query(sql, job_config=cfg).to_dataframe()
    for c in ["pace", "home_off_rtg", "away_off_rtg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rows = []
    for r in df.itertuples():
        rows.append((core.bbref(r.home_team), r.game_date, r.home_off_rtg, r.away_off_rtg, r.pace))
        rows.append((core.bbref(r.away_team), r.game_date, r.away_off_rtg, r.home_off_rtg, r.pace))
    t = pd.DataFrame(rows, columns=["team", "game_date", "ortg", "drtg", "pace"]).dropna()
    return {k: g.sort_values("game_date").reset_index(drop=True) for k, g in t.groupby("team")}


# ---------------------------------------------------------------
# Prior <-> in-season blending
# ---------------------------------------------------------------
def blend_player(prior: dict, hist: pd.DataFrame | None) -> tuple[dict, float]:
    if hist is None or hist.empty:
        return dict(prior), 0.0
    n = len(hist)
    w = n / (n + K_PLAYER)
    out = dict(prior)
    for key, col in PLAYER_FIELDS.items():
        vals = hist[col].dropna().to_numpy()
        if len(vals):
            out[key] = w * ewma(vals) + (1 - w) * prior[key]
    out["games_this_season"] = n
    return out, w


def blend_team(prior: dict, hist: pd.DataFrame | None) -> tuple[dict, float]:
    if hist is None or hist.empty:
        return dict(prior), 0.0
    n = len(hist)
    w = n / (n + K_TEAM)
    out = dict(prior)
    out["ortg"] = w * ewma(hist["ortg"].to_numpy()) + (1 - w) * prior["ortg"]
    out["drtg"] = w * ewma(hist["drtg"].to_numpy()) + (1 - w) * prior["drtg"]
    out["pace"] = w * ewma(hist["pace"].to_numpy()) + (1 - w) * prior["pace"]
    out["net"] = out["ortg"] - out["drtg"]
    return out, w


# ---------------------------------------------------------------
# ESPN roster -> our player records
# ---------------------------------------------------------------
def build_name_index(player_priors: dict[str, list[dict]]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = {}
    for plist in player_priors.values():
        for p in plist:
            idx.setdefault(core.norm_name(p["name"]), []).append(p)
    return idx


def resolve_roster(espn_team_id: int, team: str, name_idx: dict) -> tuple[list[dict], list[str]]:
    matched, unmatched = [], []
    for a in espn.get_roster(espn_team_id):
        cands = name_idx.get(core.norm_name(a["name"]))
        if not cands:
            unmatched.append(a["name"])
            continue
        pick = next((c for c in cands if c["team_id"] == team), cands[0])
        matched.append({**pick, "team_id": team})
    return matched, unmatched


# ---------------------------------------------------------------
# Output
# ---------------------------------------------------------------
def to_tr_time(iso_utc: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        return (dt + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return ""


def write_history(output: dict) -> None:
    """Persist predictions for later accuracy tracking (table auto-created)."""
    client = bq_client()
    schema = [
        bigquery.SchemaField("prediction_date", "DATE"),
        bigquery.SchemaField("generated_at", "TIMESTAMP"),
        bigquery.SchemaField("game_id", "STRING"),
        bigquery.SchemaField("home_team", "STRING"),
        bigquery.SchemaField("away_team", "STRING"),
        bigquery.SchemaField("pred_home_score", "FLOAT"),
        bigquery.SchemaField("pred_away_score", "FLOAT"),
        bigquery.SchemaField("pred_spread", "FLOAT"),
        bigquery.SchemaField("pred_total", "FLOAT"),
        bigquery.SchemaField("home_win_prob", "FLOAT"),
        bigquery.SchemaField("game_json", "STRING"),
        bigquery.SchemaField("props_json", "STRING"),
    ]
    try:
        client.get_table(HISTORY_TABLE)
    except Exception:
        tbl = bigquery.Table(HISTORY_TABLE, schema=schema)
        tbl.time_partitioning = bigquery.TimePartitioning(field="prediction_date")
        client.create_table(tbl)
        log.info("Created %s", HISTORY_TABLE)

    pdate = datetime.strptime(output["date"], "%Y%m%d").date().isoformat()
    props_by_game: dict[str, list] = {}
    for p in output["player_props"]:
        props_by_game.setdefault(p["game_id"], []).append(p)
    rows = [{
        "prediction_date": pdate,
        "generated_at": output["generated_at"],
        "game_id": g["game_id"],
        "home_team": g["home_team"], "away_team": g["away_team"],
        "pred_home_score": g["proj_home_score"], "pred_away_score": g["proj_away_score"],
        "pred_spread": g["proj_spread"], "pred_total": g["proj_total"],
        "home_win_prob": g["home_win_prob"] / 100.0,
        "game_json": json.dumps(g, ensure_ascii=False),
        "props_json": json.dumps(props_by_game.get(g["game_id"], []), ensure_ascii=False),
    } for g in output["games"]]
    errs = client.insert_rows_json(HISTORY_TABLE, rows)
    if errs:
        raise RuntimeError(f"BigQuery insert failed: {errs}")
    log.info("%d games written to %s", len(rows), HISTORY_TABLE)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run(date_str: str, season: str | None = None, use_injuries: bool = True,
        use_bq: bool = True, write_bq: bool = False, out_dir: Path = Path("output")) -> dict | None:
    day = datetime.strptime(date_str, "%Y%m%d").date()
    season = season or season_for(day)
    out_dir.mkdir(exist_ok=True)
    log.info("=" * 65)
    log.info("Live engine | date %s | season %s", date_str, season)

    games = espn.get_scoreboard(date_str)
    if not games:
        log.warning("No games on %s", date_str)
        return None

    registry = core.ModelRegistry()
    team_priors, lg_rtg, lg_pace = core.load_team_priors()
    player_priors = core.load_player_priors()
    name_idx = build_name_index(player_priors)
    injuries = core.load_injuries() if use_injuries else {}

    sched = pd.read_csv(core.SCHEDULE_FILE)
    rest_lookup = core.build_rest_lookup(sched)
    cup = {(core.bbref(r.home_team), core.bbref(r.away_team), r.game_date): bool(r.is_nba_cup_group)
           for r in sched.itertuples()}
    iso_day = day.isoformat()

    # --- rosters (ESPN, today) ---
    rosters: dict[str, list[dict]] = {}
    unmatched_all: dict[str, list[str]] = {}
    for g in games:
        for tid, abbr in ((g["home_team_id"], g["home_team_abbr"]), (g["away_team_id"], g["away_team_abbr"])):
            t = core.bbref(abbr)
            if t not in rosters:
                rosters[t], um = resolve_roster(tid, t, name_idx)
                if um:
                    unmatched_all[t] = um

    # --- BigQuery history ---
    p_hist: dict[str, pd.DataFrame] = {}
    t_hist: dict[str, pd.DataFrame] = {}
    if use_bq:
        try:
            client = bq_client()
            ids = sorted({p["player_id"] for pl in rosters.values() for p in pl})
            p_hist = fetch_player_history(client, season, day, ids)
            t_hist = fetch_team_history(client, season, day)
            log.info("History: %d players, %d teams with %s games before %s",
                     len(p_hist), len(t_hist), season, iso_day)
        except Exception as exc:
            log.warning("BigQuery unavailable (%s) - falling back to pre-season priors only.", exc)

    all_games, all_props = [], []
    for g in games:
        home, away = core.bbref(g["home_team_abbr"]), core.bbref(g["away_team_abbr"])
        if home not in team_priors or away not in team_priors:
            log.warning("Skipping %s @ %s (no team prior)", away, home)
            continue

        hp, h_w = blend_team(team_priors[home], t_hist.get(home))
        ap, a_w = blend_team(team_priors[away], t_hist.get(away))

        def players(t):
            out = []
            for p in rosters.get(t, []):
                bp, _ = blend_player(p, p_hist.get(p["player_id"]))
                out.append(bp)
            out.sort(key=lambda x: x["mpg"], reverse=True)
            return out

        def rest(t):
            h = t_hist.get(t)
            if h is not None and len(h):
                last = pd.Timestamp(h["game_date"].iloc[-1]).date()
                return int(np.clip((day - last).days, 1, core.REST_DAYS_CAP))
            return rest_lookup.get((t, iso_day), core.REST_DAYS_CAP)

        game, props = core.predict_matchup(
            registry, home, away, str(g["event_id"]), iso_day, to_tr_time(g["game_date"]),
            cup.get((home, away, iso_day), False),
            hp, ap, players(home), players(away), rest(home), rest(away),
            (home not in t_hist or away not in t_hist), lg_rtg, lg_pace, injuries, h_w, a_w,
        )
        all_games.append(game)
        all_props.extend(props)
        log.info("  %s @ %s -> %.1f - %.1f | spread %+.1f | home %.1f%% | in-season w %.2f/%.2f",
                 away, home, game["proj_away_score"], game["proj_home_score"],
                 game["proj_spread"], game["home_win_prob"], a_w, h_w)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "season": season,
        "methodology": "ESPN roster/injuries + BigQuery in-season history, Bayesian blend with "
                       "pre-season prior (K_team=%d, K_player=%d) -> calibrated ratings/XGBoost" % (K_TEAM, K_PLAYER),
        "unmatched_roster_players": unmatched_all,
        "games": all_games,
        "player_props": all_props,
    }
    (out_dir / f"predictions_{date_str}.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "predictions_latest.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    n_um = sum(len(v) for v in unmatched_all.values())
    log.info("Done: %d games, %d player lines; %d roster players without prior (rookies/two-way)",
             len(all_games), len(all_props), n_um)
    if write_bq:
        write_history(output)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="NBA live prediction engine")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"), help="YYYYMMDD (ET game date)")
    ap.add_argument("--season", default=None, help="override season label, e.g. 2025-26 (testing)")
    ap.add_argument("--no-injuries", action="store_true")
    ap.add_argument("--no-bq", action="store_true", help="skip BigQuery history (pure priors)")
    ap.add_argument("--write-bq", action="store_true", help="persist to prediction_history table")
    ap.add_argument("--out-dir", default="output")
    a = ap.parse_args()
    run(a.date, a.season, not a.no_injuries, not a.no_bq, a.write_bq, Path(a.out_dir))
