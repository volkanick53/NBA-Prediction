"""
engine/prediction_store.py
==========================
Persists every daily run to BigQuery so predictions can be graded against real results.

Tables (dataset nba_analytics, created automatically):
    pred_games            one row per game   (latest pre-tipoff prediction)
    pred_player_props     one row per player per game
Grading views (joined with the ESPN result tables written by engine/data.py):
    v_game_prediction_results
    v_player_prop_results

Only games that have NOT started yet are (re)written, so re-running the engine
during or after a game never overwrites the honest pre-game prediction.
Load jobs (not streaming inserts) are used so rows can be replaced immediately.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
from google.cloud import bigquery

from engine.data import DATASET, bq_client

log = logging.getLogger("prediction_store")

T_GAMES = f"{DATASET}.pred_games"
T_PROPS = f"{DATASET}.pred_player_props"
T_RESULTS = f"{DATASET}.espn_game_results"
T_BOX = f"{DATASET}.espn_player_boxscores"

S = bigquery.SchemaField
GAMES_SCHEMA = [
    S("game_date", "DATE"), S("generated_at", "TIMESTAMP"), S("model_version", "STRING"),
    S("game_id", "STRING"), S("home_team", "STRING"), S("away_team", "STRING"),
    S("match_time_tr", "STRING"),
    S("pred_home_score", "FLOAT"), S("pred_away_score", "FLOAT"), S("pred_spread", "FLOAT"),
    S("pred_total", "FLOAT"), S("spread_sd", "FLOAT"), S("total_sd", "FLOAT"),
    S("pred_pace", "FLOAT"), S("home_win_prob", "FLOAT"), S("predicted_winner", "STRING"),
    S("home_rest_days", "INTEGER"), S("away_rest_days", "INTEGER"),
    S("home_net_rating", "FLOAT"), S("away_net_rating", "FLOAT"), S("injuries_out", "STRING"),
    S("breakout_candidates", "STRING"),
]
PROPS_SCHEMA = [
    S("game_date", "DATE"), S("generated_at", "TIMESTAMP"), S("model_version", "STRING"),
    S("game_id", "STRING"), S("player_id", "STRING"), S("espn_id", "STRING"),
    S("player_name", "STRING"), S("team", "STRING"), S("position", "STRING"), S("role", "STRING"),
    S("proj_minutes", "FLOAT"),
    S("proj_pts", "FLOAT"), S("sd_pts", "FLOAT"), S("proj_trb", "FLOAT"), S("sd_trb", "FLOAT"),
    S("proj_ast", "FLOAT"), S("sd_ast", "FLOAT"), S("proj_fg3", "FLOAT"), S("sd_fg3", "FLOAT"),
    S("proj_stl", "FLOAT"), S("sd_stl", "FLOAT"), S("proj_blk", "FLOAT"), S("sd_blk", "FLOAT"),
    S("injury_status", "STRING"),
    # predictive distribution: 10th / 90th percentile and threshold probabilities
    *[S(f"{k}_{q}", "FLOAT") for k in ("pts", "trb", "ast", "fg3", "stl", "blk") for q in ("p10", "p90")],
    S("prob_pts_20", "FLOAT"), S("prob_pts_30", "FLOAT"), S("prob_trb_10", "FLOAT"),
    S("prob_ast_10", "FLOAT"), S("prob_fg3_3", "FLOAT"), S("probabilities_json", "STRING"),
]

V_GAMES = f"""
CREATE OR REPLACE VIEW `{DATASET}.v_game_prediction_results` AS
SELECT
  p.game_date, p.game_id, p.model_version, p.away_team, p.home_team,
  p.pred_away_score, p.pred_home_score, p.pred_spread, p.pred_total, p.home_win_prob,
  p.predicted_winner,
  r.away_pts AS actual_away_score, r.home_pts AS actual_home_score,
  r.home_pts - r.away_pts AS actual_spread, r.home_pts + r.away_pts AS actual_total,
  IF(r.game_id IS NULL, NULL, (p.home_win_prob >= 0.5) = (r.home_pts > r.away_pts)) AS winner_correct,
  ABS(p.pred_spread - (r.home_pts - r.away_pts)) AS spread_abs_error,
  ABS(p.pred_total - (r.home_pts + r.away_pts)) AS total_abs_error,
  POW(p.home_win_prob - IF(r.home_pts > r.away_pts, 1, 0), 2) AS brier,
  -IF(r.home_pts > r.away_pts, LN(p.home_win_prob), LN(1 - p.home_win_prob)) AS log_loss,
  p.generated_at
FROM `{T_GAMES}` p
LEFT JOIN `{T_RESULTS}` r ON r.game_id = p.game_id
"""

V_PROPS = f"""
CREATE OR REPLACE VIEW `{DATASET}.v_player_prop_results` AS
SELECT
  p.game_date, p.game_id, p.model_version, p.player_name, p.team, p.role, p.injury_status,
  p.proj_minutes, b.minutes AS actual_minutes,
  p.proj_pts, p.pts_p10, p.pts_p90, b.pts AS actual_pts, p.prob_pts_20,
  p.proj_trb, b.trb AS actual_trb,
  p.proj_ast, b.ast AS actual_ast, p.proj_fg3, b.fg3m AS actual_fg3,
  p.proj_stl, b.stl AS actual_stl, p.proj_blk, b.blk AS actual_blk,
  r.game_id IS NOT NULL AS game_final,
  b.espn_id IS NULL AND r.game_id IS NOT NULL AS did_not_play
FROM `{T_PROPS}` p
LEFT JOIN `{T_RESULTS}` r ON r.game_id = p.game_id
LEFT JOIN `{T_BOX}` b ON b.game_id = p.game_id AND b.espn_id = p.espn_id
"""


def _ensure(client: bigquery.Client, table: str, schema: list) -> None:
    try:
        t = client.get_table(table)
        have = {f.name for f in t.schema}
        missing = [f for f in schema if f.name not in have]
        if missing:                       # schema evolution: append new nullable columns
            t.schema = list(t.schema) + missing
            client.update_table(t, ["schema"])
            log.info("Added %d columns to %s", len(missing), table)
    except Exception as exc:
        if "Not found" not in str(exc) and "404" not in str(exc):
            raise
        t = bigquery.Table(table, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(field="game_date")
        client.create_table(t)
        log.info("Created %s", table)


def reset_tables() -> None:
    """Drop and recreate the prediction tables (removes all stored predictions)."""
    client = bq_client()
    for t, sch in ((T_GAMES, GAMES_SCHEMA), (T_PROPS, PROPS_SCHEMA)):
        client.delete_table(t, not_found_ok=True)
        _ensure(client, t, sch)
    client.delete_table(f"{DATASET}.prediction_history", not_found_ok=True)
    log.info("Prediction tables reset.")


def save_predictions(output: dict, model_version: str) -> dict:
    """Upsert not-yet-started games of this run and refresh the grading views."""
    client = bq_client()
    _ensure(client, T_GAMES, GAMES_SCHEMA)
    _ensure(client, T_PROPS, PROPS_SCHEMA)

    games = [g for g in output["games"] if g.get("status", "STATUS_SCHEDULED") == "STATUS_SCHEDULED"]
    skipped = len(output["games"]) - len(games)
    if not games:
        log.info("No pre-tipoff games to store (%d already started/final).", skipped)
        return {"games": 0, "props": 0, "skipped_started": skipped}

    gdate = datetime.strptime(output["date"], "%Y%m%d").date()
    gen = pd.Timestamp(output["generated_at"])
    ids = [g["game_id"] for g in games]
    gdf = pd.DataFrame([{
        "game_date": gdate, "generated_at": gen, "model_version": model_version,
        "game_id": g["game_id"], "home_team": g["home_team"], "away_team": g["away_team"],
        "match_time_tr": g.get("match_time_tr"),
        "pred_home_score": g["proj_home_score"], "pred_away_score": g["proj_away_score"],
        "pred_spread": g["proj_spread"], "pred_total": g["proj_total"],
        "spread_sd": g.get("spread_sd"), "total_sd": g.get("total_sd"),
        "pred_pace": g["pred_game_pace"], "home_win_prob": g["home_win_prob"] / 100.0,
        "predicted_winner": g["predicted_winner"],
        "home_rest_days": g["home_rest_days"], "away_rest_days": g["away_rest_days"],
        "home_net_rating": g.get("home_net_rating"), "away_net_rating": g.get("away_net_rating"),
        "injuries_out": "; ".join(f"{t}: {r['player']}" for t, rep in g.get("injury_report", {}).items()
                                  for r in rep if r.get("excluded")),
        "breakout_candidates": json.dumps(g.get("breakout_candidates", []), ensure_ascii=False),
    } for g in games])
    pdf = pd.DataFrame([{
        "game_date": gdate, "generated_at": gen, "model_version": model_version,
        "game_id": p["game_id"], "player_id": str(p["player_id"]), "espn_id": p.get("espn_id"),
        "player_name": p["player_name"], "team": p["team_id"], "position": p.get("position"),
        "role": p.get("role"), "proj_minutes": p["projected_mins"],
        **{f"{k}_{s}": p.get(f"{k}_{s}") for s in ("pts", "trb", "ast", "fg3", "stl", "blk")
           for k in ("proj", "sd")},
        "injury_status": p.get("injury_status"),
        **{f"{k}_{q}": p.get(f"{k}_{q}") for k in ("pts", "trb", "ast", "fg3", "stl", "blk")
           for q in ("p10", "p90")},
        "prob_pts_20": p.get("probabilities", {}).get("pts_20+"),
        "prob_pts_30": p.get("probabilities", {}).get("pts_30+"),
        "prob_trb_10": p.get("probabilities", {}).get("trb_10+"),
        "prob_ast_10": p.get("probabilities", {}).get("ast_10+"),
        "prob_fg3_3": p.get("probabilities", {}).get("fg3_3+"),
        "probabilities_json": json.dumps(p.get("probabilities", {})),
    } for p in output["player_props"] if p["game_id"] in ids])

    params = [bigquery.ArrayQueryParameter("ids", "STRING", ids)]
    for t in (T_GAMES, T_PROPS):
        client.query(f"DELETE FROM `{t}` WHERE game_id IN UNNEST(@ids)",
                     job_config=bigquery.QueryJobConfig(query_parameters=params)).result()
    cfg = lambda sch: bigquery.LoadJobConfig(schema=sch, write_disposition="WRITE_APPEND")  # noqa: E731
    client.load_table_from_dataframe(gdf, T_GAMES, job_config=cfg(GAMES_SCHEMA)).result()
    if not pdf.empty:
        client.load_table_from_dataframe(pdf[[f.name for f in PROPS_SCHEMA]], T_PROPS,
                                         job_config=cfg(PROPS_SCHEMA)).result()
    for sql in (V_GAMES, V_PROPS):
        client.query(sql).result()
    log.info("Stored %d games / %d props in BigQuery (%d already started, kept as-is).",
             len(gdf), len(pdf), skipped)
    return {"games": len(gdf), "props": len(pdf), "skipped_started": skipped}
