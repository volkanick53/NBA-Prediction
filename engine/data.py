"""
engine/data.py
==============
Single source of player-game data for training and live prediction.

    completed seasons : BigQuery  nba_analytics.fact_player_boxscores  (Basketball-Reference)
    current season    : ESPN API  -> written to BigQuery  nba_analytics.espn_player_boxscores
    unified view      : BigQuery  nba_analytics.v_player_games  (same columns, BBRef player ids)

ESPN athletes are mapped to Basketball-Reference ids by normalised name (team
used to break ties); unmapped players (e.g. rookies) keep an "espn_<id>" id, so
their history is just their ESPN games. The mapping is stored in
nba_analytics.dim_player_id_map.

Only regular-season games are used (play-offs/play-in = team game > 82).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date
from pathlib import Path

import db_dtypes  # noqa: F401  (BigQuery DATE -> pandas)
import pandas as pd

from engine import espn_history

log = logging.getLogger("data")

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT = "nba-analytics-503718"
DATASET = f"{PROJECT}.nba_analytics"
HIST_CACHE = Path("data/cache/hist_player_games.parquet")

CURRENT_SEASON = "2026-27"
SEASON_STARTS = {"2026-27": date(2026, 10, 20)}

COLS = ["game_id", "game_date", "season", "team", "opp", "is_home", "player_id", "player_name",
        "minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "trb",
        "ast", "stl", "blk", "tov", "pf", "plus_minus"]


def bq_client():
    """Local: service-account key file. Cloud Run: the job's attached service account (ADC)."""
    from google.cloud import bigquery
    if Path(KEY_PATH).exists():
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(KEY_PATH)
        return bigquery.Client(credentials=creds, project=PROJECT)
    return bigquery.Client(project=PROJECT)


def season_for(d: date) -> str:
    y = d.year if d.month >= 9 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def norm_name(s: str) -> str:
    """Accent/suffix-free lowercase key; also repairs double-encoded UTF-8 (DonÄiÄ)."""
    s = str(s)
    for enc in ("cp1252", "latin-1"):
        try:
            s = s.encode(enc).decode("utf-8")
            break
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace(".", "").replace("'", "").replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


# ---------------------------------------------------------------
# Completed seasons (BigQuery)
# ---------------------------------------------------------------
HIST_SQL = f"""
SELECT game_id, game_date, season, team_id AS team, opponent_id AS opp, is_home,
       player_id, player_name, minutes, pts, fg AS fgm, fga, fg3 AS fg3m, fg3a,
       ft AS ftm, fta, orb, drb, trb, ast, stl, blk, tov, pf, plus_minus
FROM `{DATASET}.fact_player_boxscores`
WHERE minutes > 0
"""


def load_history(refresh: bool = False) -> pd.DataFrame:
    """Completed seasons from BigQuery (cached locally - those seasons never change)."""
    if HIST_CACHE.exists() and not refresh:
        return pd.read_parquet(HIST_CACHE)
    log.info("Reading fact_player_boxscores from BigQuery ...")
    df = bq_client().query(HIST_SQL).to_dataframe()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_home"] = df["is_home"].astype(bool)
    for c in COLS[8:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    df = regular_season_only(df)
    HIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_parquet(HIST_CACHE, index=False)
    log.info("History: %d player-games, seasons %s", len(df), sorted(df["season"].unique()))
    return df[COLS]


def regular_season_only(df: pd.DataFrame) -> pd.DataFrame:
    """BigQuery labels every game 'Regular Season'; team game no. > 82 = play-in/play-offs."""
    tg = df[["game_id", "season", "team", "game_date"]].drop_duplicates()
    tg = tg.sort_values(["team", "game_date", "game_id"])
    tg["n"] = tg.groupby(["team", "season"]).cumcount() + 1
    post = set(tg.loc[tg["n"] > 82, "game_id"])
    return df[~df["game_id"].isin(post)].reset_index(drop=True)


# ---------------------------------------------------------------
# ESPN -> BBRef id mapping
# ---------------------------------------------------------------
def build_name_index(hist: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """norm name -> [(player_id, latest team)], most recent first."""
    last = hist.sort_values("game_date").groupby("player_id").agg(
        name=("player_name", "last"), team=("team", "last"), d=("game_date", "max")).reset_index()
    idx: dict[str, list[tuple[str, str]]] = {}
    for r in last.sort_values("d", ascending=False).itertuples():
        idx.setdefault(norm_name(r.name), []).append((r.player_id, r.team))
    return idx


def map_espn(name: str, team: str, idx: dict) -> str | None:
    c = idx.get(norm_name(name))
    if not c:
        return None
    return next((pid for pid, t in c if t == team), c[0][0])


def load_current(until: date, hist: pd.DataFrame, push_bq: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Current season from ESPN (games before `until`), mapped to BBRef ids."""
    G, P = espn_history.load_season(CURRENT_SEASON, SEASON_STARTS[CURRENT_SEASON], until)
    idx = build_name_index(hist)
    id_map: dict[str, str] = {}
    if not P.empty:
        for r in P.drop_duplicates("espn_id").itertuples():
            id_map[r.espn_id] = map_espn(r.player_name, r.team, idx) or f"espn_{r.espn_id}"
        P["player_id"] = P["espn_id"].map(id_map)
    if push_bq and not P.empty:
        try:
            push_current(G, P, id_map)
        except Exception as exc:
            log.warning("BigQuery push of current season failed (%s) - continuing.", exc)
    return G, P, id_map


def push_current(G: pd.DataFrame, P: pd.DataFrame, id_map: dict) -> None:
    from google.cloud import bigquery
    c = bq_client()
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    g = G.copy(); g["game_date"] = g["game_date"].dt.date
    p = P.copy(); p["game_date"] = p["game_date"].dt.date
    c.load_table_from_dataframe(g, f"{DATASET}.espn_game_results", job_config=cfg).result()
    c.load_table_from_dataframe(p, f"{DATASET}.espn_player_boxscores", job_config=cfg).result()
    m = pd.DataFrame({"espn_id": list(id_map), "player_id": list(id_map.values())})
    c.load_table_from_dataframe(m, f"{DATASET}.dim_player_id_map", job_config=cfg).result()
    cols = ", ".join(COLS)
    c.query(f"""
        CREATE OR REPLACE VIEW `{DATASET}.v_player_games` AS
        SELECT {cols}, 'bbref' AS source FROM (
          SELECT game_id, game_date, season, team_id AS team, opponent_id AS opp, is_home,
                 player_id, player_name, minutes, CAST(pts AS FLOAT64) pts, CAST(fg AS FLOAT64) fgm,
                 CAST(fga AS FLOAT64) fga, CAST(fg3 AS FLOAT64) fg3m, CAST(fg3a AS FLOAT64) fg3a,
                 CAST(ft AS FLOAT64) ftm, CAST(fta AS FLOAT64) fta, CAST(orb AS FLOAT64) orb,
                 CAST(drb AS FLOAT64) drb, CAST(trb AS FLOAT64) trb, CAST(ast AS FLOAT64) ast,
                 CAST(stl AS FLOAT64) stl, CAST(blk AS FLOAT64) blk, CAST(tov AS FLOAT64) tov,
                 CAST(pf AS FLOAT64) pf, CAST(plus_minus AS FLOAT64) plus_minus
          FROM `{DATASET}.fact_player_boxscores` WHERE minutes > 0)
        UNION ALL
        SELECT {cols}, 'espn' AS source FROM `{DATASET}.espn_player_boxscores`
    """).result()
    log.info("BigQuery: espn_game_results / espn_player_boxscores / dim_player_id_map / v_player_games refreshed")


def player_games(until: date, refresh_hist: bool = False, push_bq: bool = True):
    """Unified regular-season player-games before `until` + ESPN games table + id map."""
    hist = load_history(refresh_hist)
    G, P, id_map = load_current(until, hist, push_bq)
    pg = pd.concat([hist, P[COLS]] if not P.empty else [hist], ignore_index=True)
    pg = pg[pg["game_date"] < pd.Timestamp(until)]
    return pg.sort_values(["game_date", "game_id"]).reset_index(drop=True), G, id_map
