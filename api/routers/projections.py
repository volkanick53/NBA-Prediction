"""
api/routers/projections.py
===========================
Sezon projeksiyon endpoint'leri.

GET /api/projections/season  → Tüm takımların sezon win/loss projeksiyonları
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.models import SeasonProjectionsResponse, TeamSeasonProjection

log = logging.getLogger(__name__)
router = APIRouter()

OUTPUT_DIR   = Path("output")
TEAM_PROJ_CSV = Path("proj_team_wins_2026_27.csv")
STANDINGS_CSV = Path("nba_2026_27_team_standings_projection.csv")

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


@router.get("/projections/season", response_model=SeasonProjectionsResponse)
def get_season_projections():
    """
    2026-27 sezon takim projeksiyonlari (kadro bazli net rating + Pythagorean).
    Kaynak: proj_team_wins_2026_27.csv (+ conference icin standings projeksiyonu).
    """
    if not TEAM_PROJ_CSV.exists():
        raise HTTPException(status_code=503, detail=f"{TEAM_PROJ_CSV} bulunamadi.")

    df = pd.read_csv(TEAM_PROJ_CSV)
    conf: dict[str, str] = {}
    if STANDINGS_CSV.exists():
        st = pd.read_csv(STANDINGS_CSV)
        conf = dict(zip(st["team_id"], st["conference"]))

    projections = []
    for _, r in df.iterrows():
        tid = str(r["team_id"])
        wins = float(r["exp_wins"])
        projections.append(TeamSeasonProjection(
            team_id=tid,
            team_name=TEAM_FULL_NAMES.get(tid, tid),
            team_abbr=tid,
            proj_wins=round(wins, 1),
            proj_losses=round(float(r["exp_losses"]), 1),
            proj_win_pct=round(wins / 82.0, 4),
            avg_pred_score=round(float(r["net_rating_proj"]), 2),  # = net rating proj
            conference=conf.get(tid),
            playoff_prob_pct=float(r["playoff_prob_pct"]),
            vegas_total=float(r["vegas_total"]) if pd.notna(r.get("vegas_total")) else None,
        ))

    projections.sort(key=lambda x: x.proj_win_pct, reverse=True)
    return SeasonProjectionsResponse(season=str(df["season"].iloc[0]), projections=projections)
