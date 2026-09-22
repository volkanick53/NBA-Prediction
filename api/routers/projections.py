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

TEAM_PARQUET = Path("data/team_features.parquet")
OUTPUT_DIR   = Path("output")


@router.get("/projections/season", response_model=SeasonProjectionsResponse)
def get_season_projections():
    """
    Tüm takımların bu sezon için tahmini galibiyet/mağlubiyet oranlarını döndür.
    Tahminler, team_features.parquet içindeki son N maçlık rolling ortalamalara
    ve birikmiş predictions klasöründeki sonuçlara dayanır.
    """
    if not TEAM_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail="team_features.parquet bulunamadı. Lütfen ETL pipeline'ı çalıştırın.",
        )

    df = pd.read_parquet(TEAM_PARQUET)

    # Her takımın en güncel metriklerini al
    latest = df.sort_values("game_date").groupby("team_id").last().reset_index()
    season = str(df["season"].max())

    # Birikmiş tahmin JSON'larından takım bazlı ortalama puan hesapla
    pred_scores: dict[int, list[float]] = {}
    for fpath in sorted(OUTPUT_DIR.glob("predictions_20*.json")):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            for p in data.get("predictions", []):
                home_id = p.get("home_team_id", 0)
                away_id = p.get("away_team_id", 0)
                if home_id:
                    pred_scores.setdefault(home_id, []).append(p.get("pred_home_score", 0))
                if away_id:
                    pred_scores.setdefault(away_id, []).append(p.get("pred_away_score", 0))
        except Exception:
            continue

    projections = []
    for _, row in latest.iterrows():
        team_id = int(row["team_id"])

        # Basit sezon win-rate tahmini: rolling ORtg / rolling DRtg oranına göre
        ortg = float(row.get("roll_ortg_10", 110) or 110)
        drtg = float(row.get("roll_drtg_10", 110) or 110)
        net_rating = ortg - drtg

        # Pythagorean win expectation (k=14 NBA standardı)
        k = 14.0
        win_pct = ortg ** k / (ortg ** k + drtg ** k)
        proj_wins   = round(win_pct * 82, 1)
        proj_losses = round((1 - win_pct) * 82, 1)

        avg_score = (
            round(sum(pred_scores[team_id]) / len(pred_scores[team_id]), 1)
            if team_id in pred_scores
            else round(float(row.get("roll_pts_10", 110) or 110), 1)
        )

        projections.append(TeamSeasonProjection(
            team_id=team_id,
            team_name=str(row.get("team_id", team_id)),  # isim için BQ join gerekirdi
            team_abbr="",
            proj_wins=proj_wins,
            proj_losses=proj_losses,
            proj_win_pct=round(win_pct, 4),
            avg_pred_score=avg_score,
        ))

    # Win % göre sırala
    projections.sort(key=lambda x: x.proj_win_pct, reverse=True)

    return SeasonProjectionsResponse(season=season, projections=projections)
