"""
api/routers/games.py
=====================
Maç tahmini endpoint'leri.

GET /api/games?date=YYYYMMDD   → Maç kartları listesi
GET /api/game/{event_id}        → Maç detayı (props + injuries)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.models import GameCard, GameDetail, GamesResponse, PlayerProp, InjuryEntry
from engine import espn_client as espn

log = logging.getLogger(__name__)
router = APIRouter()

OUTPUT_DIR = Path("output")


def _load_predictions(date_str: str) -> dict:
    """Verilen tarihe ait JSON tahmin dosyasını yükle."""
    path = OUTPUT_DIR / f"predictions_{date_str}.json"
    if not path.exists():
        # Bugün için latest dene
        latest = OUTPUT_DIR / "predictions_latest.json"
        if latest.exists():
            data = json.loads(latest.read_text(encoding="utf-8"))
            if data.get("date") == date_str:
                return data
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} için tahmin bulunamadı. Önce 'python -m engine.predict_daily --date {date_str}' çalıştırın.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _to_game_card(p: dict) -> GameCard:
    return GameCard(
        event_id=p.get("event_id", ""),
        home_team_id=p.get("home_team_id", 0),
        away_team_id=p.get("away_team_id", 0),
        home_team_name=p.get("home_team_name", ""),
        away_team_name=p.get("away_team_name", ""),
        home_team_abbr=p.get("home_team_abbr", ""),
        away_team_abbr=p.get("away_team_abbr", ""),
        home_team_logo=p.get("home_team_logo", ""),
        away_team_logo=p.get("away_team_logo", ""),
        game_date=p.get("game_date", ""),
        status=p.get("status", ""),
        pred_home_score=p.get("pred_home_score", 0.0),
        pred_away_score=p.get("pred_away_score", 0.0),
        pred_spread=p.get("pred_spread", 0.0),
        pred_total=p.get("pred_total", 0.0),
        home_win_prob=p.get("home_win_prob", 0.5),
        pred_game_pace=p.get("pred_game_pace", 0.0),
    )


def _to_player_prop(raw: dict) -> PlayerProp:
    return PlayerProp(
        player_id=raw.get("player_id", 0),
        name=raw.get("name", ""),
        position=raw.get("position", ""),
        proj_minutes=raw.get("proj_minutes", 0.0),
        proj_pts=raw.get("proj_pts", 0.0),
        proj_trb=raw.get("proj_trb", 0.0),
        proj_ast=raw.get("proj_ast", 0.0),
        proj_fg3=raw.get("proj_fg3", 0.0),
        proj_stl=raw.get("proj_stl", 0.0),
        proj_blk=raw.get("proj_blk", 0.0),
    )


@router.get("/games", response_model=GamesResponse)
def get_games(
    date: str = Query(
        default=datetime.now().strftime("%Y%m%d"),
        description="Tarih YYYYMMDD formatında",
    )
):
    """Belirtilen tarihe ait tüm maç kartlarını döndür."""
    data = _load_predictions(date)
    cards = [_to_game_card(p) for p in data.get("predictions", [])]
    return GamesResponse(date=date, count=len(cards), games=cards)


@router.get("/game/{event_id}", response_model=GameDetail)
def get_game_detail(event_id: str):
    """
    Tek bir maçın detayını döndür.
    Gerçek zamanlı sakatlık bilgisini de ESPN'den çeker.
    """
    # En son tahmin dosyasını tara
    latest = OUTPUT_DIR / "predictions_latest.json"
    if not latest.exists():
        raise HTTPException(status_code=404, detail="Henüz tahmin üretilmemiş.")

    data = json.loads(latest.read_text(encoding="utf-8"))
    pred = next(
        (p for p in data.get("predictions", []) if p.get("event_id") == event_id),
        None,
    )
    if not pred:
        raise HTTPException(status_code=404, detail=f"Maç bulunamadı: {event_id}")

    card = _to_game_card(pred)
    props = pred.get("player_props", {})

    home_props = [_to_player_prop(p) for p in props.get("home", [])]
    away_props = [_to_player_prop(p) for p in props.get("away", [])]

    # Gerçek zamanlı sakatlık listesini çek
    injuries = espn.get_injuries()
    home_inj = [
        InjuryEntry(**i) for i in injuries.get(pred["home_team_id"], [])
    ]
    away_inj = [
        InjuryEntry(**i) for i in injuries.get(pred["away_team_id"], [])
    ]

    return GameDetail(
        **card.model_dump(),
        home_player_props=home_props,
        away_player_props=away_props,
        home_injuries=home_inj,
        away_injuries=away_inj,
    )
