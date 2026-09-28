"""
api/routers/games.py
=====================
Mac tahmini endpoint'leri.

GET /api/games?date=YYYYMMDD   -> Mac kartlari listesi
GET /api/game/{event_id}        -> Mac detayi (props + injuries)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from api.models import GameCard, GameDetail, GamesResponse, PlayerProp, InjuryEntry

log = logging.getLogger(__name__)
router = APIRouter()

OUTPUT_DIR = Path("output")


def _load_predictions(date_str: str) -> dict:
    """Verilen tarihe ait JSON tahmin dosyasini yukle."""
    path = OUTPUT_DIR / f"predictions_{date_str}.json"
    if not path.exists():
        latest = OUTPUT_DIR / "predictions_latest.json"
        if latest.exists():
            data = json.loads(latest.read_text(encoding="utf-8"))
            if data.get("date") == date_str:
                return data
        raise HTTPException(
            status_code=404,
            detail=f"{date_str} icin tahmin bulunamadi. Once 'python predict_oct20_offline.py' calistirin.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _to_game_card(g: dict) -> GameCard:
    """Yeni JSON formatindaki game dict'ini GameCard modeline donustur."""
    # home_win_prob: yeni formatta 0-100 araliginda geliyor, GameCard 0-1 bekliyor
    hwp = g.get("home_win_prob", 50.0)
    if hwp > 1.0:
        hwp = hwp / 100.0

    return GameCard(
        event_id   = g.get("game_id", ""),
        home_team_id   = 0,
        away_team_id   = 0,
        home_team_name = g.get("home_team_full", g.get("home_team", "")),
        away_team_name = g.get("away_team_full", g.get("away_team", "")),
        home_team_abbr = g.get("home_team", ""),
        away_team_abbr = g.get("away_team", ""),
        home_team_logo = g.get("home_logo", ""),
        away_team_logo = g.get("away_logo", ""),
        game_date      = g.get("match_time_tr", g.get("game_date", "")),
        status         = "SCHEDULED",
        pred_home_score = g.get("proj_home_score", 0.0),
        pred_away_score = g.get("proj_away_score", 0.0),
        pred_spread     = g.get("proj_spread", 0.0),
        pred_total      = g.get("proj_total", 0.0),
        home_win_prob   = hwp,
        pred_game_pace  = g.get("pred_game_pace", 98.0),
    )


def _to_player_prop(raw: dict) -> PlayerProp:
    return PlayerProp(
        player_id    = raw.get("player_id", 0),
        name         = raw.get("player_name", raw.get("name", "")),
        position     = raw.get("position", ""),
        proj_minutes = raw.get("projected_mins", raw.get("proj_minutes", 0.0)),
        proj_pts     = raw.get("proj_pts", 0.0),
        proj_trb     = raw.get("proj_trb", 0.0),
        proj_ast     = raw.get("proj_ast", 0.0),
        proj_fg3     = raw.get("proj_fg3", 0.0),
        proj_stl     = raw.get("proj_stl", 0.0),
        proj_blk     = raw.get("proj_blk", 0.0),
    )


@router.get("/games", response_model=GamesResponse)
def get_games(
    date: str = Query(
        default=datetime.now().strftime("%Y%m%d"),
        description="Tarih YYYYMMDD formatinda",
    )
):
    """Belirtilen tarihe ait tum mac kartlarini dondur."""
    data = _load_predictions(date)
    # Yeni format: top-level 'games' listesi
    games_list = data.get("games", [])
    cards = [_to_game_card(g) for g in games_list]
    return GamesResponse(date=date, count=len(cards), games=cards)


@router.get("/game/{event_id}", response_model=GameDetail)
def get_game_detail(event_id: str):
    """
    Tek bir macin detayini dondur.
    Oyuncu prop'lari flat player_props listesinden game_id ile filtrelenir.
    """
    # En son tahmin dosyasini bul
    latest = OUTPUT_DIR / "predictions_latest.json"
    # Tarihe gore dene
    pred_file: Path | None = None
    if latest.exists():
        pred_file = latest
    else:
        # output/ altindaki ilk dosyayi dene
        files = sorted(OUTPUT_DIR.glob("predictions_2*.json"), reverse=True)
        if files:
            pred_file = files[0]

    if not pred_file:
        raise HTTPException(status_code=404, detail="Henuz tahmin uretilmemis.")

    data = json.loads(pred_file.read_text(encoding="utf-8"))

    # game_id ile eslesen mac satirini bul
    game_row = next(
        (g for g in data.get("games", []) if g.get("game_id") == event_id),
        None,
    )
    if not game_row:
        raise HTTPException(status_code=404, detail=f"Mac bulunamadi: {event_id}")

    card = _to_game_card(game_row)

    # Flat player_props listesinden bu maca ait oyunculari filtrele
    all_props = data.get("player_props", [])
    home_abbr = game_row.get("home_team", "").upper()
    away_abbr = game_row.get("away_team", "").upper()

    home_props = [
        _to_player_prop(p) for p in all_props
        if p.get("game_id") == event_id and p.get("team_id", "").upper() == home_abbr
    ]
    away_props = [
        _to_player_prop(p) for p in all_props
        if p.get("game_id") == event_id and p.get("team_id", "").upper() == away_abbr
    ]

    return GameDetail(
        **card.model_dump(),
        home_player_props = home_props,
        away_player_props = away_props,
        home_injuries     = [],
        away_injuries     = [],
    )
