"""
engine/espn_client.py
======================
ESPN Site API istemcisi.
Tüm HTTP çağrıları bu modülden yapılır; HTML scraping yoktur.

Kullanılan endpoint'ler (basketball.md referans):
  - scoreboard   : günün maçları + ev/deplasman takım ID'leri
  - injuries     : lig geneli sakatlık listesi
  - roster       : takım kadrosu (aktif oyuncular)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
HEADERS = {"User-Agent": "NBA-Prediction-Engine/1.0"}
TIMEOUT = 20.0
RETRY_DELAY = 2.0


def _get(url: str, params: dict | None = None, retries: int = 3) -> dict[str, Any]:
    """Basit GET isteği; hata durumunda retry uygular."""
    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(headers=HEADERS, timeout=TIMEOUT) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.warning("Deneme %d/%d başarısız: %s — %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"ESPN API isteği başarısız oldu: {url}")


# ---------------------------------------------------------------
# Scoreboard — günün maçları
# ---------------------------------------------------------------
def get_scoreboard(date_str: str) -> list[dict]:
    """
    YYYYMMDD formatında tarih al, o güne ait maç listesini döndür.

    Dönen liste elemanları:
        {
          "event_id": str,
          "home_team_id": int,
          "away_team_id": int,
          "home_team_name": str,
          "away_team_name": str,
          "home_team_abbr": str,
          "away_team_abbr": str,
          "home_team_logo": str,
          "away_team_logo": str,
          "game_date": str,      # ISO datetime
          "status": str,
        }
    """
    data = _get(f"{BASE}/scoreboard", params={"dates": date_str})
    games = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        home_team = home.get("team", {})
        away_team = away.get("team", {})

        games.append({
            "event_id":        event.get("id", ""),
            "home_team_id":    int(home_team.get("id", 0)),
            "away_team_id":    int(away_team.get("id", 0)),
            "home_team_name":  home_team.get("displayName", ""),
            "away_team_name":  away_team.get("displayName", ""),
            "home_team_abbr":  home_team.get("abbreviation", ""),
            "away_team_abbr":  away_team.get("abbreviation", ""),
            "home_team_logo":  home_team.get("logo", ""),
            "away_team_logo":  away_team.get("logo", ""),
            "home_score":      int(home.get("score", 0) or 0),
            "away_score":      int(away.get("score", 0) or 0),
            "game_date":       event.get("date", ""),
            "status":          event.get("status", {}).get("type", {}).get("name", ""),
        })
    log.info("Scoreboard: %d maç bulundu (%s)", len(games), date_str)
    return games


# ---------------------------------------------------------------
# Injuries — lig geneli sakatlık listesi
# ---------------------------------------------------------------

# Oyuncu tahminlerinden dışlanacak statüler
EXCLUDED_STATUSES = {"Out", "Doubtful", "Injured Reserve", "Suspended", "Out For Season"}

def get_injuries() -> dict[int, list[dict]]:
    """
    Lig geneli güncel sakatlık raporunu çek.

    Dönen dict: { team_id: [ {player_id, name, status, description}, ... ] }
    """
    data = _get(f"{BASE}/injuries")
    injured: dict[int, list[dict]] = {}

    for team_entry in data.get("injuries", []):
        team_id = int(team_entry.get("team", {}).get("id", 0))
        team_injuries = []
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            status  = inj.get("status", "")
            team_injuries.append({
                "player_id":   int(athlete.get("id", 0)),
                "name":        athlete.get("displayName", ""),
                "position":    athlete.get("position", {}).get("abbreviation", ""),
                "status":      status,
                "description": inj.get("longComment", ""),
                "is_excluded": status in EXCLUDED_STATUSES,
            })
        if team_injuries:
            injured[team_id] = team_injuries

    log.info("Sakatlık raporu: %d takım", len(injured))
    return injured


# ---------------------------------------------------------------
# Roster — takım kadrosu
# ---------------------------------------------------------------
def get_roster(team_id: int) -> list[dict]:
    """
    Takımın güncel kadrosunu çek.

    Dönen liste: [ {player_id, name, position, jersey, status}, ... ]
    """
    data = _get(f"{BASE}/teams/{team_id}/roster")
    players = []
    for group in data.get("athletes", []):
        for athlete in group.get("items", []):
            players.append({
                "player_id": int(athlete.get("id", 0)),
                "name":      athlete.get("displayName", ""),
                "position":  athlete.get("position", {}).get("abbreviation", ""),
                "jersey":    athlete.get("jersey", ""),
                "status":    athlete.get("status", {}).get("type", {}).get("name", "Active"),
            })
    log.info("Roster (team %d): %d oyuncu", team_id, len(players))
    return players


# ---------------------------------------------------------------
# Active Lineup — Sakatlıklar çıkarılmış kadro
# ---------------------------------------------------------------
def get_active_lineup(team_id: int, injuries: dict[int, list[dict]]) -> list[dict]:
    """
    Rosto al, sakatlık listesindeki dışlanacak oyuncuları çıkar.
    """
    roster   = get_roster(team_id)
    excluded = {
        p["player_id"]
        for p in injuries.get(team_id, [])
        if p.get("is_excluded")
    }
    active = [p for p in roster if p["player_id"] not in excluded]
    log.info(
        "Aktif kadro (team %d): %d/%d oyuncu (%d dışlandı)",
        team_id, len(active), len(roster), len(excluded),
    )
    return active
