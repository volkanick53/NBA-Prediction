"""
engine/espn_history.py
======================
Current-season game & player box scores from the public ESPN Site API
(scoreboard + summary). The BigQuery warehouse holds completed seasons
(Basketball-Reference); this module supplies the season in progress with the
same box-score columns.

Finished games never change, so every summary is cached on disk
(data/espn_cache/) and fetched only once.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from engine import espn_client as espn

log = logging.getLogger("espn_history")

CACHE = Path("data/espn_cache")
TO_BBREF = {"CHA": "CHO", "PHX": "PHO", "BKN": "BRK", "NY": "NYK", "SA": "SAS",
            "NO": "NOP", "UTAH": "UTA", "GS": "GSW", "WSH": "WAS"}


def bbref(a: str) -> str:
    a = str(a).upper().strip()
    return TO_BBREF.get(a, a)


def _num(s, default=0.0) -> float:
    try:
        return float(str(s).replace("+", ""))
    except (TypeError, ValueError):
        return default


def _made_att(s) -> tuple[float, float]:
    try:
        m, a = str(s).split("-")
        return float(m), float(a)
    except ValueError:
        return 0.0, 0.0


def _summary(event_id: str) -> dict:
    """Trimmed, cached ESPN summary. Only called for FINAL games."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"summary_{event_id}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    s = espn._get(f"{espn.BASE}/summary", params={"event": event_id})
    keep = {"boxscore": s.get("boxscore", {}), "header": s.get("header", {})}
    f.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    return keep


def scoreboard(d: date, today: date) -> list[dict]:
    """Scoreboard for a date; cached once the date is past and every game is final."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = d.strftime("%Y%m%d")
    f = CACHE / f"scoreboard_{ds}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    games = espn.get_scoreboard(ds)
    if d < today and all(g["status"] == "STATUS_FINAL" for g in games):
        f.write_text(json.dumps(games), encoding="utf-8")
    return games


def _parse_game(ev: dict, d: date, season: str) -> tuple[dict | None, list[dict]]:
    s = _summary(ev["event_id"])
    comp = s.get("header", {}).get("competitions", [{}])[0]
    side_of, score = {}, {}
    for c in comp.get("competitors", []):
        side_of[bbref(c["team"]["abbreviation"])] = c.get("homeAway")
        score[c.get("homeAway")] = _num(c.get("score"))
    if set(side_of.values()) != {"home", "away"}:
        return None, []
    team_of = {v: k for k, v in side_of.items()}

    players = []
    for blk in s.get("boxscore", {}).get("players", []):
        team = bbref(blk["team"]["abbreviation"])
        side = side_of.get(team)
        if side is None:
            continue
        st = blk["statistics"][0]
        keys = st.get("keys", [])
        for a in st.get("athletes", []):
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            v = dict(zip(keys, a["stats"]))
            mins = _num(v.get("minutes"))
            if mins <= 0:
                continue
            fgm, fga = _made_att(v.get("fieldGoalsMade-fieldGoalsAttempted"))
            f3m, f3a = _made_att(v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = _made_att(v.get("freeThrowsMade-freeThrowsAttempted"))
            players.append({
                "game_id": ev["event_id"], "game_date": pd.Timestamp(d), "season": season,
                "team": team, "opp": team_of["away" if side == "home" else "home"],
                "is_home": side == "home",
                "espn_id": str(a["athlete"]["id"]), "player_name": a["athlete"]["displayName"],
                "starter": bool(a.get("starter")), "minutes": mins,
                "pts": _num(v.get("points")), "fgm": fgm, "fga": fga, "fg3m": f3m, "fg3a": f3a,
                "ftm": ftm, "fta": fta, "orb": _num(v.get("offensiveRebounds")),
                "drb": _num(v.get("defensiveRebounds")), "trb": _num(v.get("rebounds")),
                "ast": _num(v.get("assists")), "stl": _num(v.get("steals")),
                "blk": _num(v.get("blocks")), "tov": _num(v.get("turnovers")),
                "pf": _num(v.get("fouls")), "plus_minus": _num(v.get("plusMinus")),
            })
    game = {"game_id": ev["event_id"], "game_date": pd.Timestamp(d), "season": season,
            "home": team_of["home"], "away": team_of["away"],
            "home_pts": score["home"], "away_pts": score["away"]}
    return game, players


def load_season(season: str, season_start: date, until: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All FINAL games with game_date in [season_start, until)."""
    games, players = [], []
    d = season_start
    while d < until:
        for ev in scoreboard(d, until):
            if ev["status"] != "STATUS_FINAL":
                continue
            try:
                g, pl = _parse_game(ev, d, season)
            except Exception as exc:   # one bad payload must not kill the run
                log.warning("Skipping event %s (%s)", ev["event_id"], exc)
                continue
            if g:
                games.append(g)
                players.extend(pl)
        d += timedelta(days=1)
    log.info("ESPN %s: %d final games, %d player lines (before %s)",
             season, len(games), len(players), until)
    return pd.DataFrame(games), pd.DataFrame(players)
