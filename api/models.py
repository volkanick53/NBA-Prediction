"""
api/models.py
==============
FastAPI Pydantic response şemaları.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class PlayerProp(BaseModel):
    player_id:    int | str
    name:         str
    position:     str
    proj_minutes: float
    proj_pts:     float
    proj_trb:     float
    proj_ast:     float
    proj_fg3:     float
    proj_stl:     float
    proj_blk:     float
    injury_status: Optional[str] = None      # e.g. Day-To-Day; None = healthy
    sd_pts: Optional[float] = None           # residual std from 2025-26 holdout
    sd_trb: Optional[float] = None
    sd_ast: Optional[float] = None
    sd_fg3: Optional[float] = None
    sd_stl: Optional[float] = None
    sd_blk: Optional[float] = None
    low_confidence_stats: list[str] = []


class InjuryEntry(BaseModel):
    player_id:   int | str
    name:        str
    position:    str
    status:      str
    description: str
    is_excluded: bool


class GameCard(BaseModel):
    event_id:        str
    home_team_id:    int
    away_team_id:    int
    home_team_name:  str
    away_team_name:  str
    home_team_abbr:  str
    away_team_abbr:  str
    home_team_logo:  str
    away_team_logo:  str
    game_date:       str
    status:          str
    pred_home_score: float
    pred_away_score: float
    pred_spread:     float
    pred_total:      float
    home_win_prob:   float
    pred_game_pace:  float
    home_injury_penalty: float = 0.0   # net-rating points removed for absences
    away_injury_penalty: float = 0.0


class GameDetail(GameCard):
    home_player_props: list[PlayerProp]
    away_player_props: list[PlayerProp]
    home_injuries:     list[InjuryEntry]
    away_injuries:     list[InjuryEntry]


class GamesResponse(BaseModel):
    date:   str
    count:  int
    games:  list[GameCard]


class TeamSeasonProjection(BaseModel):
    team_id:        str
    team_name:      str
    team_abbr:      str
    proj_wins:      float
    proj_losses:    float
    proj_win_pct:   float
    avg_pred_score: float          # projected net rating
    conference:     Optional[str] = None
    playoff_prob_pct: Optional[float] = None
    vegas_total:      Optional[float] = None


class SeasonProjectionsResponse(BaseModel):
    season:      str
    projections: list[TeamSeasonProjection]
