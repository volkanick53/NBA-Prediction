"""
predict_oct20_offline.py
========================
NBA maclari icin ESPN API'ye bagimli olmadan, yerel dim_* / proj_* CSV
dosyalari ve egitilmis XGBoost modelleri ile offline gunluk tahmin uretir.

GERCEKCI SEZON BASI SENARYOSU
-----------------------------
Sezonun ilk maclarinda takim/oyuncu icin "bu sezon" rolling verisi yoktur.
Gecen sezonun son 10 maci (playoff + eski kadro) kullanmak yaniltici olur.
Bu nedenle endustri standardi uygulanir:

  1) PRIOR (preseason) feature'lar:
       - Oyuncu : MARCELS projeksiyonu (proj_player_stats_2026_27.csv)
       - Takim  : Kadro bazli net rating + pace (proj_team_wins_2026_27.csv)
                  ORtg/DRtg ayrimi gecen sezon profiline gore yapilir.
  2) Bu prior'lar, egitimdeki rolling feature isimleriyle XGBoost'a beslenir.
  3) Sezon ilerledikce in-season rolling verisi ile Bayesian blend:
         w_in = n / (n + K)    (K: takim=10 mac, oyuncu=8 mac)
     Offline modda in-season veri olmadigi icin w_in = 0 (saf prior).
  4) Rest days fikstürden hesaplanir (acilis gecesi = 7 cap, B2B = 1).

Stage 1 (Takim): Ratings modeli (ORtg x DRtg / lig ort.) + HCA,
                 XGBoost team modeli ile %20 blend (R2 ~0.03 oldugu icin).
Stage 2 (Oyuncu): Rotasyon (proj_mpg sirali, 8-10 kisi), 240 dk kisiti,
                  XGBoost prop modelleri, takim skoruna tutarlilik kalibrasyonu.

Kullanim:
    python predict_oct20_offline.py                       # 2026-10-20
    python predict_oct20_offline.py --date 20261021
    python predict_oct20_offline.py --date 20261020 --days 7   # acilis haftasi
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("predict_offline")

# ---------------------------------------------------------------
# Sabitler / Dosya Yollari
# ---------------------------------------------------------------
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")

ROSTER_FILE          = Path("dim_current_rosters_2026_27.csv")
PLAYER_PROFILES_FILE = Path("dim_player_season_profiles.csv")
TEAM_PROFILES_FILE   = Path("dim_team_season_profiles.csv")
SCHEDULE_FILE        = Path("dim_schedule_2026_27.csv")
PLAYER_PROJ_FILE     = Path("proj_player_stats_2026_27.csv")
TEAM_PROJ_FILE       = Path("proj_team_wins_2026_27.csv")

SEASON_START = "2026-10-20"

# Belirsizlik: sezon basinda (kadro yeni, veri yok) daha yuksek
SCORE_STD_OPENER = 13.0
SCORE_STD_BASE   = 12.0

# Stage 1 blend agirligi (XGBoost team points modeli)
XGB_TEAM_WEIGHT = 0.20

# Stage 2 blend: XGBoost noisy rolling girdilerle egitildi; MARCELS girdisi zaten
# regress edilmis oldugundan saf XGB yildizlari cift shrink eder -> %50/%50 blend
XGB_PLAYER_WEIGHT = 0.50

# Bayesian prior gucu (mac sayisi cinsinden)
K_TEAM_PRIOR   = 10
K_PLAYER_PRIOR = 8

# Rotasyon kurallari
TOTAL_GAME_MINUTES = 240.0
ROTATION_MAX       = 10
ROTATION_MIN       = 8
ROTATION_MIN_MPG   = 12.0
MAX_PLAYER_MINUTES = 38.0
REST_DAYS_CAP      = 7

LOGO_TPL = "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr}.png"
ET_TO_TR_OFFSET_HOURS = 7   # Ekim sonu: EDT(UTC-4) -> TR(UTC+3)

# Schedule/BBRef kisaltmasi -> ESPN kisaltmasi
TEAM_ABBR_MAP = {
    "NYK": "NY", "SAS": "SA", "NOP": "NO", "UTA": "UTAH",
    "BRK": "BKN", "CHO": "CHA", "GSW": "GS", "PHO": "PHX",
    "WAS": "WSH",
}
# Bazi dosyalar CHA/PHX/BKN kullanabilir -> BBRef'e cevir
TO_BBREF = {
    "CHA": "CHO", "PHX": "PHO", "BKN": "BRK", "NY": "NYK", "SA": "SAS",
    "NO": "NOP", "UTAH": "UTA", "GS": "GSW", "WSH": "WAS",
}

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


def bbref(abbr: str) -> str:
    a = str(abbr).upper().strip()
    return TO_BBREF.get(a, a)


def espn_abbr(abbr: str) -> str:
    return TEAM_ABBR_MAP.get(bbref(abbr), bbref(abbr))


def fix_name(s: str) -> str:
    """Cift encode edilmis UTF-8 isimleri duzeltir (DonÄiÄ -> Dončić)."""
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------
# Model Yukleyici
# ---------------------------------------------------------------
class ModelRegistry:
    """Egitilmis XGBoost modelleri ve meta bilgilerini yukler."""

    def __init__(self):
        log.info("Modeller yukleniyor...")
        self.team_meta   = self._load_json("team_model_meta.json")
        self.player_meta = self._load_json("player_model_meta.json")

        self.pace_model        = self._load_xgb("team_pace_model.json")
        self.points_home_model = self._load_xgb("team_points_home_model.json")
        self.points_away_model = self._load_xgb("team_points_away_model.json")

        self.player_models: dict[str, XGBRegressor] = {
            "pts": self._load_xgb("player_pts_model.json"),
            "trb": self._load_xgb("player_reb_model.json"),
            "ast": self._load_xgb("player_ast_model.json"),
            "fg3": self._load_xgb("player_3pm_model.json"),
            "stl": self._load_xgb("player_stl_model.json"),
            "blk": self._load_xgb("player_blk_model.json"),
        }
        log.info("Tum modeller yuklendi.")

    @staticmethod
    def _load_json(filename: str) -> dict:
        with open(MODELS_DIR / filename, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_xgb(filename: str) -> XGBRegressor:
        model = XGBRegressor()
        model.load_model(MODELS_DIR / filename)
        return model


# ---------------------------------------------------------------
# Bayesian blend (prior <-> in-season)
# ---------------------------------------------------------------
def bayes_blend(prior: float, in_season: float | None, n_games: int, k: int) -> float:
    """w_in = n/(n+k). In-season veri yoksa prior doner."""
    if in_season is None or n_games <= 0:
        return prior
    w = n_games / (n_games + k)
    return w * in_season + (1 - w) * prior


# ---------------------------------------------------------------
# Takim Prior'lari
# ---------------------------------------------------------------
def load_team_priors() -> tuple[dict[str, dict], float, float]:
    """
    Kadro bazli net rating & pace projeksiyonunu ORtg/DRtg'ye ayirir.
    ORtg = prev_ORtg + (net_proj - prev_net)/2
    DRtg = prev_DRtg - (net_proj - prev_net)/2
    """
    proj = pd.read_csv(TEAM_PROJ_FILE)
    prof = pd.read_csv(TEAM_PROFILES_FILE)
    prev = prof[prof["season"] == prof["season"].max()].copy()
    prev["team_id"] = prev["team_id"].map(bbref)

    lg_rtg  = float(prev["off_rtg"].mean())
    lg_pace = float(prev["pace"].mean())

    prev_map = prev.set_index("team_id").to_dict("index")
    priors: dict[str, dict] = {}
    for _, r in proj.iterrows():
        tid = bbref(r["team_id"])
        p = prev_map.get(tid, {})
        prev_o = fnum(p.get("off_rtg"), lg_rtg)
        prev_d = fnum(p.get("def_rtg"), lg_rtg)
        delta  = fnum(r["net_rating_proj"]) - (prev_o - prev_d)
        ortg = prev_o + delta / 2
        drtg = prev_d - delta / 2
        pace = fnum(r["proj_pace"], lg_pace)
        priors[tid] = {
            "ortg": ortg,
            "drtg": drtg,
            "pace": pace,
            "net":  fnum(r["net_rating_proj"]),
            "hca":  fnum(r.get("empirical_hca"), 2.3),
            "exp_wins": fnum(r.get("exp_wins"), 41.0),
        }
    log.info("Takim prior'lari: %d takim | lig ORtg=%.1f pace=%.1f",
             len(priors), lg_rtg, lg_pace)
    return priors, lg_rtg, lg_pace


def team_feature_row(prior: dict, rest_days: float) -> dict[str, float]:
    """Prior'u egitimdeki rolling feature isimlerine map eder."""
    pts = prior["ortg"] * prior["pace"] / 100.0
    return {
        "roll_ortg_5": prior["ortg"],  "roll_ortg_10": prior["ortg"],
        "roll_drtg_5": prior["drtg"],  "roll_drtg_10": prior["drtg"],
        "roll_pace_5": prior["pace"],  "roll_pace_10": prior["pace"],
        "roll_pts_5":  pts,            "roll_pts_10":  pts,
        "rest_days":   float(rest_days),
    }


# ---------------------------------------------------------------
# Fikstur -> rest days
# ---------------------------------------------------------------
def build_rest_lookup(sched: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(team, date) -> rest days (ilk mac = cap)."""
    rows = []
    for _, g in sched.iterrows():
        rows.append((bbref(g["home_team"]), g["game_date"]))
        rows.append((bbref(g["away_team"]), g["game_date"]))
    df = pd.DataFrame(rows, columns=["team", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"])
    df["prev"] = df.groupby("team")["date"].shift(1)
    df["rest"] = (df["date"] - df["prev"]).dt.days.fillna(REST_DAYS_CAP)
    df["rest"] = df["rest"].clip(1, REST_DAYS_CAP).astype(int)
    return {(t, d.strftime("%Y-%m-%d")): r for t, d, r in zip(df["team"], df["date"], df["rest"])}


def games_played_before(sched: pd.DataFrame, team: str, date: str) -> int:
    m = ((sched["home_team"].map(bbref) == team) | (sched["away_team"].map(bbref) == team)) \
        & (sched["game_date"] < date)
    return int(m.sum())


# ---------------------------------------------------------------
# Oyuncu Prior'lari + Rotasyon
# ---------------------------------------------------------------
def load_player_priors() -> dict[str, list[dict]]:
    """Roster + MARCELS projeksiyonu + pozisyon/USG -> takim bazli oyuncu listesi."""
    roster = pd.read_csv(ROSTER_FILE)
    proj   = pd.read_csv(PLAYER_PROJ_FILE).set_index("player_id")
    prof   = pd.read_csv(PLAYER_PROFILES_FILE, low_memory=False)
    prof   = prof[prof["season"] == prof["season"].max()].copy()
    # Takas edilen oyuncular icin en cok dakika oynadigi satiri al
    sort_col = "mp" if "mp" in prof.columns else "mp_per_g"
    prof = prof.sort_values(sort_col, ascending=False).drop_duplicates("player_id")
    prof = prof.set_index("player_id")

    teams: dict[str, list[dict]] = {}
    for _, r in roster.iterrows():
        pid = str(r["player_id"])
        tid = bbref(r["team_id"])
        pj  = proj.loc[pid] if pid in proj.index else None
        pf  = prof.loc[pid] if pid in prof.index else None
        if pj is None:
            continue
        mpg = fnum(pj["proj_mpg"])
        teams.setdefault(tid, []).append({
            "player_id": pid,
            "name":      fix_name(str(r["player_name"])),
            "team_id":   tid,
            "position":  str(pf["pos"]) if pf is not None and pd.notna(pf.get("pos")) else "",
            "mpg":  mpg,
            "ppg":  fnum(pj["proj_ppg"]),
            "rpg":  fnum(pj["proj_rpg"]),
            "apg":  fnum(pj["proj_apg"]),
            "spg":  fnum(pj["proj_spg"]),
            "bpg":  fnum(pj["proj_bpg"]),
            "fg3m": fnum(pj["proj_fg3m"]),
            "ts":   fnum(pj["proj_ts_pct"], 0.55),
            "usg":  fnum(pf.get("usg_pct"), 18.0) if pf is not None else 18.0,
        })
    for tid in teams:
        teams[tid].sort(key=lambda p: p["mpg"], reverse=True)
    log.info("Oyuncu prior'lari: %d takim, %d oyuncu",
             len(teams), sum(len(v) for v in teams.values()))
    return teams


def allocate_minutes(players: list[dict]) -> list[dict]:
    """
    Rotasyon secimi + 240 dk kisiti (hiyerarsik).
    - proj_mpg sirali, mpg >= 12 olanlardan max 10 (en az 8) oyuncu
    - Toplam 240'i asarsa kesinti (40 - mpg) ile orantili dagitilir:
      yildizlarin dakikasi "yapiskan", kesintiyi bench ustlenir.
    - Toplam 240'in altindaysa fark mpg ile orantili eklenir.
    - Bireysel cap 38 dk, taban 6 dk.
    """
    rot = [p for p in players if p["mpg"] >= ROTATION_MIN_MPG][:ROTATION_MAX]
    if len(rot) < ROTATION_MIN:
        rot = players[:ROTATION_MIN]
    mins = np.array([min(max(p["mpg"], 6.0), MAX_PLAYER_MINUTES) for p in rot], dtype=float)

    for _ in range(30):
        diff = TOTAL_GAME_MINUTES - mins.sum()
        if abs(diff) < 0.05:
            break
        if diff < 0:
            w = np.where(mins > 6.0, 40.0 - mins, 0.0)
        else:
            w = np.where(mins < MAX_PLAYER_MINUTES, mins, 0.0)
        if w.sum() <= 0:
            break
        mins = np.clip(mins + diff * w / w.sum(), 6.0, MAX_PLAYER_MINUTES)

    out = []
    for i, (p, m) in enumerate(zip(rot, mins)):
        out.append({**p, "alloc_min": float(m), "role": "starter" if i < 5 else "bench"})
    return out


# ---------------------------------------------------------------
# Stage 1: Takim Tahmini
# ---------------------------------------------------------------
def predict_game_team(
    registry: ModelRegistry,
    home: dict, away: dict,
    home_rest: int, away_rest: int,
    lg_rtg: float, lg_pace: float,
    is_opener: bool,
) -> dict[str, float]:
    pace_cols   = registry.team_meta["pace_features"]
    points_cols = registry.team_meta["points_features"]

    hf = team_feature_row(home, home_rest)
    af = team_feature_row(away, away_rest)

    def row(base: dict, opp: dict, is_home: int, cols: list[str]) -> pd.DataFrame:
        r = {}
        for c in cols:
            if c.endswith("_opp_pregame"):
                r[c] = opp.get(c.replace("_opp_pregame", ""), 0.0)
            elif c == "is_home":
                r[c] = float(is_home)
            else:
                r[c] = base.get(c, 0.0)
        return pd.DataFrame([r])[cols]

    # --- Ratings modeli (Dean Oliver / log5 tarzi) ---
    pace_rt = home["pace"] * away["pace"] / lg_pace
    hca = home["hca"]
    home_pts_rt = home["ortg"] * away["drtg"] / lg_rtg * pace_rt / 100 + hca / 2
    away_pts_rt = away["ortg"] * home["drtg"] / lg_rtg * pace_rt / 100 - hca / 2

    # Yorgunluk: B2B ise ~1.5 puan ceza
    if home_rest == 1:
        home_pts_rt -= 1.5
    if away_rest == 1:
        away_pts_rt -= 1.5

    # --- XGBoost team modelleri ---
    pace_x = (float(registry.pace_model.predict(row(hf, af, 1, pace_cols))[0]) +
              float(registry.pace_model.predict(row(af, hf, 0, pace_cols))[0])) / 2
    home_x = float(registry.points_home_model.predict(row(hf, af, 1, points_cols))[0])
    away_x = float(registry.points_away_model.predict(row(af, hf, 0, points_cols))[0])

    pace = 0.5 * pace_rt + 0.5 * pace_x
    home_pts = (1 - XGB_TEAM_WEIGHT) * home_pts_rt + XGB_TEAM_WEIGHT * home_x
    away_pts = (1 - XGB_TEAM_WEIGHT) * away_pts_rt + XGB_TEAM_WEIGHT * away_x

    spread = home_pts - away_pts
    total  = home_pts + away_pts
    std    = SCORE_STD_OPENER if is_opener else SCORE_STD_BASE
    hwp    = float(norm.cdf(spread / std))

    return {
        "pred_game_pace":  round(pace, 1),
        "pred_home_score": round(home_pts, 1),
        "pred_away_score": round(away_pts, 1),
        "pred_spread":     round(spread, 1),
        "pred_total":      round(total, 1),
        "home_win_prob":   round(hwp * 100, 1),
        "away_win_prob":   round((1 - hwp) * 100, 1),
        "ratings_home_pts": round(home_pts_rt, 1),
        "ratings_away_pts": round(away_pts_rt, 1),
        "xgb_home_pts":    round(home_x, 1),
        "xgb_away_pts":    round(away_x, 1),
    }


# ---------------------------------------------------------------
# Stage 2: Oyuncu Prop Tahmini
# ---------------------------------------------------------------
def predict_player_props(
    registry: ModelRegistry,
    rotation: list[dict],
    team_prior: dict, opp_prior: dict,
    is_home: int, rest_days: int,
    game_pace: float, team_score: float,
) -> list[dict]:
    feats = registry.player_meta["features"]
    rows = []
    for p in rotation:
        m   = p["alloc_min"]
        mpg = max(p["mpg"], 1.0)
        r = m / mpg   # dakika oranina gore per-game prior'u olcekle
        f = {
            "p_roll_min_5": m, "p_roll_min_10": m,
            "p_roll_pts_5": p["ppg"] * r,  "p_roll_pts_10": p["ppg"] * r,
            "p_roll_trb_5": p["rpg"] * r,  "p_roll_trb_10": p["rpg"] * r,
            "p_roll_ast_5": p["apg"] * r,  "p_roll_ast_10": p["apg"] * r,
            "p_roll_fg3_5": p["fg3m"] * r, "p_roll_fg3_10": p["fg3m"] * r,
            "p_roll_stl_5": p["spg"] * r,  "p_roll_stl_10": p["spg"] * r,
            "p_roll_blk_5": p["bpg"] * r,  "p_roll_blk_10": p["bpg"] * r,
            "p_roll_usg_5": p["usg"], "p_roll_usg_10": p["usg"],
            "p_roll_ts_5":  p["ts"],  "p_roll_ts_10":  p["ts"],
            "p_pts_per_min_10": p["ppg"] / mpg,
            "p_trb_per_min_10": p["rpg"] / mpg,
            "p_ast_per_min_10": p["apg"] / mpg,
            "player_rest_days": float(rest_days),
            "is_home": float(is_home),
            "roll_pace_10": team_prior["pace"],
            "roll_ortg_10": team_prior["ortg"],
            "roll_drtg_10": team_prior["drtg"],
            "roll_drtg_10_opp_pregame": opp_prior["drtg"],
            "roll_pace_10_opp_pregame": opp_prior["pace"],
            "actual_game_pace": game_pace,
        }
        rows.append(f)
    X = pd.DataFrame(rows)[feats].fillna(0)

    xgb = {stat: np.clip(model.predict(X), 0, None) for stat, model in registry.player_models.items()}
    prior_key = {"pts": "p_roll_pts_10", "trb": "p_roll_trb_10", "ast": "p_roll_ast_10",
                 "fg3": "p_roll_fg3_10", "stl": "p_roll_stl_10", "blk": "p_roll_blk_10"}
    preds = {
        stat: XGB_PLAYER_WEIGHT * xgb[stat] + (1 - XGB_PLAYER_WEIGHT) * X[prior_key[stat]].to_numpy()
        for stat in xgb
    }

    # Takim skoru tutarliligi (usage-aware): oyuncu sayi toplami ~ takim skoru.
    # Fark (D) oyunculara dagitilirken dusuk usage'li rol oyunculari daha cok,
    # yildizlar daha az etkilenir (gercekte yildizlarin sut hacmi korunur).
    pts = preds["pts"]
    pts_sum = float(pts.sum())
    if pts_sum > 0:
        target = float(np.clip(team_score, pts_sum * 0.85, pts_sum * 1.15))
        diff = target - pts_sum
        usg = X["p_roll_usg_10"].clip(5, 38).to_numpy()
        w = pts * ((40.0 - usg) if diff < 0 else usg)
        if w.sum() > 0:
            pts = pts + diff * w / w.sum()
        preds["pts"] = np.clip(pts, 0, None)

    results = []
    for i, p in enumerate(rotation):
        results.append({
            "player_id":      p["player_id"],
            "player_name":    p["name"],
            "team_id":        espn_abbr(p["team_id"]),
            "position":       p["position"],
            "role":           p["role"],
            "projected_mins": round(p["alloc_min"], 1),
            "proj_pts": round(float(preds["pts"][i]), 1),
            "proj_trb": round(float(preds["trb"][i]), 1),
            "proj_ast": round(float(preds["ast"][i]), 1),
            "proj_fg3": round(float(preds["fg3"][i]), 1),
            "proj_stl": round(float(preds["stl"][i]), 1),
            "proj_blk": round(float(preds["blk"][i]), 1),
            "season_proj_ppg": round(p["ppg"], 1),
        })
    return results


# ---------------------------------------------------------------
# Zaman donusumu
# ---------------------------------------------------------------
def convert_et_to_tr(date_str: str, time_et_str: str) -> str:
    try:
        dt_et = datetime.strptime(f"{date_str} {time_et_str}", "%Y-%m-%d %I:%M %p")
        return (dt_et + timedelta(hours=ET_TO_TR_OFFSET_HOURS)).strftime("%d.%m.%Y %H:%M")
    except Exception:
        return f"{date_str.replace('-', '.')} --:--"


# ---------------------------------------------------------------
# Tek gun pipeline
# ---------------------------------------------------------------
def run_date(target_date: str, registry, team_priors, lg_rtg, lg_pace,
             player_priors, sched, rest_lookup) -> dict | None:
    games_today = sched[sched["game_date"] == target_date]
    if games_today.empty:
        log.warning("%s icin mac yok.", target_date)
        return None

    log.info("=" * 65)
    log.info("Tarih: %s | Mac sayisi: %d", target_date, len(games_today))

    all_games, all_props = [], []
    for _, g in games_today.iterrows():
        home, away = bbref(g["home_team"]), bbref(g["away_team"])
        gid = str(g["game_id"])
        h_rest = rest_lookup.get((home, target_date), REST_DAYS_CAP)
        a_rest = rest_lookup.get((away, target_date), REST_DAYS_CAP)
        h_gp = games_played_before(sched, home, target_date)
        a_gp = games_played_before(sched, away, target_date)
        is_opener = (h_gp == 0 or a_gp == 0)

        hp, ap = team_priors[home], team_priors[away]
        s1 = predict_game_team(registry, hp, ap, h_rest, a_rest, lg_rtg, lg_pace, is_opener)

        h_rot = allocate_minutes(player_priors.get(home, []))
        a_rot = allocate_minutes(player_priors.get(away, []))
        h_props = predict_player_props(registry, h_rot, hp, ap, 1, h_rest,
                                       s1["pred_game_pace"], s1["pred_home_score"])
        a_props = predict_player_props(registry, a_rot, ap, hp, 0, a_rest,
                                       s1["pred_game_pace"], s1["pred_away_score"])

        winner = home if s1["home_win_prob"] >= 50 else away
        all_games.append({
            "game_id": gid,
            "game_date": target_date.replace("-", ""),
            "match_time_tr": convert_et_to_tr(target_date, str(g.get("game_time_et", ""))),
            "home_team": espn_abbr(home),
            "away_team": espn_abbr(away),
            "home_team_full": TEAM_FULL_NAMES.get(home, home),
            "away_team_full": TEAM_FULL_NAMES.get(away, away),
            "home_logo": LOGO_TPL.format(abbr=espn_abbr(home).lower()),
            "away_logo": LOGO_TPL.format(abbr=espn_abbr(away).lower()),
            "proj_home_score": s1["pred_home_score"],
            "proj_away_score": s1["pred_away_score"],
            "proj_spread": s1["pred_spread"],
            "proj_total": s1["pred_total"],
            "pred_game_pace": s1["pred_game_pace"],
            "home_win_prob": s1["home_win_prob"],
            "away_win_prob": s1["away_win_prob"],
            "predicted_winner": espn_abbr(winner),
            "home_rest_days": h_rest,
            "away_rest_days": a_rest,
            "home_net_rating_prior": round(hp["net"], 2),
            "away_net_rating_prior": round(ap["net"], 2),
            "model_breakdown": {
                "ratings": [s1["ratings_home_pts"], s1["ratings_away_pts"]],
                "xgboost": [s1["xgb_home_pts"], s1["xgb_away_pts"]],
                "xgb_weight": XGB_TEAM_WEIGHT,
                "in_season_weight_home": round(h_gp / (h_gp + K_TEAM_PRIOR), 2) if h_gp else 0.0,
            },
            "is_nba_cup": bool(g.get("is_nba_cup_group", False)),
        })
        for p in h_props + a_props:
            all_props.append({**p, "game_id": gid})

        log.info("  %s @ %s  ->  %.1f - %.1f | spread %+.1f | total %.1f | ev %%%.1f | rest %d/%d",
                 away, home, s1["pred_away_score"], s1["pred_home_score"],
                 s1["pred_spread"], s1["pred_total"], s1["home_win_prob"], a_rest, h_rest)

    compact = target_date.replace("-", "")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": compact,
        "methodology": "preseason_prior(MARCELS+roster net rating) -> XGBoost; in-season Bayesian blend",
        "games": all_games,
        "player_props": all_props,
    }
    (OUTPUT_DIR / f"predictions_{compact}.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def print_summary(out: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {out['date']}")
    print(f"  {'MAC':<14} {'SKOR (DEP-EV)':>15} {'SPREAD':>8} {'TOTAL':>7} {'EV%':>6}  KAZANAN")
    print("-" * 78)
    for g in out["games"]:
        m = f"{g['away_team']} @ {g['home_team']}"
        print(f"  {m:<14} {g['proj_away_score']:>7.1f}-{g['proj_home_score']:<7.1f}"
              f" {g['proj_spread']:>+8.1f} {g['proj_total']:>7.1f} {g['home_win_prob']:>5.1f}%  "
              f"{g['predicted_winner']}")
    # Gunun en yuksek skorer tahminleri
    top = sorted(out["player_props"], key=lambda p: p["proj_pts"], reverse=True)[:5]
    print("-" * 78)
    print("  Top 5 sayi tahmini:")
    for p in top:
        print(f"    {p['player_name']:<26} {p['team_id']:<5} {p['projected_mins']:>4.1f} dk"
              f"  {p['proj_pts']:>4.1f} pts  {p['proj_trb']:>4.1f} reb  {p['proj_ast']:>4.1f} ast")


# ---------------------------------------------------------------
# Ana
# ---------------------------------------------------------------
def main(start_date: str, days: int) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    registry = ModelRegistry()
    team_priors, lg_rtg, lg_pace = load_team_priors()
    player_priors = load_player_priors()

    sched = pd.read_csv(SCHEDULE_FILE)
    rest_lookup = build_rest_lookup(sched)

    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    outputs = []
    for i in range(days):
        ds = (d0 + timedelta(days=i)).strftime("%Y-%m-%d")
        out = run_date(ds, registry, team_priors, lg_rtg, lg_pace,
                       player_priors, sched, rest_lookup)
        if out:
            outputs.append(out)
            print_summary(out)

    if outputs:
        latest = outputs[0]
        (OUTPUT_DIR / "predictions_latest.json").write_text(
            json.dumps(latest, indent=2, ensure_ascii=False), encoding="utf-8")
        n_g = sum(len(o["games"]) for o in outputs)
        n_p = sum(len(o["player_props"]) for o in outputs)
        log.info("Bitti: %d gun, %d mac, %d oyuncu tahmini. latest=%s",
                 len(outputs), n_g, n_p, latest["date"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Offline Tahmin Motoru")
    parser.add_argument("--date", type=str, default=SEASON_START,
                        help="Baslangic tarihi (YYYY-MM-DD veya YYYYMMDD)")
    parser.add_argument("--days", type=int, default=1, help="Kac gun tahmin edilecek")
    args = parser.parse_args()
    raw = args.date.strip()
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    main(raw, args.days)
