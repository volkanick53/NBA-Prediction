"""
predict_oct20_offline.py
========================
20 Ekim 2026 maçları için ESPN API'ye bağımlı olmadan,
yerel dim_* CSV dosyaları ve eğitilmiş XGBoost modelleri ile
offline tahmin üretir.

Kullanım:
    python predict_oct20_offline.py

Çıktı:
    output/predictions_20261020.json  (düzeltilmiş format)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
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
log = logging.getLogger("predict_oct20")

# ---------------------------------------------------------------
# Sabitler / Dosya Yolları
# ---------------------------------------------------------------
MODELS_DIR   = Path("models")
OUTPUT_DIR   = Path("output")
DATA_DIR     = Path("data")

ROSTER_FILE          = Path("dim_current_rosters_2026_27.csv")
PLAYER_PROFILES_FILE = Path("dim_player_season_profiles.csv")
TEAM_PROFILES_FILE   = Path("dim_team_season_profiles.csv")
SCHEDULE_FILE        = Path("dim_schedule_2026_27.csv")

TEAM_PARQUET   = DATA_DIR / "team_features.parquet"
PLAYER_PARQUET = DATA_DIR / "player_features.parquet"

TARGET_DATE = "2026-10-20"  # varsayilan; --date ile degistirilebilir
SCORE_STD_EST = 11.5   # NBA puan farkı std sapması

# ESPN logo URL şablonu — NBA standart kısaltmalar (küçük harf)
LOGO_TPL = "https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr}.png"

# Zaman dilimi farkı: ET → TR (UTC+3): +8 saat
ET_TO_TR_OFFSET_HOURS = 8

# schedule'da bazı takımlar farklı kısaltmayla geliyor → normalize
TEAM_ABBR_MAP = {
    "NYK": "NY",   # Knicks
    "SAS": "SA",   # Spurs
    "NOP": "NO",   # Pelicans
    "UTA": "UTAH", # Jazz
    "BRK": "BKN",  # Nets
    "CHO": "CHA",  # Hornets
    "GSW": "GS",   # Warriors
    "NJN": "NJ",
}
# Takım sezon profillerindeki kısaltmalar ↔ schedule kısaltmaları aynı (kural tersi de gerekebilir)
ABBR_REVERSE_MAP = {v: k for k, v in TEAM_ABBR_MAP.items()}

TOTAL_GAME_MINUTES = 240.0  # 5 × 48


# ---------------------------------------------------------------
# Yardımcı: kısaltma normalizer
# ---------------------------------------------------------------
def norm_abbr(abbr: str) -> str:
    """Schedule→logo/ESPN kısaltmasına çevir."""
    return TEAM_ABBR_MAP.get(abbr.upper(), abbr.upper())


def sched_abbr(abbr: str) -> str:
    """Logo→schedule kısaltmasına çevir (roster eşleme için)."""
    return ABBR_REVERSE_MAP.get(abbr.upper(), abbr.upper())


# ---------------------------------------------------------------
# Model Yükleyici
# ---------------------------------------------------------------
class ModelRegistry:
    """Eğitilmiş XGBoost modelleri ve meta bilgilerini yükler."""

    def __init__(self):
        log.info("Modeller yükleniyor...")
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
        log.info("Tüm modeller yüklendi.")

    def _load_json(self, filename: str) -> dict:
        path = MODELS_DIR / filename
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_xgb(self, filename: str) -> XGBRegressor:
        model = XGBRegressor()
        model.load_model(MODELS_DIR / filename)
        return model


# ---------------------------------------------------------------
# Takım Feature Lookup (team_features.parquet)
# ---------------------------------------------------------------
def load_team_lookup() -> dict[str, dict]:
    """Her takımın son sezon rolling feature satırını yükler."""
    if not TEAM_PARQUET.exists():
        log.warning("team_features.parquet bulunamadı → boş lookup.")
        return {}

    df = pd.read_parquet(TEAM_PARQUET)
    df = df.sort_values("game_date").groupby("team_id").last().reset_index()
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        tid = str(row["team_id"]).upper()
        d   = row.to_dict()
        # Parquet kısaltması (SAS, NYK, BRK...) olduğu gibi ekle
        lookup[tid] = d
        # ESPN / logo kısaltması (SA, NY, BKN...) da ekle
        lookup[norm_abbr(tid)] = d
    return lookup


def load_player_lookup() -> dict[str, dict]:
    """Her oyuncunun son sezon rolling feature satırını yükler."""
    if not PLAYER_PARQUET.exists():
        log.warning("player_features.parquet bulunamadı → boş lookup.")
        return {}

    df = pd.read_parquet(PLAYER_PARQUET)
    df = df.sort_values("game_date").groupby("player_id").last().reset_index()
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[str(row["player_id"])] = row.to_dict()
    return lookup


# ---------------------------------------------------------------
# dim_team_season_profiles fallback feature builder
# ---------------------------------------------------------------
def load_team_profile_lookup() -> dict[str, dict]:
    """
    dim_team_season_profiles.csv'den en son sezonu okur.
    team_features.parquet'te bulunamayan takımlar için fallback.
    """
    if not TEAM_PROFILES_FILE.exists():
        return {}

    df = pd.read_csv(TEAM_PROFILES_FILE)
    # En son sezonu al
    latest = df["season"].max()
    df = df[df["season"] == latest].copy()

    # Feature adlarını model beklentisine uygun hale getir
    # off_rtg → roll_ortg_10, def_rtg → roll_drtg_10, pace → roll_pace_10
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        tid = str(row["team_id"]).upper()
        profile = {
            "roll_ortg_5":   row.get("off_rtg", 112.0),
            "roll_ortg_10":  row.get("off_rtg", 112.0),
            "roll_drtg_5":   row.get("def_rtg", 112.0),
            "roll_drtg_10":  row.get("def_rtg", 112.0),
            "roll_pts_5":    row.get("off_rtg", 112.0) * row.get("pace", 98.0) / 100.0,
            "roll_pts_10":   row.get("off_rtg", 112.0) * row.get("pace", 98.0) / 100.0,
            "roll_pace_5":   row.get("pace", 98.0),
            "roll_pace_10":  row.get("pace", 98.0),
            "rest_days":     2.0,
            "is_home":       0.0,
        }
        result[tid] = profile
        result[norm_abbr(tid)] = profile
        result[sched_abbr(tid)] = profile
    return result


# ---------------------------------------------------------------
# dim_player_season_profiles fallback
# ---------------------------------------------------------------
def load_player_profile_lookup() -> dict[str, dict]:
    """
    dim_player_season_profiles.csv'den son sezonu okur.
    Rolling feature bulunmayan oyuncular için fallback.
    """
    if not PLAYER_PROFILES_FILE.exists():
        return {}

    df = pd.read_csv(PLAYER_PROFILES_FILE, low_memory=False)
    # player_id sütununu index olarak kullan
    df["player_id"] = df["player_id"].astype(str)
    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        pid = str(row["player_id"])
        result[pid] = {
            "p_roll_min_5":   float(row.get("mp_per_g", 0) or 0),
            "p_roll_min_10":  float(row.get("mp_per_g", 0) or 0),
            "p_roll_pts_5":   float(row.get("pts_per_g", 0) or 0),
            "p_roll_pts_10":  float(row.get("pts_per_g", 0) or 0),
            "p_roll_trb_5":   float(row.get("trb_per_g", 0) or 0),
            "p_roll_trb_10":  float(row.get("trb_per_g", 0) or 0),
            "p_roll_ast_5":   float(row.get("ast_per_g", 0) or 0),
            "p_roll_ast_10":  float(row.get("ast_per_g", 0) or 0),
            "p_roll_fg3_5":   float(row.get("fg3_per_g", 0) or 0),
            "p_roll_fg3_10":  float(row.get("fg3_per_g", 0) or 0),
            "p_roll_stl_5":   float(row.get("stl_per_g", 0) or 0),
            "p_roll_stl_10":  float(row.get("stl_per_g", 0) or 0),
            "p_roll_blk_5":   float(row.get("blk_per_g", 0) or 0),
            "p_roll_blk_10":  float(row.get("blk_per_g", 0) or 0),
            "p_roll_usg_5":   float(row.get("usg_pct", 0) or 0),
            "p_roll_usg_10":  float(row.get("usg_pct", 0) or 0),
            "p_roll_ts_5":    float(row.get("ts_pct", 0) or 0),
            "p_roll_ts_10":   float(row.get("ts_pct", 0) or 0),
            "p_pts_per_min_10": (
                float(row.get("pts_per_g", 0) or 0) /
                max(float(row.get("mp_per_g", 1) or 1), 1)
            ),
            "p_trb_per_min_10": (
                float(row.get("trb_per_g", 0) or 0) /
                max(float(row.get("mp_per_g", 1) or 1), 1)
            ),
            "p_ast_per_min_10": (
                float(row.get("ast_per_g", 0) or 0) /
                max(float(row.get("mp_per_g", 1) or 1), 1)
            ),
            "player_rest_days": 2.0,
            "is_home": 0.0,
        }
    return result


# ---------------------------------------------------------------
# Stage 1: Takım Tahmini
# ---------------------------------------------------------------
def predict_game_team(
    registry: ModelRegistry,
    team_lookup: dict,
    home_abbr: str,
    away_abbr: str,
) -> dict[str, float]:
    """
    Stage 1: Maç tempo ve puan tahminleri.
    Döner: pred_game_pace, pred_home_score, pred_away_score,
           pred_spread, pred_total, home_win_prob
    """
    pace_cols   = registry.team_meta["pace_features"]
    points_cols = registry.team_meta["points_features"]

    home_feats = team_lookup.get(home_abbr, team_lookup.get(norm_abbr(home_abbr), {}))
    away_feats = team_lookup.get(away_abbr, team_lookup.get(norm_abbr(away_abbr), {}))

    if not home_feats:
        log.warning("  Home takım feature bulunamadı: %s → sıfır kullanılıyor", home_abbr)
    if not away_feats:
        log.warning("  Away takım feature bulunamadı: %s → sıfır kullanılıyor", away_abbr)

    def build_row(base: dict, opp: dict, is_home: int) -> pd.DataFrame:
        row: dict[str, float] = {}
        for col in set(pace_cols + points_cols):
            if col.endswith("_opp_pregame"):
                orig = col.replace("_opp_pregame", "")
                row[col] = float(opp.get(orig, 0) or 0)
            elif col == "is_home":
                row[col] = float(is_home)
            else:
                row[col] = float(base.get(col, 0) or 0)
        return pd.DataFrame([row])

    home_row = build_row(home_feats, away_feats, is_home=1)
    away_row = build_row(away_feats, home_feats, is_home=0)

    pace_home = float(registry.pace_model.predict(home_row[pace_cols].fillna(0))[0])
    pace_away = float(registry.pace_model.predict(away_row[pace_cols].fillna(0))[0])
    pred_pace = (pace_home + pace_away) / 2.0

    pred_home_score = float(
        registry.points_home_model.predict(home_row[points_cols].fillna(0))[0]
    )
    pred_away_score = float(
        registry.points_away_model.predict(away_row[points_cols].fillna(0))[0]
    )

    # ---- Kalibrasyon: Model R² düşük (~0.03) olduğu için ----
    # Rolling ortalama puan ile ağırlıklı blend uygula (%30 model, %70 rolling avg)
    # Bu özellikle sezon başında (sıfır context) daha gerçekçi sonuç verir.
    home_roll_pts = float(home_feats.get("roll_pts_10", 0) or home_feats.get("roll_pts_5", 0) or 0)
    away_roll_pts = float(away_feats.get("roll_pts_10", 0) or away_feats.get("roll_pts_5", 0) or 0)

    LEAGUE_AVG_PTS = 113.5   # NBA 2024-25 liga ortalaması
    HCA_BONUS      = 2.5     # Ev sahibi avantajı (puan)

    if home_roll_pts > 50:   # Geçerli rolling veri var
        blend_home = 0.30 * pred_home_score + 0.70 * (home_roll_pts + HCA_BONUS)
        blend_away = 0.30 * pred_away_score + 0.70 * (away_roll_pts - HCA_BONUS * 0.5)
    else:                    # Veri yok → liga ortalaması + HCA
        blend_home = 0.30 * pred_home_score + 0.70 * (LEAGUE_AVG_PTS + HCA_BONUS)
        blend_away = 0.30 * pred_away_score + 0.70 * (LEAGUE_AVG_PTS - HCA_BONUS * 0.5)

    pred_home_score = float(np.clip(blend_home, 95, 145))
    pred_away_score = float(np.clip(blend_away, 95, 145))

    spread        = round(pred_home_score - pred_away_score, 1)
    total         = round(pred_home_score + pred_away_score, 1)
    home_win_prob = float(norm.cdf(spread / SCORE_STD_EST))

    return {
        "pred_game_pace":  round(pred_pace, 1),
        "pred_home_score": round(pred_home_score, 1),
        "pred_away_score": round(pred_away_score, 1),
        "pred_spread":     spread,
        "pred_total":      total,
        "home_win_prob":   round(home_win_prob * 100, 1),
        "away_win_prob":   round((1 - home_win_prob) * 100, 1),
    }


# ---------------------------------------------------------------
# Stage 2: Oyuncu Prop Tahmini
# ---------------------------------------------------------------
def predict_player_props(
    registry: ModelRegistry,
    player_lookup: dict,
    player_profile_lookup: dict,
    active_players: list[dict],
    team_context: dict[str, float],
    opp_team_feats: dict,
    is_home: int,
) -> list[dict]:
    """
    Stage 2: 240-dakika kısıtıyla oyuncu prop tahminleri.
    """
    player_feats_list = registry.player_meta["features"]
    pred_pace = team_context.get("pred_game_pace", 97.0)

    results = []
    for player in active_players:
        pid  = player["player_id"]
        name = player["name"]

        # Rolling feature: önce parquet, yoksa season profile
        prow = player_lookup.get(str(pid), player_profile_lookup.get(str(pid), {}))

        if not prow:
            log.debug("  Oyuncu feature bulunamadı: %s (%s)", name, pid)

        feat_row: dict[str, float] = {}
        for col in player_feats_list:
            if col == "actual_game_pace":
                feat_row[col] = pred_pace
            elif col == "is_home":
                feat_row[col] = float(is_home)
            elif col.endswith("_opp_pregame"):
                orig = col.replace("_opp_pregame", "")
                feat_row[col] = float(opp_team_feats.get(orig, 0) or 0)
            else:
                feat_row[col] = float(prow.get(col, 0) or 0)

        X = pd.DataFrame([feat_row])[player_feats_list].fillna(0)

        # Projeksiyon dakikasını season profile'dan al (daha güvenilir)
        proj_min = float(prow.get("p_roll_min_10", prow.get("mp_per_g", 0)) or 0)

        prop: dict[str, Any] = {
            "player_id":     pid,
            "player_name":   name,
            "team_id":       player.get("team_id", ""),
            "position":      player.get("position", ""),
            "projected_mins": round(proj_min, 1),
        }
        for stat, model in registry.player_models.items():
            raw = float(model.predict(X)[0])
            prop[f"proj_{stat}"] = round(max(raw, 0.0), 1)

        results.append(prop)

    # ---- 240-dakika kısıtı: Her oyuncu max 40 dk, takım toplam = 240 dk ----
    # Önce 0 dakikalık oyuncuları (veri yok) takımın ortalama bench süresiyle doldur
    active_with_mins = [p for p in results if p["projected_mins"] > 0]
    zero_min_players = [p for p in results if p["projected_mins"] == 0]

    if active_with_mins:
        avg_bench_min = min(10.0, TOTAL_GAME_MINUTES / max(len(results), 10) * 0.5)
        for p in zero_min_players:
            p["projected_mins"] = avg_bench_min

    # Bireysel cap: max 40 dakika
    for p in results:
        p["projected_mins"] = min(p["projected_mins"], 40.0)

    # Takım toplam 240'a normalize et
    total_proj_mins = sum(p["projected_mins"] for p in results)
    if total_proj_mins > 0:
        scale = TOTAL_GAME_MINUTES / total_proj_mins
        for p in results:
            p["projected_mins"] = round(p["projected_mins"] * scale, 1)

    # ---- Takım toplam puana normalize et (Stage 2 kalibrasyon) ----
    team_proj_total = team_context["pred_home_score"] if is_home else team_context["pred_away_score"]
    raw_pts_sum = sum(p["proj_pts"] for p in results)

    if raw_pts_sum > 0 and team_proj_total > 0:
        pts_scale = team_proj_total / raw_pts_sum
        # Sert normalizasyonu yumuşat (ağırlıklı blend: %60 model, %40 takım total)
        blend = 0.60 + 0.40 * pts_scale
        for p in results:
            p["proj_pts"] = round(max(p["proj_pts"] * blend, 0.0), 1)

    # Sıralama: projeksiyon dakikasına göre azalan
    results.sort(key=lambda x: x["projected_mins"], reverse=True)
    return results


# ---------------------------------------------------------------
# Roster Yükleyici (dim_current_rosters_2026_27.csv)
# ---------------------------------------------------------------
def load_rosters(player_profile_lookup: dict) -> dict[str, list[dict]]:
    """
    Güncel roster CSV'sinden takım bazlı oyuncu listesi oluşturur.
    Pozisyon bilgisini dim_player_season_profiles'tan alır.
    """
    roster_df = pd.read_csv(ROSTER_FILE)
    profile_df = pd.read_csv(PLAYER_PROFILES_FILE, low_memory=False)

    # En son sezonu al (oyuncu pozisyonu için)
    latest_season = profile_df["season"].max() if "season" in profile_df.columns else None
    if latest_season:
        profile_latest = profile_df[profile_df["season"] == latest_season].copy()
    else:
        profile_latest = profile_df.copy()

    pos_map: dict[str, str] = {}
    if "player_id" in profile_latest.columns and "pos" in profile_latest.columns:
        for _, row in profile_latest.iterrows():
            pos_map[str(row["player_id"])] = str(row.get("pos", "") or "")

    rosters: dict[str, list[dict]] = {}
    for _, row in roster_df.iterrows():
        tid  = str(row["team_id"]).upper()
        pid  = str(row["player_id"])
        name = str(row["player_name"])
        pos  = pos_map.get(pid, "")
        rosters.setdefault(tid, []).append({
            "player_id": pid,
            "name":      name,
            "team_id":   tid,
            "position":  pos,
        })
    return rosters


# ---------------------------------------------------------------
# Maç Zamanı ET → TR Dönüştürücü
# ---------------------------------------------------------------
def convert_et_to_tr(date_str: str, time_et_str: str) -> str:
    """
    Örnek: '2026-10-20', '3:00 PM' → '20.10.2026 23:00'
    NBA maçları ET'de verilir; UTC+3 (Türkiye) için +8 saat eklenir.
    """
    try:
        from datetime import timedelta
        dt_str = f"{date_str} {time_et_str}"
        dt_et = datetime.strptime(dt_str, "%Y-%m-%d %I:%M %p")
        dt_tr = dt_et + __import__("datetime").timedelta(hours=ET_TO_TR_OFFSET_HOURS)
        return dt_tr.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return f"{date_str.replace('-', '.')} --:--"


# ---------------------------------------------------------------
# Ana Pipeline
# ---------------------------------------------------------------
def run():
    log.info("=" * 65)
    log.info("NBA Offline Tahmin — 20 Ekim 2026")
    log.info("=" * 65)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Modelleri yükle
    registry = ModelRegistry()

    # 2. Feature lookup'ları yükle
    team_lookup          = load_team_lookup()
    player_lookup        = load_player_lookup()

    # Fallback: season profiles
    team_profile_lookup  = load_team_profile_lookup()
    player_profile_lookup = load_player_profile_lookup()

    # Birleşik lookup: parquet rolling features öncelikli,
    # bulunamazsa season profile fallback
    combined_team_lookup: dict[str, dict] = {}
    # Önce season profile'ları yükle (düşük öncelik)
    for k, v in team_profile_lookup.items():
        combined_team_lookup[k.upper()] = v
    # Sonra parquet'i üzerine yaz (yüksek öncelik)
    for k, v in team_lookup.items():
        combined_team_lookup[k.upper()] = v

    log.info("Kombine takım lookup: %d anahtar", len(combined_team_lookup))
    # Hangi takımların parquet'ten geldiğini logla
    for abbr in ["DET", "BOS", "NYK", "PHI", "SAS", "OKC"]:
        feat = combined_team_lookup.get(abbr, {})
        log.info("  %s → roll_pts_10=%.1f, roll_ortg_10=%.1f",
                 abbr,
                 feat.get("roll_pts_10", 0),
                 feat.get("roll_ortg_10", 0))

    # 3. Roster yükle
    rosters = load_rosters(player_profile_lookup)
    log.info("Roster yüklendi: %d takım", len(rosters))

    # 4. 20 Ekim maçlarını schedule'dan al
    sched_df = pd.read_csv(SCHEDULE_FILE)
    games_today = sched_df[sched_df["game_date"] == TARGET_DATE].copy()
    log.info("20 Ekim maç sayısı: %d", len(games_today))

    if games_today.empty:
        log.error("Schedule'da 20 Ekim maçı bulunamadı!")
        return

    all_game_results   = []
    all_player_props   = []

    for idx, game_row in games_today.iterrows():
        away_sched = str(game_row["away_team"]).upper()   # schedule kısaltması
        home_sched = str(game_row["home_team"]).upper()
        game_id    = str(game_row["game_id"])
        time_et    = str(game_row.get("game_time_et", "12:00 AM"))
        match_time_tr = convert_et_to_tr(TARGET_DATE, time_et)

        # Logo kısaltmaları (ESPN)
        home_logo_abbr = norm_abbr(home_sched).lower()
        away_logo_abbr = norm_abbr(away_sched).lower()

        log.info("-" * 55)
        log.info("Maç: %s @ %s  (%s ET)", away_sched, home_sched, time_et)

        # Stage 1: takım tahmini
        stage1 = predict_game_team(
            registry,
            combined_team_lookup,
            home_abbr=home_sched,
            away_abbr=away_sched,
        )

        log.info(
            "  Stage1 → Ev: %.1f | Dep: %.1f | Total: %.1f | EV kazanma: %.1f%%",
            stage1["pred_home_score"], stage1["pred_away_score"],
            stage1["pred_total"], stage1["home_win_prob"],
        )

        # Stage 2: kadroları al
        home_roster = rosters.get(home_sched, [])
        away_roster = rosters.get(away_sched, [])

        # Top 10 oyuncu (az veri varsa tümünü kullan)
        home_active = home_roster[:10] if len(home_roster) >= 10 else home_roster
        away_active = away_roster[:10] if len(away_roster) >= 10 else away_roster

        home_team_feats = combined_team_lookup.get(home_sched.upper(), {})
        away_team_feats = combined_team_lookup.get(away_sched.upper(), {})

        home_props = predict_player_props(
            registry, player_lookup, player_profile_lookup,
            home_active, stage1, opp_team_feats=away_team_feats, is_home=1,
        )
        away_props = predict_player_props(
            registry, player_lookup, player_profile_lookup,
            away_active, stage1, opp_team_feats=home_team_feats, is_home=0,
        )

        winner = home_sched if stage1["home_win_prob"] >= 50 else away_sched

        game_result = {
            "game_id":           game_id,
            "game_date":         TARGET_DATE.replace("-", ""),
            "match_time_tr":     match_time_tr,
            "home_team":         norm_abbr(home_sched),
            "away_team":         norm_abbr(away_sched),
            "home_team_full":    home_sched,
            "away_team_full":    away_sched,
            "home_logo":         LOGO_TPL.format(abbr=home_logo_abbr),
            "away_logo":         LOGO_TPL.format(abbr=away_logo_abbr),
            "proj_home_score":   stage1["pred_home_score"],
            "proj_away_score":   stage1["pred_away_score"],
            "proj_spread":       stage1["pred_spread"],
            "proj_total":        stage1["pred_total"],
            "pred_game_pace":    stage1["pred_game_pace"],
            "home_win_prob":     stage1["home_win_prob"],
            "away_win_prob":     stage1["away_win_prob"],
            "predicted_winner":  norm_abbr(winner),
        }
        all_game_results.append(game_result)

        # Oyuncu prop'larını düzleştir ve game_id ekle
        for p in home_props + away_props:
            flat_p = {**p, "game_id": game_id}
            all_player_props.append(flat_p)

    # ---------------------------------------------------------------
    # Çıktıyı Oluştur
    # ---------------------------------------------------------------
    out_date_compact = TARGET_DATE.replace("-", "")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date":         out_date_compact,
        "games":        all_game_results,
        "player_props": all_player_props,
    }

    out_path    = OUTPUT_DIR / f"predictions_{out_date_compact}.json"
    latest_path = OUTPUT_DIR / "predictions_latest.json"
    payload_str = json.dumps(output, indent=2, ensure_ascii=False)
    out_path.write_text(payload_str, encoding="utf-8")
    latest_path.write_text(payload_str, encoding="utf-8")
    log.info("=" * 65)
    log.info("Tahminler kaydedildi -> %s", out_path)
    log.info("Latest guncellendi   -> %s", latest_path)
    log.info("Toplam mac: %d | Toplam oyuncu prop: %d", len(all_game_results), len(all_player_props))

    # Ozet tablosu
    print("\n" + "=" * 65)
    print(f"{'MAC':<25} {'EV':>6} {'DEP':>6} {'TOTAL':>7} {'EV%':>6}  KAZANAN")
    print("-" * 65)
    for g in all_game_results:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        print(
            f"{matchup:<25} {g['proj_home_score']:>6.1f} {g['proj_away_score']:>6.1f}"
            f" {g['proj_total']:>7.1f} {g['home_win_prob']:>5.1f}%  {g['predicted_winner']}"
        )
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NBA Offline Tahmin Motoru")
    parser.add_argument(
        "--date", type=str, default="2026-10-20",
        help="Tahmin tarihi (YYYY-MM-DD veya YYYYMMDD). Varsayilan: 2026-10-20"
    )
    args = parser.parse_args()
    # YYYYMMDD -> YYYY-MM-DD donusumu
    raw = args.date.strip()
    if len(raw) == 8 and raw.isdigit():
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    TARGET_DATE = raw
    run()
