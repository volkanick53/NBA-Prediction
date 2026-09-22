"""
engine/predict_daily.py
========================
Günlük NBA tahmin motoru.

Kullanım:
    python -m engine.predict_daily --date 20250320

Adımlar:
    1. ESPN'den günün maç programını çek
    2. Sakatlık listesini çek; "Out" / "Doubtful" oyuncuları dışla
    3. Stage 1: Takım modelleriyle Pace ve Puan tahmini
    4. Stage 2: 240-dakika kısıtıyla oyuncu prop tahminleri
    5. Çıktıyı JSON dosyasına kaydet + BigQuery'e yaz
"""

from __future__ import annotations

import argparse
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

from engine import espn_client as espn
from engine import bigquery_writer as bq_writer

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("predict_daily")

# ---------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------
MODELS_DIR   = Path("models")
OUTPUT_DIR   = Path("output")
KEY_PATH     = "nba-analytics-503718-9f3bbd399bc1.json"

PLAYER_PARQUET = Path("data/player_features.parquet")
TEAM_PARQUET   = Path("data/team_features.parquet")

# Tahmin noktası belirsizliği (score std tahmini için)
SCORE_STD_EST = 12.0   # NBA'de puan farkı standart sapması ~11-13 puan


# ---------------------------------------------------------------
# Model Yükleyici
# ---------------------------------------------------------------
class ModelRegistry:
    """Tüm XGBoost modelleri ve meta bilgilerini yükler."""

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
        with open(path) as f:
            return json.load(f)

    def _load_xgb(self, filename: str) -> XGBRegressor:
        model = XGBRegressor()
        model.load_model(MODELS_DIR / filename)
        return model


# ---------------------------------------------------------------
# Takım Rolling Feature Tablosu
# ---------------------------------------------------------------
def load_team_lookup() -> dict[int, dict]:
    """
    team_features.parquet'ten her takımın en son rolling feature satırını yükle.
    Inference sırasında tarihsel ortalamalar olarak kullanılır.
    """
    if not TEAM_PARQUET.exists():
        log.warning("team_features.parquet bulunamadı. Sıfır feature ile devam ediliyor.")
        return {}

    df = pd.read_parquet(TEAM_PARQUET)
    # Her takımın en son (en güncel) satırını al
    df = df.sort_values("game_date").groupby("team_id").last().reset_index()
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[str(row["team_id"])] = row.to_dict()
    return lookup


def load_player_lookup() -> dict[int, dict]:
    """Her oyuncunun en son rolling feature satırını yükle."""
    if not PLAYER_PARQUET.exists():
        log.warning("player_features.parquet bulunamadı.")
        return {}

    df = pd.read_parquet(PLAYER_PARQUET)
    df = df.sort_values("game_date").groupby("player_id").last().reset_index()
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[str(row["player_id"])] = row.to_dict()
    return lookup


# ---------------------------------------------------------------
# Stage 1: Takım Tahmini
# ---------------------------------------------------------------
def predict_game_team(
    registry: ModelRegistry,
    team_lookup: dict,
    home_id: int | str,
    away_id: int | str,
) -> dict[str, float]:
    """
    Stage 1: Maç tempo ve puan tahmini.
    Döner: pred_game_pace, pred_home_score, pred_away_score,
           pred_spread, pred_total, home_win_prob
    """
    home_feats = team_lookup.get(str(home_id), {})
    away_feats = team_lookup.get(str(away_id), {})

    # Pace özellik vektörü (is_home = 1 perspektifinden)
    pace_cols   = registry.team_meta["pace_features"]
    points_cols = registry.team_meta["points_features"]

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

    # Pace tahmini (ortalama al)
    pace_home = float(registry.pace_model.predict(home_row[pace_cols].fillna(0))[0])
    pace_away = float(registry.pace_model.predict(away_row[pace_cols].fillna(0))[0])
    pred_pace = (pace_home + pace_away) / 2.0

    # Puan tahminleri
    pred_home_score = float(
        registry.points_home_model.predict(home_row[points_cols].fillna(0))[0]
    )
    pred_away_score = float(
        registry.points_away_model.predict(away_row[points_cols].fillna(0))[0]
    )

    # Spread & Total
    spread    = round(pred_home_score - pred_away_score, 1)
    total     = round(pred_home_score + pred_away_score, 1)

    # Kazanma Olasılığı (Normal CDF)
    home_win_prob = float(norm.cdf(spread / SCORE_STD_EST))

    return {
        "pred_game_pace":  round(pred_pace, 2),
        "pred_home_score": round(pred_home_score, 1),
        "pred_away_score": round(pred_away_score, 1),
        "pred_spread":     spread,
        "pred_total":      total,
        "home_win_prob":   round(home_win_prob, 4),
    }


# ---------------------------------------------------------------
# Stage 2: Oyuncu Prop Tahmini
# ---------------------------------------------------------------
TOTAL_GAME_MINUTES = 240.0   # 5 oyuncu × 48 dakika

def predict_player_props(
    registry: ModelRegistry,
    player_lookup: dict,
    active_players: list[dict],
    team_context: dict[str, float],
    opp_team_id: int | str,
    team_lookup: dict,
) -> list[dict]:
    """
    Stage 2: Aktif oyuncu listesi için prop tahminleri.
    240 dakika kısıtı dikkate alınır.
    """
    player_feats_list = registry.player_meta["features"]

    opp_ctx = team_lookup.get(str(opp_team_id), {})
    pred_pace = team_context.get("pred_game_pace", 97.0)

    results = []
    for player in active_players:
        pid  = player["player_id"]
        prow = player_lookup.get(str(pid), {})

        feat_row: dict[str, float] = {}
        for col in player_feats_list:
            if col == "actual_game_pace":
                feat_row[col] = pred_pace
            elif col.endswith("_opp_pregame"):
                orig = col.replace("_opp_pregame", "")
                feat_row[col] = float(opp_ctx.get(orig, 0) or 0)
            else:
                feat_row[col] = float(prow.get(col, 0) or 0)

        X = pd.DataFrame([feat_row])[player_feats_list].fillna(0)

        prop: dict[str, Any] = {
            "player_id":   pid,
            "name":        player["name"],
            "position":    player.get("position", ""),
            "proj_minutes": round(float(feat_row.get("p_roll_min_10", 0)), 1),
        }
        for stat, model in registry.player_models.items():
            raw = float(model.predict(X)[0])
            prop[f"proj_{stat}"] = round(max(raw, 0.0), 1)

        results.append(prop)

    # 240-dakika kısıtı: toplam projeksiyon dakikalarını normalize et
    total_proj_mins = sum(p["proj_minutes"] for p in results)
    if total_proj_mins > TOTAL_GAME_MINUTES and total_proj_mins > 0:
        scale = TOTAL_GAME_MINUTES / total_proj_mins
        for p in results:
            p["proj_minutes"] = round(p["proj_minutes"] * scale, 1)

    return results


# ---------------------------------------------------------------
# Ana Pipeline
# ---------------------------------------------------------------
def run(date_str: str) -> list[dict]:
    log.info("=" * 60)
    log.info("NBA Günlük Tahmin Motoru — Tarih: %s", date_str)
    log.info("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Model ve veri yükle
    registry     = ModelRegistry()
    team_lookup  = load_team_lookup()
    player_lookup = load_player_lookup()

    # ESPN'den günlük program + sakatlıklar
    games    = espn.get_scoreboard(date_str)
    injuries = espn.get_injuries()

    if not games:
        log.warning("Bu tarih için maç bulunamadı: %s", date_str)
        return []

    predictions = []
    for game in games:
        event_id    = game["event_id"]
        home_id     = game["home_team_id"]
        away_id     = game["away_team_id"]
        log.info(
            "İşleniyor: %s vs %s (event %s)",
            game["home_team_name"], game["away_team_name"], event_id,
        )

        # Stage 1
        stage1 = predict_game_team(registry, team_lookup, home_id, away_id)

        # Aktif kadrolar
        home_active = espn.get_active_lineup(home_id, injuries)
        away_active = espn.get_active_lineup(away_id, injuries)

        # Stage 2
        home_props = predict_player_props(
            registry, player_lookup, home_active, stage1, away_id, team_lookup
        )
        away_props = predict_player_props(
            registry, player_lookup, away_active, stage1, home_id, team_lookup
        )

        prediction = {
            **game,
            **stage1,
            "player_props": {
                "home": home_props,
                "away": away_props,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        predictions.append(prediction)

        log.info(
            "  %s %.1f — %s %.1f | Total: %.1f | HomeWinProb: %.1f%%",
            game["home_team_name"], stage1["pred_home_score"],
            game["away_team_name"], stage1["pred_away_score"],
            stage1["pred_total"],
            stage1["home_win_prob"] * 100,
        )

    # Çıktıyı dosyaya yaz
    out_dated  = OUTPUT_DIR / f"predictions_{date_str}.json"
    out_latest = OUTPUT_DIR / "predictions_latest.json"
    payload = {"date": date_str, "predictions": predictions}

    out_dated.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    out_latest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log.info("Tahminler kaydedildi -> %s", out_dated)
    log.info("En son tahmin -> %s", out_latest)

    # BigQuery'e yaz
    flat_predictions = []
    for p in predictions:
        flat = {k: v for k, v in p.items() if k != "player_props"}
        flat["player_props"] = p["player_props"].get("home", []) + p["player_props"].get("away", [])
        flat_predictions.append(flat)

    try:
        bq_writer.write_predictions(flat_predictions, date_str)
    except Exception as exc:
        log.error("BigQuery yazma hatası (devam ediliyor): %s", exc)

    log.info("=" * 60)
    log.info("TAMAMLANDI: %d maç tahmini üretildi.", len(predictions))
    log.info("=" * 60)
    return predictions


# ---------------------------------------------------------------
# CLI Giriş Noktası
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NBA Günlük Tahmin Motoru")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="Tahmin tarihi (YYYYMMDD). Varsayılan: bugün.",
    )
    args = parser.parse_args()
    run(args.date)
