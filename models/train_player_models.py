"""
models/train_player_models.py
==============================
Phase 2 — Stage 2: Oyuncu Prop Modellerini XGBoost ile Eğit.

6 ayrı model: PTS, TRB, AST, 3PM (fg3), STL, BLK

Çalıştırma:
    python models/train_player_models.py

Çıktılar:
    models/player_pts_model.json
    models/player_reb_model.json
    models/player_ast_model.json
    models/player_3pm_model.json
    models/player_stl_model.json
    models/player_blk_model.json
    models/player_model_meta.json
    models/evaluation_report.txt (güncellenir)
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)

REPORT_LINES = []

def log(msg: str):
    print(msg)
    REPORT_LINES.append(msg)

# ---------------------------------------------------------------
# 1. Oyuncu Feature Tablosu Yükle
# ---------------------------------------------------------------
PLAYER_FILE = "data/player_features.parquet"

if not os.path.exists(PLAYER_FILE):
    raise FileNotFoundError(
        f"'{PLAYER_FILE}' bulunamadı. Önce 'python etl/build_training_features.py' çalıştırın."
    )

log("=" * 72)
log("  NBA OYUNCU PROP MODELLERİ — XGBoost Eğitimi")
log("=" * 72)
log(f"\n1. Oyuncu feature tablosu yükleniyor: {PLAYER_FILE}")
df = pd.read_parquet(PLAYER_FILE)
log(f"   -> {len(df):,} satır, {df.shape[1]} kolon yüklendi.")

# ---------------------------------------------------------------
# 2. Minimum Oynama Filtresi (anlamsız kısa süre oynayanları at)
# ---------------------------------------------------------------
MIN_MINUTES = 8
df = df[df["minutes"] >= MIN_MINUTES].copy()
log(f"   -> {MIN_MINUTES}+ dakika filtresi sonrası: {len(df):,} satır")

# ---------------------------------------------------------------
# 3. Feature Seti
# ---------------------------------------------------------------
# Temel oyuncu rolling features
PLAYER_BASE_FEATURES = [
    "p_roll_min_5",  "p_roll_min_10",
    "p_roll_pts_5",  "p_roll_pts_10",
    "p_roll_trb_5",  "p_roll_trb_10",
    "p_roll_ast_5",  "p_roll_ast_10",
    "p_roll_fg3_5",  "p_roll_fg3_10",
    "p_roll_stl_5",  "p_roll_stl_10",
    "p_roll_blk_5",  "p_roll_blk_10",
    "p_roll_usg_5",  "p_roll_usg_10",
    "p_roll_ts_5",   "p_roll_ts_10",
    "p_pts_per_min_10",
    "p_trb_per_min_10",
    "p_ast_per_min_10",
    "player_rest_days",
    "is_home",
    # Stage 1 context (takım/rakip bağlamı)
    "roll_pace_10",
    "roll_ortg_10",
    "roll_drtg_10",
    "roll_drtg_10_opp_pregame",   # rakip savunma kalitesi
    "roll_pace_10_opp_pregame",
    "actual_game_pace",           # Stage 1 tahmininde gerçek ile değiştir
]

# Mevcut kolonları seç (bazıları join'de kaybolmuş olabilir)
AVAILABLE = [c for c in PLAYER_BASE_FEATURES if c in df.columns]
log(f"\n2. Kullanılan özellikler ({len(AVAILABLE)}): {AVAILABLE}")

# ---------------------------------------------------------------
# 4. Hedef Değişkenler
# ---------------------------------------------------------------
TARGETS = {
    "pts": "player_pts_model",
    "trb": "player_reb_model",
    "ast": "player_ast_model",
    "fg3": "player_3pm_model",
    "stl": "player_stl_model",
    "blk": "player_blk_model",
}

# ---------------------------------------------------------------
# 5. Time-Series Split
# ---------------------------------------------------------------
log("\n3. Time-series split (son sezon = validation)...")
seasons = sorted(df["season"].unique())
val_season = seasons[-1]
train_seasons = seasons[:-1]
log(f"   Eğitim  : {train_seasons}")
log(f"   Valida. : {val_season}")

df_train = df[df["season"].isin(train_seasons)].copy()
df_val   = df[df["season"] == val_season].copy()
log(f"   Train: {len(df_train):,} | Val: {len(df_val):,}")

# ---------------------------------------------------------------
# 6. XGBoost Parametreleri
# ---------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.035,
    subsample=0.75,
    colsample_bytree=0.75,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

# ---------------------------------------------------------------
# 7. Her Hedef için Eğitim
# ---------------------------------------------------------------
results = {}

for target_col, model_name in TARGETS.items():
    if target_col not in df.columns:
        log(f"\n[ATLA] '{target_col}' kolonu bulunamadı.")
        continue

    log(f"\n" + "-" * 60)
    log(f"MODEL: {model_name.upper()} (hedef: {target_col})")
    log("-" * 60)

    X_tr = df_train[AVAILABLE].fillna(0)
    y_tr = df_train[target_col]
    X_vl = df_val[AVAILABLE].fillna(0)
    y_vl = df_val[target_col]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False,
    )

    preds = model.predict(X_vl)
    preds = np.clip(preds, 0, None)  # negatif tahmin olamaz

    rmse = np.sqrt(mean_squared_error(y_vl, preds))
    mae  = mean_absolute_error(y_vl, preds)
    r2   = r2_score(y_vl, preds)

    save_path = f"models/{model_name}.json"
    model.save_model(save_path)

    log(f"   RMSE : {rmse:.3f}")
    log(f"   MAE  : {mae:.3f}")
    log(f"   R²   : {r2:.4f}")
    log(f"   Kaydedildi -> {save_path}")

    results[model_name] = {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4)}

    # Top 5 feature importance
    fi = pd.Series(model.feature_importances_, index=AVAILABLE).sort_values(ascending=False)
    log(f"   Top 5 Feature: {list(fi.head(5).index)}")

# ---------------------------------------------------------------
# 8. Meta Kaydet
# ---------------------------------------------------------------
meta = {
    "features": AVAILABLE,
    "targets": TARGETS,
    "min_minutes_filter": MIN_MINUTES,
    "val_season": str(val_season),
    "results": results,
}
with open("models/player_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
log("\n   Meta dosyası -> models/player_model_meta.json")

# ---------------------------------------------------------------
# 9. Evaluation Report Güncelle
# ---------------------------------------------------------------
report_path = "models/evaluation_report.txt"
mode = "a" if os.path.exists(report_path) else "w"
with open(report_path, mode, encoding="utf-8") as f:
    f.write("\n\n" + "\n".join(REPORT_LINES))
log(f"   Rapor güncellendi -> {report_path}")

log("\n" + "=" * 72)
log("  PHASE 2B TAMAMLANDI: Oyuncu prop modelleri hazır!")
log("=" * 72)
