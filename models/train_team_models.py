"""
models/train_team_models.py
============================
Phase 2 — Stage 1: Takım Pace ve Puan Modellerini XGBoost ile Eğit.

Çalıştırma:
    python models/train_team_models.py

Çıktılar:
    models/team_pace_model.json
    models/team_points_home_model.json
    models/team_points_away_model.json
    models/evaluation_report.txt
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
# 1. Eğitim Verisi Yükle
# ---------------------------------------------------------------
TEAM_FEATURES_FILE = "data/team_features.parquet"

if not os.path.exists(TEAM_FEATURES_FILE):
    raise FileNotFoundError(
        f"'{TEAM_FEATURES_FILE}' bulunamadı. Önce 'python etl/build_training_features.py' çalıştırın."
    )

log("=" * 72)
log("  NBA TAKIM MODELLERİ — XGBoost Eğitimi")
log("=" * 72)
log(f"\n1. Takım feature tablosu yükleniyor: {TEAM_FEATURES_FILE}")
df = pd.read_parquet(TEAM_FEATURES_FILE)
log(f"   -> {len(df):,} satır, {df.shape[1]} kolon yüklendi.")

# ---------------------------------------------------------------
# 2. Feature & Target Tanımı
# ---------------------------------------------------------------
PACE_FEATURES = [
    "roll_pace_5", "roll_pace_10",
    "rest_days",
    "roll_pace_10_opp_pregame",
    "rest_days_opp_pregame",
    "is_home",
]

POINTS_FEATURES = [
    "roll_ortg_5", "roll_ortg_10",
    "roll_drtg_5", "roll_drtg_10",
    "roll_pts_5", "roll_pts_10",
    "roll_pace_5", "roll_pace_10",
    "rest_days",
    "roll_ortg_10_opp_pregame",
    "roll_drtg_10_opp_pregame",
    "roll_pts_10_opp_pregame",
    "roll_pace_10_opp_pregame",
    "rest_days_opp_pregame",
    "is_home",
]

PACE_TARGET    = "actual_game_pace"
HOME_PTS_TARGET = "team_pts"    # home perspective (is_home=1 → kendi puanı)
AWAY_PTS_TARGET = "team_pts_opp"

# Gerekli sütunları kontrol et; opp_pregame isim eşlemesi
# build_training_features.py'de join suffix "_opp_pregame" kullanılmış
REQUIRED_PACE   = [c for c in PACE_FEATURES   if c in df.columns]
REQUIRED_POINTS = [c for c in POINTS_FEATURES if c in df.columns]

log(f"\n2. Pace özellikleri ({len(REQUIRED_PACE)}): {REQUIRED_PACE}")
log(f"   Puan özellikleri ({len(REQUIRED_POINTS)}): {REQUIRED_POINTS}")

# ---------------------------------------------------------------
# 3. Time-Series Split (son sezon = validation)
# ---------------------------------------------------------------
log("\n3. Time-series split uygulanıyor (son sezon = validation set)...")
seasons = sorted(df["season"].unique())
val_season = seasons[-1]
train_seasons = seasons[:-1]
log(f"   Eğitim sezonları : {train_seasons}")
log(f"   Validation sezonu: {val_season}")

df_train = df[df["season"].isin(train_seasons)].copy()
df_val   = df[df["season"] == val_season].copy()
log(f"   Train satırı: {len(df_train):,} | Val satırı: {len(df_val):,}")

# ---------------------------------------------------------------
# 4. XGBoost Yardımcı Fonksiyonu
# ---------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

def train_and_evaluate(
    model_name: str,
    feature_cols: list[str],
    target_col: str,
    df_tr: pd.DataFrame,
    df_vl: pd.DataFrame,
) -> XGBRegressor:
    """Modeli eğit, değerlendir ve kaydet."""
    X_tr = df_tr[feature_cols].fillna(0)
    y_tr = df_tr[target_col]
    X_vl = df_vl[feature_cols].fillna(0)
    y_vl = df_vl[target_col]

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False,
    )

    preds = model.predict(X_vl)
    rmse = np.sqrt(mean_squared_error(y_vl, preds))
    mae  = mean_absolute_error(y_vl, preds)
    r2   = r2_score(y_vl, preds)

    save_path = f"models/{model_name}.json"
    model.save_model(save_path)

    log(f"\n   [{model_name}]")
    log(f"   RMSE : {rmse:.3f}")
    log(f"   MAE  : {mae:.3f}")
    log(f"   R²   : {r2:.4f}")
    log(f"   Model kaydedildi -> {save_path}")

    return model

# ---------------------------------------------------------------
# 5. Model 1A — Game Pace
# ---------------------------------------------------------------
log("\n" + "-" * 60)
log("MODEL 1A: Game Pace (toplam pozisyon / 48 dk)")
log("-" * 60)
model_pace = train_and_evaluate(
    model_name="team_pace_model",
    feature_cols=REQUIRED_PACE,
    target_col=PACE_TARGET,
    df_tr=df_train,
    df_vl=df_val,
)

# ---------------------------------------------------------------
# 6. Model 1B — Home Team Points
# ---------------------------------------------------------------
log("\n" + "-" * 60)
log("MODEL 1B-HOME: Ev Sahibi Takım Puanı")
log("-" * 60)
df_train_home = df_train[df_train["is_home"] == 1]
df_val_home   = df_val[df_val["is_home"] == 1]
model_pts_home = train_and_evaluate(
    model_name="team_points_home_model",
    feature_cols=REQUIRED_POINTS,
    target_col=HOME_PTS_TARGET,
    df_tr=df_train_home,
    df_vl=df_val_home,
)

# ---------------------------------------------------------------
# 7. Model 1B — Away Team Points
# ---------------------------------------------------------------
log("\n" + "-" * 60)
log("MODEL 1B-AWAY: Deplasman Takımı Puanı")
log("-" * 60)
df_train_away = df_train[df_train["is_home"] == 0]
df_val_away   = df_val[df_val["is_home"] == 0]
model_pts_away = train_and_evaluate(
    model_name="team_points_away_model",
    feature_cols=REQUIRED_POINTS,
    target_col=HOME_PTS_TARGET,  # "team_pts" her zaman "bu takımın puanı" demek
    df_tr=df_train_away,
    df_vl=df_val_away,
)

# ---------------------------------------------------------------
# 8. Feature Importance Özeti
# ---------------------------------------------------------------
log("\n" + "-" * 60)
log("FEATURE IMPORTANCE — Pace Modeli (İlk 10)")
log("-" * 60)
fi = pd.Series(model_pace.feature_importances_, index=REQUIRED_PACE).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    log(f"   {feat:<40} {imp:.4f}")

# ---------------------------------------------------------------
# 9. Feature Meta Kaydet (inference için sütun sırası)
# ---------------------------------------------------------------
meta = {
    "pace_features":   REQUIRED_PACE,
    "points_features": REQUIRED_POINTS,
    "pace_target":     PACE_TARGET,
    "home_pts_target": HOME_PTS_TARGET,
    "away_pts_target": HOME_PTS_TARGET,
    "val_season":      str(val_season),
}
with open("models/team_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
log("\n   Meta dosyası kaydedildi -> models/team_model_meta.json")

# ---------------------------------------------------------------
# 10. Evaluation Report
# ---------------------------------------------------------------
report_path = "models/evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT_LINES))
log(f"\n   Değerlendirme raporu kaydedildi -> {report_path}")

log("\n" + "=" * 72)
log("  PHASE 2A TAMAMLANDI: Takım modelleri hazır!")
log("=" * 72)
