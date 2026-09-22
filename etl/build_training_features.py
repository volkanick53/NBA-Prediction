import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

os.makedirs("data", exist_ok=True)
RAW_CACHE_FILE = "data/raw_boxscores_5seasons.parquet"

# ---------------------------------------------------------
# 1. BigQuery'den 5 Sezonluk Ham Veriyi Çek / Cache Kontrolü
# ---------------------------------------------------------
if os.path.exists(RAW_CACHE_FILE):
    print(f"1. Ham veriler yerel onbellekten yukleniyor: {RAW_CACHE_FILE}")
    df_raw = pd.read_parquet(RAW_CACHE_FILE)
else:
    print("1. BigQuery'den son 5 sezonun tum mac kutu istatistikleri indiriliyor...")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    query = f"""
        SELECT 
            game_id, game_date, season, season_type,
            player_id, player_name, team_id, opponent_id,
            is_home, minutes,
            fg, fga, fg_pct, fg3, fg3a, ft, fta,
            orb, drb, trb, ast, stl, blk, tov, pf, pts,
            plus_minus, ts_pct, usg_pct, off_rtg, def_rtg, bpm
        FROM `{PROJECT_ID}.{DATASET_ID}.fact_player_boxscores`
        WHERE minutes IS NOT NULL
        ORDER BY game_date ASC
    """
    df_raw = client.query(query).to_dataframe()
    df_raw.to_parquet(RAW_CACHE_FILE, index=False)
    print(f"   -> Toplam {len(df_raw):,} satir veri BigQuery'den cekildi ve onbellegi alindi.")

# ---------------------------------------------------------
# 2. Veri Tiplerini Temizleme ve Standartlaştırma
# ---------------------------------------------------------
print("2. Veri tipleri ve sayisal donusumler yapiliyor...")

numeric_cols = [
    "minutes", "fg", "fga", "fg3", "fg3a", "ft", "fta",
    "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts",
    "plus_minus", "ts_pct", "usg_pct", "off_rtg", "def_rtg", "bpm"
]

for col in numeric_cols:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0.0)

df_raw["game_date"] = pd.to_datetime(df_raw["game_date"])
df_raw["is_home"] = df_raw["is_home"].astype(str).str.lower().isin(["true", "1"]).astype(int)

# ---------------------------------------------------------
# 3. Takım Seviyesine İndirgeme ve Maç Dinamikleri
# ---------------------------------------------------------
print("3. Takim seviyesi istatistikler ve pozisyon (Pace) hesaplaniyor...")

team_game_agg = df_raw.groupby(["game_id", "game_date", "season", "team_id", "opponent_id", "is_home"]).agg(
    team_pts=("pts", "sum"),
    team_fga=("fga", "sum"),
    team_fta=("fta", "sum"),
    team_orb=("orb", "sum"),
    team_drb=("drb", "sum"),
    team_trb=("trb", "sum"),
    team_ast=("ast", "sum"),
    team_stl=("stl", "sum"),
    team_blk=("blk", "sum"),
    team_tov=("tov", "sum"),
    team_fg3=("fg3", "sum"),
    team_fg3a=("fg3a", "sum"),
    team_total_mins=("minutes", "sum")
).reset_index()

# Dean Oliver Formülü ile Takım Pozisyon Sayısı: FGA - ORB + TOV + (0.44 * FTA)
team_game_agg["team_possessions"] = (
    team_game_agg["team_fga"] 
    - team_game_agg["team_orb"] 
    + team_game_agg["team_tov"] 
    + (0.44 * team_game_agg["team_fta"])
)

# Aynı maçtaki iki takımın kayıtlarını birleştirerek karşılıklı maç temposunu (Pace) ve DRtg'yi bul
game_pairs = team_game_agg.merge(
    team_game_agg,
    on="game_id",
    suffixes=("", "_opp")
)
# Sadece karşılıklı eşleşenleri tut
game_pairs = game_pairs[game_pairs["team_id"] != game_pairs["team_id_opp"]].copy()

# Maçın Gerçek Temposu = 48 dakika başına düşen ortalama pozisyon sayısı
game_pairs["actual_game_pace"] = (
    48.0 * (game_pairs["team_possessions"] + game_pairs["team_possessions_opp"]) 
    / (2.0 * (game_pairs["team_total_mins"] / 5.0))
)

# 100 Pozisyon Başına Hücum ve Savunma Verimliliği (ORtg & DRtg)
game_pairs["actual_ortg"] = (game_pairs["team_pts"] / (game_pairs["team_possessions"] + 1e-5)) * 100.0
game_pairs["actual_drtg"] = (game_pairs["team_pts_opp"] / (game_pairs["team_possessions_opp"] + 1e-5)) * 100.0
game_pairs["actual_point_diff"] = game_pairs["team_pts"] - game_pairs["team_pts_opp"]
game_pairs["actual_total_pts"] = game_pairs["team_pts"] + game_pairs["team_pts_opp"]

# ---------------------------------------------------------
# 4. Takım Seviyesi Hareketli Ortalama (Rolling Features)
# ---------------------------------------------------------
print("4. Veri sizintisiz (leakage-free) takim rolling metrikleri uretiliyor...")

# Kronolojik sıralama
game_pairs = game_pairs.sort_values(by=["team_id", "game_date"]).reset_index(drop=True)

# Shift(1) kuralı: O maçın kendi istatistiği kesinlikle hesaplamaya katılamaz!
grouped_team = game_pairs.groupby("team_id")

# Dinlenme Günü (Rest Days)
game_pairs["prev_game_date"] = grouped_team["game_date"].shift(1)
game_pairs["rest_days"] = (game_pairs["game_date"] - game_pairs["prev_game_date"]).dt.days.fillna(3)
game_pairs["rest_days"] = game_pairs["rest_days"].clip(upper=7)

# Son 5 ve 10 Maçlık Form (Pace, ORtg, DRtg, Pts, 3PT%)
for window in [5, 10]:
    game_pairs[f"roll_pace_{window}"] = grouped_team["actual_game_pace"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    game_pairs[f"roll_ortg_{window}"] = grouped_team["actual_ortg"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    game_pairs[f"roll_drtg_{window}"] = grouped_team["actual_drtg"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    game_pairs[f"roll_pts_{window}"] = grouped_team["team_pts"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    game_pairs[f"roll_pts_opp_{window}"] = grouped_team["team_pts_opp"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )

# Rakip Takımın Maç Öncesi Formunu Ekleme (Matchup Feature Join)
team_feature_cols = [
    "game_id", "team_id", "opponent_id", "game_date", "season", "is_home",
    "rest_days", "roll_pace_5", "roll_pace_10", "roll_ortg_5", "roll_ortg_10",
    "roll_drtg_5", "roll_drtg_10", "roll_pts_5", "roll_pts_10", "roll_pts_opp_5", "roll_pts_opp_10",
    # Hedef Değişkenler (Targets):
    "team_pts", "team_pts_opp", "actual_total_pts", "actual_point_diff", "actual_game_pace"
]

df_team_features = game_pairs[team_feature_cols].copy()

# Rakip formunu da satıra ekle
df_team_features = df_team_features.merge(
    df_team_features[["game_id", "team_id", "rest_days", "roll_pace_10", "roll_ortg_10", "roll_drtg_10", "roll_pts_10"]],
    left_on=["game_id", "opponent_id"],
    right_on=["game_id", "team_id"],
    suffixes=("", "_opp_pregame")
).drop(columns=["team_id_opp_pregame"])

# İlk birkaç maçlık eksik verileri temizle
df_team_features = df_team_features.dropna(subset=["roll_pace_5", "roll_ortg_5", "roll_drtg_5"]).reset_index(drop=True)
df_team_features.to_parquet("data/team_features.parquet", index=False)
print(f"   -> [TAMAMLANDI] 'data/team_features.parquet' kaydedildi ({len(df_team_features):,} takim maci).")

# ---------------------------------------------------------
# 5. Oyuncu Seviyesi Rolling Features
# ---------------------------------------------------------
print("5. Oyuncu seviyesi rolling form ve dakika basina uretim hesaplaniyor...")

df_raw = df_raw.sort_values(by=["player_id", "game_date"]).reset_index(drop=True)
grouped_player = df_raw.groupby("player_id")

# Oyuncunun Dinlenme Günü
df_raw["prev_game_date"] = grouped_player["game_date"].shift(1)
df_raw["player_rest_days"] = (df_raw["game_date"] - df_raw["prev_game_date"]).dt.days.fillna(3).clip(upper=7)

# Oyuncunun Son 5 ve 10 Maçlık Üretimi (Dakika Başına Hızlar)
for window in [5, 10]:
    df_raw[f"p_roll_min_{window}"] = grouped_player["minutes"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_pts_{window}"] = grouped_player["pts"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_trb_{window}"] = grouped_player["trb"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_ast_{window}"] = grouped_player["ast"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_fg3_{window}"] = grouped_player["fg3"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_stl_{window}"] = grouped_player["stl"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_blk_{window}"] = grouped_player["blk"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_usg_{window}"] = grouped_player["usg_pct"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df_raw[f"p_roll_ts_{window}"] = grouped_player["ts_pct"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

# Dakika Başına Hız Metrikleri (Per-Minute Rates)
df_raw["p_pts_per_min_10"] = df_raw["p_roll_pts_10"] / (df_raw["p_roll_min_10"] + 1e-4)
df_raw["p_trb_per_min_10"] = df_raw["p_roll_trb_10"] / (df_raw["p_roll_min_10"] + 1e-4)
df_raw["p_ast_per_min_10"] = df_raw["p_roll_ast_10"] / (df_raw["p_roll_min_10"] + 1e-4)

# Takım Maç Bilgileri ile Oyuncu Tablosunu Birleştir (Context Join)
df_player_features = df_raw.merge(
    df_team_features[[
        "game_id", "team_id", "roll_pace_10", "roll_ortg_10", "roll_drtg_10",
        "roll_pace_10_opp_pregame", "roll_drtg_10_opp_pregame", "actual_game_pace", "team_pts"
    ]],
    on=["game_id", "team_id"],
    how="inner"
)

# Anlamsız az süre almış çöpleri filtrele (En az 1 maç oynamış olanlar)
df_player_features = df_player_features.dropna(subset=["p_roll_min_5", "p_roll_pts_5"]).reset_index(drop=True)
df_player_features.to_parquet("data/player_features.parquet", index=False)
print(f"   -> [TAMAMLANDI] 'data/player_features.parquet' kaydedildi ({len(df_player_features):,} oyuncu satiri).")

print("\n" + "=" * 80)
print("  ADIM 1 BASARIYLA BITTI: Veri sizintisiz egitim setleri hazir!")
print("=" * 80)