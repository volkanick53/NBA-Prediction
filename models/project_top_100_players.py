import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ---------------------------------------------------------
# GCP BigQuery Bağlantısı
# ---------------------------------------------------------
KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("1. BigQuery'den oyuncuların sezonluk istatistikleri çekiliyor...")

query = """
WITH player_seasons AS (
    SELECT 
        player_id,
        player_name,
        team_id,
        season,
        COUNT(DISTINCT game_id) AS games_played,
        SUM(minutes) AS total_minutes,
        ROUND(AVG(minutes), 1) AS mpg,
        ROUND(AVG(pts), 1) AS ppg,
        ROUND(AVG(trb), 1) AS rpg,
        ROUND(AVG(ast), 1) AS apg,
        ROUND(AVG(stl), 1) AS spg,
        ROUND(AVG(blk), 1) AS bpg,
        ROUND(AVG(fg3), 1) AS tpm_pg,
        ROUND(AVG(tov), 1) AS topg,
        ROUND(AVG(usg_pct), 1) AS usg_pct,
        ROUND(AVG(ts_pct), 3) AS ts_pct
    FROM `nba-analytics-503718.nba_analytics.fact_player_boxscores`
    WHERE minutes > 0
    GROUP BY player_id, player_name, team_id, season
)
SELECT * FROM player_seasons
"""

df = client.query(query).to_dataframe()

# ---------------------------------------------------------
# 2. 2025-26 Sezonunda En Çok Süre Alan Top 100 Oyuncuyu Belirle
# ---------------------------------------------------------
last_season_df = df[df["season"] == "2025-26"].copy()
top_100_players = (
    last_season_df.sort_values(by="total_minutes", ascending=False)
    .head(100)["player_id"]
    .tolist()
)

print(f"-> Top 100 oyuncu belirlendi. (Toplam veri satırı: {len(df)})")

# ---------------------------------------------------------
# 3. Ağırlıklı Projeksiyon Algoritması (2026-27 Sezonu İçin)
# ---------------------------------------------------------
weights = {"2025-26": 0.55, "2024-25": 0.30, "2023-24": 0.15}
metrics = [
    "mpg",
    "ppg",
    "rpg",
    "apg",
    "spg",
    "bpg",
    "tpm_pg",
    "topg",
    "usg_pct",
    "ts_pct",
]

projections = []

for pid in top_100_players:
    p_data = df[df["player_id"] == pid].copy()
    p_name = p_data["player_name"].iloc[0]
    p_team = p_data[p_data["season"] == "2025-26"]["team_id"].iloc[0]

    player_proj = {
        "player_id": pid,
        "player_name": p_name,
        "team_id": p_team,
    }

    # Sezon ağırlıklı ortalamaları hesapla
    for m in metrics:
        weighted_sum = 0
        total_weight = 0
        for season, w in weights.items():
            s_val = p_data[p_data["season"] == season]
            if not s_val.empty and not pd.isna(s_val[m].values[0]):
                weighted_sum += s_val[m].values[0] * w
                total_weight += w

        if total_weight > 0:
            final_val = weighted_sum / total_weight
        else:
            final_val = p_data[m].mean()

        player_proj[m] = round(final_val, 1 if m != "ts_pct" else 3)

    projections.append(player_proj)

proj_df = pd.DataFrame(projections)

# Kolon isimlerini temiz ve anlaşılır yapalım
proj_df.rename(
    columns={
        "mpg": "Proj_MIN",
        "ppg": "Proj_PTS",
        "rpg": "Proj_REB",
        "apg": "Proj_AST",
        "spg": "Proj_STL",
        "bpg": "Proj_BLK",
        "tpm_pg": "Proj_3PM",
        "topg": "Proj_TOV",
        "usg_pct": "Proj_USG%",
        "ts_pct": "Proj_TS%",
    },
    inplace=True,
)

# Skor beklentisine göre sırala
proj_df.sort_values(by="Proj_PTS", ascending=False, inplace=True)
proj_df.reset_index(drop=True, inplace=True)
proj_df.index += 1  # 1'den 100'e sıralama

# ---------------------------------------------------------
# 4. Sonuçları Kaydet ve Önizleme Göster
# ---------------------------------------------------------
output_file = "nba_2026_27_top100_projections.csv"
proj_df.to_csv(output_file, index_label="Rank")
print(f"\n[BAŞARILI] 2026-27 Top 100 Projeksiyonu '{output_file}' olarak kaydedildi!\n")

print("=== 2026-27 EN YÜKSEK SAYI BEKLENTİSİ OLAN İLK 15 YILDIZ ===")
print(
    proj_df[
        [
            "player_name",
            "team_id",
            "Proj_MIN",
            "Proj_PTS",
            "Proj_REB",
            "Proj_AST",
            "Proj_3PM",
            "Proj_TS%",
        ]
    ]
    .head(15)
    .to_string()
)