import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("1. Takımların geçmiş sezon hücum/savunma reytingleri hesaplanıyor...")

query = """
WITH team_games AS (
    SELECT 
        season,
        home_team AS team_id,
        home_score AS pts_scored,
        away_score AS pts_allowed,
        home_off_rtg AS off_rtg,
        away_off_rtg AS def_rtg,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS is_win
    FROM `nba-analytics-503718.nba_analytics.fact_game_summaries`
    UNION ALL
    SELECT 
        season,
        away_team AS team_id,
        away_score AS pts_scored,
        home_score AS pts_allowed,
        away_off_rtg AS off_rtg,
        home_off_rtg AS def_rtg,
        CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS is_win
    FROM `nba-analytics-503718.nba_analytics.fact_game_summaries`
)
SELECT 
    season,
    team_id,
    COUNT(*) as games,
    SUM(is_win) as wins,
    ROUND(AVG(pts_scored), 1) as avg_pts,
    ROUND(AVG(pts_allowed), 1) as avg_opp_pts,
    ROUND(AVG(off_rtg) - AVG(def_rtg), 2) as net_rating
FROM team_games
WHERE season IN ('2023-24', '2024-25', '2025-26')
GROUP BY season, team_id
"""

df_teams = client.query(query).to_dataframe()

# NBA Konferans Haritası
CONFERENCES = {
    "BOS": "East",
    "NYK": "East",
    "MIL": "East",
    "CLE": "East",
    "IND": "East",
    "PHI": "East",
    "MIA": "East",
    "ORL": "East",
    "CHI": "East",
    "ATL": "East",
    "BRK": "East",
    "TOR": "East",
    "CHO": "East",
    "WAS": "East",
    "DET": "East",
    "OKC": "West",
    "DEN": "West",
    "MIN": "West",
    "LAC": "West",
    "DAL": "West",
    "PHO": "West",
    "NOP": "West",
    "LAL": "West",
    "SAC": "West",
    "GSW": "West",
    "HOU": "West",
    "UTA": "West",
    "MEM": "West",
    "SAS": "West",
    "POR": "West",
}

# ---------------------------------------------------------
# 2. Takım Net Rating Projeksiyonu
# ---------------------------------------------------------
team_projections = []
weights = {"2025-26": 0.55, "2024-25": 0.30, "2023-24": 0.15}

all_teams = list(CONFERENCES.keys())

for t in all_teams:
    t_data = df_teams[df_teams["team_id"] == t]
    if t_data.empty:
        continue

    weighted_net_rtg = 0
    total_w = 0
    for s, w in weights.items():
        val = t_data[t_data["season"] == s]
        if not val.empty:
            weighted_net_rtg += val["net_rating"].values[0] * w
            total_w += w

    proj_net = (
        (weighted_net_rtg / total_w) if total_w > 0 else t_data["net_rating"].mean()
    )

    # Ortalamaya hafif regresyon (%15 küçülme faktörü)
    proj_net = proj_net * 0.85

    team_projections.append(
        {
            "team_id": t,
            "conference": CONFERENCES.get(t, "East"),
            "proj_net_rating": round(proj_net, 2),
        }
    )

team_proj_df = pd.DataFrame(team_projections)

# ---------------------------------------------------------
# 3. Galibiyet Sayısı Hesabı (Pythagorean & Win Scaling)
# NBA'de her +1.0 Net Rating ortalama ~2.7 galibiyete karşılık gelir (Baz: 41 Win)
# ---------------------------------------------------------
team_proj_df["raw_wins"] = 41.0 + (team_proj_df["proj_net_rating"] * 2.7)

# Toplam lig galibiyetini tam 1230'a sabitleme (Sıfır Toplamlı Lig Dengesi)
scale_factor = 1230.0 / team_proj_df["raw_wins"].sum()
team_proj_df["Proj_Wins"] = (team_proj_df["raw_wins"] * scale_factor).round().astype(int)
team_proj_df["Proj_Losses"] = 82 - team_proj_df["Proj_Wins"]
team_proj_df["Record"] = (
    team_proj_df["Proj_Wins"].astype(str)
    + "-"
    + team_proj_df["Proj_Losses"].astype(str)
)

# ---------------------------------------------------------
# 4. Konferans Sıralamalarını Yazdır ve Kaydet
# ---------------------------------------------------------
team_proj_df.sort_values(by=["conference", "Proj_Wins"], ascending=[True, False], inplace=True)
team_proj_df.to_csv("nba_2026_27_team_standings_projection.csv", index=False)

print("\n[BAŞARILI] 2026-27 Takım Sıralamaları 'nba_2026_27_team_standings_projection.csv' dosyasına kaydedildi!\n")

for conf in ["East", "West"]:
    print(f"\n{'='*20} {conf.upper()}ERN CONFERENCE STANDINGS PROJECTION {'='*20}")
    conf_table = team_proj_df[team_proj_df["conference"] == conf].copy()
    conf_table.reset_index(drop=True, inplace=True)
    conf_table.index += 1
    print(
        conf_table[["team_id", "Record", "Proj_Wins", "Proj_Losses", "proj_net_rating"]]
        .to_string()
    )