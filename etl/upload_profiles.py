import os
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

# Otomatik şema algılama ve üzerine yazma ayarı
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    autodetect=True
)

# 1. Oyuncu Profillerini Yükle
if os.path.exists("dim_player_season_profiles.csv"):
    print("1. 'dim_player_season_profiles.csv' okunuyor...")
    df_players = pd.read_csv("dim_player_season_profiles.csv")
    p_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles"
    
    print(f"BigQuery '{p_ref}' tablosuna yukleniyor ({len(df_players)} satir)...")
    job = client.load_table_from_dataframe(df_players, p_ref, job_config=job_config)
    job.result()
    print(f"[BASARILI] '{p_ref}' tablosu olusturuldu ve veriler yuklendi!")
else:
    print("Hata: 'dim_player_season_profiles.csv' bulunamadi.")

# 2. Takım Profillerini Yükle
if os.path.exists("dim_team_season_profiles.csv"):
    print("\n2. 'dim_team_season_profiles.csv' okunuyor...")
    df_teams = pd.read_csv("dim_team_season_profiles.csv")
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_team_season_profiles"
    
    print(f"BigQuery '{t_ref}' tablosuna yukleniyor ({len(df_teams)} satir)...")
    job = client.load_table_from_dataframe(df_teams, t_ref, job_config=job_config)
    job.result()
    print(f"[BASARILI] '{t_ref}' tablosu olusturuldu ve veriler yuklendi!")
else:
    print("Hata: 'dim_team_season_profiles.csv' bulunamadi.")