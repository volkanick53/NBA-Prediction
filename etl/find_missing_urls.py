import json
from google.cloud import bigquery
from google.oauth2 import service_account

# BigQuery Bağlantı Ayarları
KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("1. BigQuery'deki mevcut maçlar sorgulanıyor...")
query = "SELECT DISTINCT game_id FROM `nba-analytics-503718.nba_analytics.fact_player_boxscores`"
query_job = client.query(query)
existing_games = set(row.game_id for row in query_job.result())

print(f"-> BigQuery'de halihazırda var olan benzersiz maç sayısı: {len(existing_games)}")

print("2. 'all_boxscore_urls.json' okunuyor...")
with open("all_boxscore_urls.json", "r") as f:
    all_urls = json.load(f)

# Eksik olan URL'leri filtrele
missing_urls = []
for url in all_urls:
    game_id = url.split("/")[-1].replace(".html", "")
    if game_id not in existing_games:
        missing_urls.append(url)

print(f"\n[ÖZET]")
print(f"Hedeflenen Toplam Maç : {len(all_urls)}")
print(f"Veritabanındaki Maç   : {len(existing_games)}")
print(f"Eksik Maç Sayısı      : {len(missing_urls)}")

if missing_urls:
    output_file = "missing_boxscore_urls.json"
    with open(output_file, "w") as f:
        json.dump(missing_urls, f, indent=4)
    print(f"\nEksik maç URL'leri '{output_file}' dosyasına kaydedildi!")
else:
    print("\nHarika! Hiç eksik maç yok, tüm veritabanı tam.")