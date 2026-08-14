import json
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

print("1. Querying completed matches involving injuries/DNPs in BigQuery...")
query = "SELECT DISTINCT game_id FROM `nba-analytics-503718.nba_analytics.fact_player_absences`"
existing_games = set(row.game_id for row in client.query(query).result())

print(f"-> Number of unique matches already existing in BigQuery: {len(existing_games)}")

with open("all_boxscore_urls.json", "r") as f:
    all_urls = json.load(f)

remaining_urls = []
for url in all_urls:
    game_id = url.split("/")[-1].replace(".html", "")
    if game_id not in existing_games:
        remaining_urls.append(url)


print(f"Total Matches      : {len(all_urls)}")
print(f"Completed Matches  : {len(existing_games)}")
print(f"Remaining for Upload: {len(remaining_urls)}")

output_file = "remaining_absences_urls.json"
with open(output_file, "w") as f:
    json.dump(remaining_urls, f, indent=4)

print(f"\nRemaining matches '{output_file}' file saved!")