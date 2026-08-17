import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

TEAMS = [
    "ATL", "BOS", "BRK", "CHI", "CHO", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def extract_contracts_table(html_text):
    pattern = r'<table[^>]*id=["\']contracts["\'][\s\S]*?</table>'
    match = re.search(pattern, html_text, re.IGNORECASE)
    if match:
        return BeautifulSoup(match.group(0), "html.parser")
    return None

def parse_active_roster(table_soup, team_code):
    roster = []
    seen_ids = set()
    if not table_soup or not table_soup.find("tbody"):
        return roster

    for row in table_soup.find("tbody").find_all("tr"):
        classes = row.get("class", [])
        if any(c in ["thead", "partial_table", "over_header"] for c in classes):
            continue

        th_cell = row.find("th", {"data-stat": "player"})
        if not th_cell or th_cell.find("em"):
            continue

        player_link = th_cell.find("a")
        if not player_link:
            continue

        player_name = player_link.text.strip()
        player_id = th_cell.get("csk")
        if not player_id:
            href = player_link.get("href", "")
            player_id = href.split("/")[-1].replace(".html", "")

        # Yaş verisini yakala (data-stat="age_today")
        age_cell = row.find("td", {"data-stat": "age_today"})
        age_val = 26
        if age_cell and age_cell.text.strip().isdigit():
            age_val = int(age_cell.text.strip())

        if player_id and player_id not in seen_ids:
            seen_ids.add(player_id)
            roster.append({
                "team_id": team_code,
                "player_id": player_id,
                "player_name": player_name,
                "age": age_val,
                "season": "2026-27"
            })

    return roster

all_roster_rows = []
total_teams = len(TEAMS)

print(f"=== 30 NBA Takiminin Guncel Kadrolari ve Yas Verileri Cekiliyor ===\n")

for idx, team in enumerate(TEAMS, 1):
    url = f"https://www.basketball-reference.com/contracts/{team}.html"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"[{idx}/{total_teams}] {team} -> HTTP {resp.status_code}")
            time.sleep(3.2)
            continue

        tbl_soup = extract_contracts_table(resp.text)
        team_players = parse_active_roster(tbl_soup, team)
        print(f"[{idx}/{total_teams}] {team} -> {len(team_players)} sozlesmeli oyuncu (yas bilgisiyle) alindi.")
        all_roster_rows.extend(team_players)

    except Exception as e:
        print(f"[{idx}/{total_teams}] {team} -> Hata: {e}")

    time.sleep(3.2)

df_rosters = pd.DataFrame(all_roster_rows)
df_rosters.to_csv("dim_current_rosters_2026_27.csv", index=False)
print(f"\n[BASARILI] Toplam {len(df_rosters)} oyuncu kaydedildi.")

if os.path.exists(KEY_PATH) and len(df_rosters) > 0:
    print("BigQuery 'dim_current_rosters_2026_27' tablosu guncelleniyor...")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27"
    client.load_table_from_dataframe(df_rosters, t_ref, job_config=job_config).result()
    print(f"[BASARILI] BigQuery '{t_ref}' tablosu guncellendi!")