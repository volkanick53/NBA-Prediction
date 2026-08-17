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

SEASONS = [
    (2022, "2021-22"),
    (2023, "2022-23"),
    (2024, "2023-24"),
    (2025, "2024-25"),
    (2026, "2025-26")
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_val(val):
    if val is None or str(val).strip() in ["", "-", "None"]:
        return None
    val_str = str(val).replace("%", "").replace("+", "").replace(",", "").strip()
    try:
        if "." in val_str:
            return float(val_str)
        return int(val_str)
    except:
        return str(val).strip()

def extract_table(html_text, table_ids):
    """HTML yorumlari icinde dahi olsa belirtilen ID'lerden ilk esleseni ceker."""
    for t_id in table_ids:
        pattern = rf'<table[^>]*id=["\']{t_id}["\'][\s\S]*?</table>'
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            return BeautifulSoup(match.group(0), "html.parser")
    return None

def parse_table_dynamic(table_soup):
    """Tablodaki tum data-stat kolonlarini dinamik olarak sozluk olarak yakalar."""
    records = {}
    if not table_soup:
        return records
    tbody = table_soup.find("tbody")
    if not tbody:
        return records

    for row in tbody.find_all("tr"):
        classes = row.get("class", [])
        if any(c in ["thead", "over_header"] for c in classes):
            continue

        player_cell = row.find(attrs={"data-append-csv": True})
        if not player_cell:
            player_cell = row.find(["th", "td"], {"data-stat": ["name_display", "player"]})

        if not player_cell:
            continue

        player_id = player_cell.get("data-append-csv")
        if not player_id and player_cell.find("a"):
            href = player_cell.find("a").get("href", "")
            player_id = href.split("/")[-1].replace(".html", "")

        if not player_id:
            continue

        row_data = {
            "player_id": player_id,
            "player_name": player_cell.text.strip()
        }

        for cell in row.find_all(["th", "td"]):
            stat_name = cell.get("data-stat")
            if stat_name and stat_name not in ["ranker", "awards"]:
                row_data[stat_name] = clean_val(cell.text.strip())

        records[player_id] = row_data
    return records

all_player_rows = []
all_team_rows = []
total_requests = len(SEASONS) * len(TEAMS)
counter = 0

print(f"=== NBA Veri Toplama Basliyor ({total_requests} Sayfa) ===\n")

for year, season_str in SEASONS:
    for team in TEAMS:
        counter += 1
        url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"[{counter}/{total_requests}] {season_str} - {team} -> HTTP {resp.status_code}")
                time.sleep(3.2)
                continue

            html = resp.text

            # 1. Bireysel Tablolari Ayrıştır
            t_adv = parse_table_dynamic(extract_table(html, ["advanced"]))
            t_p36 = parse_table_dynamic(extract_table(html, ["per_minute_stats", "per_minute"]))
            t_p100 = parse_table_dynamic(extract_table(html, ["per_poss"]))
            t_adj = parse_table_dynamic(extract_table(html, ["adj_shooting"]))
            t_shot = parse_table_dynamic(extract_table(html, ["shooting", "shooting_stats"]))
            t_pbp = parse_table_dynamic(extract_table(html, ["pbp", "advanced_pbp", "play-by-play"]))

            player_ids = set(t_adv.keys()) | set(t_p36.keys()) | set(t_p100.keys()) | set(t_adj.keys()) | set(t_shot.keys()) | set(t_pbp.keys())
            print(f"[{counter}/{total_requests}] {season_str} - {team} -> {len(player_ids)} oyuncu basariyla yakalandi.")

            for pid in player_ids:
                row = {
                    "player_id": pid,
                    "team_id": team,
                    "season": season_str
                }
                # Tablolari sirayla birlestir
                row.update(t_adv.get(pid, {}))
                row.update(t_p36.get(pid, {}))
                row.update(t_p100.get(pid, {}))
                row.update(t_adj.get(pid, {}))
                row.update(t_shot.get(pid, {}))
                row.update(t_pbp.get(pid, {}))

                all_player_rows.append(row)

            # 2. Takim Seviyesi (team_misc)
            t_misc = extract_table(html, ["team_misc"])
            if t_misc and t_misc.find("tbody"):
                m_row = t_misc.find("tbody").find("tr")
                if m_row:
                    team_dict = {"team_id": team, "season": season_str}
                    for cell in m_row.find_all(["th", "td"]):
                        st = cell.get("data-stat")
                        if st and st != "player":
                            team_dict[st] = clean_val(cell.text.strip())
                    all_team_rows.append(team_dict)

        except Exception as e:
            print(f"  -> Hata ({team} {season_str}): {e}")

        time.sleep(3.2)

# ---------------------------------------------------------
# CSV & BigQuery Yukleme
# ---------------------------------------------------------
df_players = pd.DataFrame(all_player_rows)
df_teams = pd.DataFrame(all_team_rows)

df_players.to_csv("dim_player_season_profiles.csv", index=False)
df_teams.to_csv("dim_team_season_profiles.csv", index=False)

print(f"\n[BASARILI] {len(df_players)} oyuncu profili ({len(df_players.columns)} kolon) ve {len(df_teams)} takim profili kaydedildi.")

if os.path.exists(KEY_PATH) and len(df_players) > 0:
    print("BigQuery tablolarina yukleniyor...")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )

    p_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles"
    client.load_table_from_dataframe(df_players, p_ref, job_config=job_config).result()
    print(f"[BASARILI] BigQuery '{p_ref}' tablosu guncellendi.")

    t_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_team_season_profiles"
    client.load_table_from_dataframe(df_teams, t_ref, job_config=job_config).result()
    print(f"[BASARILI] BigQuery '{t_ref}' tablosu guncellendi.")