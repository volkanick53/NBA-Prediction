import json
import os
import random
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from google.cloud import bigquery
from google.oauth2 import service_account

# ---------------------------------------------------------
# GCP & BigQuery Configuration
# ---------------------------------------------------------
KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"  # Path to Service Account JSON key file
PROJECT_ID = "nba-analytics-503718"                 # GCP Project ID
DATASET_ID = "nba_analytics"
TABLE_ID = "fact_player_boxscores"

# Initialize BigQuery Client
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
bq_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

# Requests Header to prevent 403 Forbidden / Rate limit blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def safe_float(val):
    """Safely converts string/number values to float, handling percentage signs and NaNs."""
    if val is None or pd.isna(val) or val == "":
        return None
    try:
        val_str = str(val).replace("%", "").strip()
        return float(val_str)
    except ValueError:
        return None


def safe_int(val):
    """Safely converts string/number values to integer."""
    if val is None or pd.isna(val) or val == "":
        return None
    try:
        return int(float(str(val).strip()))
    except ValueError:
        return None


def parse_minutes(mp_str):
    """Converts minutes string '36:54' into float minutes (36.9)."""
    if not mp_str or pd.isna(mp_str):
        return 0.0
    mp_str = str(mp_str).strip()
    if ":" in mp_str:
        parts = mp_str.split(":")
        try:
            return round(int(parts[0]) + int(parts[1]) / 60.0, 1)
        except ValueError:
            return 0.0
    return safe_float(mp_str) or 0.0


def calculate_season(game_date_str):
    """Dynamically calculates NBA season string (e.g., '2022-23') based on game date."""
    year, month, _ = map(int, game_date_str.split("-"))
    if month >= 9:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def extract_game_boxscore(url):
    """
    Fetches game HTML source, unwraps commented HTML tables,
    and parses both basic and advanced player statistics.
    """
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 429:
        print(f"[RATE LIMIT] Server returned 429 for {url}. Sleeping 30 seconds...")
        time.sleep(30)
        response = requests.get(url, headers=HEADERS)
        
    if response.status_code != 200:
        print(f"[HTTP ERROR] Failed to fetch {url} (Status Code: {response.status_code})")
        return []

    # Unwrap HTML comments containing hidden tables
    soup = BeautifulSoup(response.content, "html.parser")
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            comment.replace_with(comment_soup)

    # Extract Game ID, Date, Season, and Teams
    game_id = url.split("/")[-1].replace(".html", "")
    game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
    season_str = calculate_season(game_date_str)
    home_team = game_id[9:12]  # Format e.g., 202603120ORL -> Home Team: ORL

    # Find basic and advanced player tables
    basic_tables = soup.find_all("table", id=re.compile(r"box-[A-Z]{3}-game-basic"))
    players_data = {}

    for table in basic_tables:
        table_id = table.get("id", "")
        team_id = table_id.split("-")[1]
        is_home = (team_id == home_team)
        
        # Identify opponent team code
        opp_tables = [t.get("id", "").split("-")[1] for t in basic_tables if t.get("id", "").split("-")[1] != team_id]
        opponent_id = opp_tables[0] if opp_tables else None

        rows = table.find("tbody").find_all("tr")
        for row in rows:
            if "thead" in row.get("class", []) or row.find("th", {"data-stat": "reason"}):
                continue  # Skip header repetitions or Did Not Play rows

            player_th = row.find("th", {"data-stat": "player"})
            if not player_th:
                continue

            player_name = player_th.text.strip()
            player_link = player_th.find("a")
            if not player_link or "href" not in player_link.attrs:
                continue  # Skip non-player summary rows

            # Extract player_id from link: /players/c/coulibi01.html -> coulibi01
            player_id = player_link["href"].split("/")[-1].replace(".html", "")

            def get_stat(stat_name):
                td = row.find("td", {"data-stat": stat_name})
                return td.text.strip() if td else None

            mp_str = get_stat("mp")
            if not mp_str:
                continue

            players_data[player_id] = {
                "game_id": game_id,
                "game_date": game_date_str,
                "season": season_str,
                "season_type": "Regular Season",
                "player_id": player_id,
                "player_name": player_name,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "is_home": is_home,
                "minutes": parse_minutes(mp_str),
                "fg": safe_int(get_stat("fg")),
                "fga": safe_int(get_stat("fga")),
                "fg_pct": safe_float(get_stat("fg_pct")),
                "fg3": safe_int(get_stat("fg3")),
                "fg3a": safe_int(get_stat("fg3a")),
                "fg3_pct": safe_float(get_stat("fg3_pct")),
                "ft": safe_int(get_stat("ft")),
                "fta": safe_int(get_stat("fta")),
                "ft_pct": safe_float(get_stat("ft_pct")),
                "orb": safe_int(get_stat("orb")),
                "drb": safe_int(get_stat("drb")),
                "trb": safe_int(get_stat("trb")),
                "ast": safe_int(get_stat("ast")),
                "stl": safe_int(get_stat("stl")),
                "blk": safe_int(get_stat("blk")),
                "tov": safe_int(get_stat("tov")),
                "pf": safe_int(get_stat("pf")),
                "pts": safe_int(get_stat("pts")),
                "plus_minus": safe_int(get_stat("plus_minus")),
                # Placeholders for Advanced Stats
                "ts_pct": None,
                "efg_pct": None,
                "fg3a_per_fga_pct": None,
                "fta_per_fga_pct": None,
                "orb_pct": None,
                "drb_pct": None,
                "trb_pct": None,
                "ast_pct": None,
                "stl_pct": None,
                "blk_pct": None,
                "tov_pct": None,
                "usg_pct": None,
                "off_rtg": None,
                "def_rtg": None,
                "bpm": None
            }

    # Extract advanced statistics table if available
    adv_tables = soup.find_all("table", id=re.compile(r"box-[A-Z]{3}-game-advanced"))
    for table in adv_tables:
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            if "thead" in row.get("class", []):
                continue
            player_th = row.find("th", {"data-stat": "player"})
            if not player_th or not player_th.find("a"):
                continue

            player_id = player_th.find("a")["href"].split("/")[-1].replace(".html", "")
            
            if player_id in players_data:
                def get_adv_stat(stat_name):
                    td = row.find("td", {"data-stat": stat_name})
                    return td.text.strip() if td else None

                players_data[player_id]["ts_pct"] = safe_float(get_adv_stat("ts_pct"))
                players_data[player_id]["efg_pct"] = safe_float(get_adv_stat("efg_pct"))
                players_data[player_id]["fg3a_per_fga_pct"] = safe_float(get_adv_stat("fg3a_per_fga_pct"))
                players_data[player_id]["fta_per_fga_pct"] = safe_float(get_adv_stat("fta_per_fga_pct"))
                players_data[player_id]["orb_pct"] = safe_float(get_adv_stat("orb_pct"))
                players_data[player_id]["drb_pct"] = safe_float(get_adv_stat("drb_pct"))
                players_data[player_id]["trb_pct"] = safe_float(get_adv_stat("trb_pct"))
                players_data[player_id]["ast_pct"] = safe_float(get_adv_stat("ast_pct"))
                players_data[player_id]["stl_pct"] = safe_float(get_adv_stat("stl_pct"))
                players_data[player_id]["blk_pct"] = safe_float(get_adv_stat("blk_pct"))
                players_data[player_id]["tov_pct"] = safe_float(get_adv_stat("tov_pct"))
                players_data[player_id]["usg_pct"] = safe_float(get_adv_stat("usg_pct"))
                players_data[player_id]["off_rtg"] = safe_float(get_adv_stat("off_rtg"))
                players_data[player_id]["def_rtg"] = safe_float(get_adv_stat("def_rtg"))
                players_data[player_id]["bpm"] = safe_float(get_adv_stat("bpm"))

    return list(players_data.values())


def upload_to_bigquery(records):
    """Loads cleaned player boxscore dictionary list into BigQuery table."""
    if not records:
        return

    df = pd.DataFrame(records)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    destination_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

    job = bq_client.load_table_from_dataframe(df, destination_table, job_config=job_config)
    job.result()
    print(f"[SUCCESS] Game {records[0]['game_id']} ({records[0]['season']}): Uploaded {len(df)} full player records to BigQuery.")


# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
# if __name__ == "__main__":
#     json_file = "all_boxscore_urls.json"

#     if not os.path.exists(json_file):
#         print(f"[ERROR] '{json_file}' not found. Please run generate_all_urls.py first.")
#         exit(1)

#     with open(json_file, "r") as f:
#         urls = json.load(f)

#     print(f"Starting Multi-Season HTML Table Scraper Pipeline... Total Games: {len(urls)}")

#     for idx, url in enumerate(urls, 1):
#         print(f"[{idx}/{len(urls)}] Processing: {url}")
#         try:
#             records = extract_game_boxscore(url)
#             upload_to_bigquery(records)
#         except Exception as e:
#             print(f"[ERROR] Failed to process {url}: {e}")

#         # Throttling delay to obey Basketball-Reference rate limits (3 to 4 seconds)
#         time.sleep(random.uniform(3.0, 4.2))

#     print("Pipeline completed successfully.")


# scrape_and_load.py içindeki ilgili alt kısım:
if __name__ == "__main__":
    # Tüm liste yerine sadece eksiklerin olduğu JSON'ı okutuyoruz
    json_file = "missing_boxscore_urls.json" 

    if not os.path.exists(json_file):
        print(f"[ERROR] '{json_file}' bulunamadı. Lütfen önce find_missing_urls.py çalıştırın.")
        exit(1)

    with open(json_file, "r") as f:
        urls = json.load(f)

    print(f"Starting Delta/Backfill Scraper Pipeline... Total Missing Games to Fetch: {len(urls)}")

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        try:
            records = extract_game_boxscore(url)
            upload_to_bigquery(records)
        except Exception as e:
            print(f"[ERROR] Failed to process {url}: {e}")

        time.sleep(random.uniform(3.0, 4.2))

    print("Missing games pipeline completed successfully.")