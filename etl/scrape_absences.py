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
KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"
TABLE_ID = "fact_player_absences"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
bq_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def calculate_season(game_date_str):
    year, month, _ = map(int, game_date_str.split("-"))
    if month >= 9:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"


def extract_absences(url):
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 429:
        print(f"[RATE LIMIT] 429 received for {url}. Sleeping 30s...")
        time.sleep(30)
        response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"[HTTP ERROR] Failed to fetch {url} (Status Code: {response.status_code})")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Unwrap HTML comments (Inactives tablosu bazen comment icindedir)
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "inactives" in comment or "<table" in comment:
            comment_soup = BeautifulSoup(comment, "html.parser")
            comment.replace_with(comment_soup)

    game_id = url.split("/")[-1].replace(".html", "")
    game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
    season_str = calculate_season(game_date_str)

    absences = []

    # ---------------------------------------------------------
    # 1. DNP / DND (Did Not Play / Did Not Dress) - From Boxscore Tables
    # ---------------------------------------------------------
    basic_tables = soup.find_all("table", id=re.compile(r"box-[A-Z]{3}-game-basic"))
    for table in basic_tables:
        team_id = table.get("id", "").split("-")[1]
        rows = table.find("tbody").find_all("tr")

        for row in rows:
            reason_td = row.find("td", {"data-stat": "reason"})
            if not reason_td:
                continue

            player_th = row.find("th", {"data-stat": "player"})
            if not player_th:
                continue

            player_name = player_th.text.strip()
            player_link = player_th.find("a")

            player_id = None
            if player_link and "href" in player_link.attrs:
                player_id = player_link["href"].split("/")[-1].replace(".html", "")

            reason_text = reason_td.text.strip()

            absences.append({
                "game_id": game_id,
                "game_date": game_date_str,
                "season": season_str,
                "team_id": team_id,
                "player_id": player_id,
                "player_name": player_name,
                "status": "DNP",
                "reason": reason_text if reason_text else "Did Not Play"
            })

    # ---------------------------------------------------------
    # 2. INACTIVE Players - From Inactives Section
    # ---------------------------------------------------------
    
    inactives_div = soup.find("div", id="all_inactives") or soup.find("div", id="inactives")
    
    if inactives_div:
       
        text_content = inactives_div.get_text()
        
        # Check if the text contains "Inactive" or "Inactives"
        player_links = inactives_div.find_all("a")
        for link in player_links:
            href = link.get("href", "")
            if "/players/" in href:
                player_id = href.split("/")[-1].replace(".html", "")
                player_name = link.text.strip()
                
               
                parent_text = link.parent.text if link.parent else ""
                team_id = None
                for t in ["ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
                          "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
                          "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]:
                    if t in parent_text:
                        team_id = t
                        break

                absences.append({
                    "game_id": game_id,
                    "game_date": game_date_str,
                    "season": season_str,
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": player_name,
                    "status": "INACTIVE",
                    "reason": "Inactive / Injury / G-League"
                })

    return absences


def upload_to_bigquery(records):
    if not records:
        return

    df = pd.DataFrame(records)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    destination_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")

    job = bq_client.load_table_from_dataframe(df, destination_table, job_config=job_config)
    job.result()
    print(f"[SUCCESS] Game {records[0]['game_id']}: Uploaded {len(df)} absence records.")


if __name__ == "__main__":
    json_file = "remaining_absences_urls.json"

    if not os.path.exists(json_file):
        print(f"[ERROR] '{json_file}' not found.")
        exit(1)

    with open(json_file, "r") as f:
        urls = json.load(f)

    print(f"Starting Cloud Absence Scraper Pipeline... Total Remaining Games: {len(urls)}")

    for idx, url in enumerate(urls, 1):
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        try:
            records = extract_absences(url)
            upload_to_bigquery(records)
        except Exception as e:
            print(f"[ERROR] Failed to process {url}: {e}")

        time.sleep(random.uniform(3.0, 4.2))

    print("Absence pipeline completed successfully.")