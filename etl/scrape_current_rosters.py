import os
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

SEASON_YEAR = 2027  # 2026-27 Sezonu

# Basketball-Reference 30 Takım Kısaltmaları
BREF_TEAMS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

def calculate_age(birth_date_str, target_year=2026):
    """Doğum tarihinden (örn: 'August 22, 2001') sezon başı yaşını hesaplar."""
    if not birth_date_str:
        return 25
    try:
        b_date = datetime.strptime(birth_date_str.strip(), "%B %d, %Y")
        ref_date = datetime(target_year, 10, 1)  # Sezon başlangıcı Ekim ayı baz alınır
        age = ref_date.year - b_date.year - ((ref_date.month, ref_date.day) < (b_date.month, b_date.day))
        return int(age)
    except Exception:
        return 25

# ---------------------------------------------------------
# 1. Basketball-Reference Sayfalarını Kazı (30 Takım)
# ---------------------------------------------------------
print(f"1. Basketball-Reference uzerinden {SEASON_YEAR} sezonu 30 takim kadrosu cekiliyor...")
print("   (Rate-limit engeline takilmamak icin takimlar arasi 3.2 sn bekleniyor)\n")

roster_records = []

for idx, team_code in enumerate(BREF_TEAMS, 1):
    url = f"https://www.basketball-reference.com/teams/{team_code}/{SEASON_YEAR}.html"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        
        # Eğer henüz 2027 sayfası açılmamışsa fallback olarak 2026'ya bak
        if resp.status_code == 404:
            alt_url = f"https://www.basketball-reference.com/teams/{team_code}/{SEASON_YEAR - 1}.html"
            resp = requests.get(alt_url, headers=HEADERS, timeout=15)

        if resp.status_code != 200:
            print(f"   [{idx}/30] [UYARI] {team_code} alinamadi! HTTP: {resp.status_code}")
            time.sleep(3.2)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Roster tablosunu bul
        roster_table = soup.find("table", id="roster")
        if not roster_table:
            # Yorum satırı içine gizlenmiş tablolar için kontrol
            comments = soup.find_all(string=lambda text: isinstance(text, str) and 'id="roster"' in text)
            for comment in comments:
                c_soup = BeautifulSoup(comment, "html.parser")
                roster_table = c_soup.find("table", id="roster")
                if roster_table:
                    break

        if not roster_table:
            print(f"   [{idx}/30] [UYARI] {team_code} roster tablosu bulunamadi.")
            time.sleep(3.2)
            continue

        tbody = roster_table.find("tbody")
        if not tbody:
            time.sleep(3.2)
            continue

        team_count = 0
        for tr in tbody.find_all("tr"):
            # Oyuncu link hücresi (<td data-stat="player">)
            player_td = tr.find("td", {"data-stat": "player"})
            if not player_td:
                continue

            a_tag = player_td.find("a")
            if not a_tag or not a_tag.get("href"):
                continue

            raw_name = a_tag.get_text().strip()
            href = a_tag.get("href")  # Örn: /players/b/ballla01.html

            # URL içinden orijinal player_id'yi çıkar
            id_match = re.search(r"/players/[a-z]/([a-z0-9_]+)\.html", href)
            p_id = id_match.group(1) if id_match else ""

            if not p_id:
                continue

            # Forma Numarası (<th data-stat="number">)
            num_tag = tr.find(["th", "td"], {"data-stat": "number"})
            jersey = num_tag.get_text().strip() if num_tag else ""

            # Pozisyon (<td data-stat="pos">)
            pos_tag = tr.find("td", {"data-stat": "pos"})
            pos = pos_tag.get_text().strip() if pos_tag else "G"

            # Doğum Tarihi & Yaş (<td data-stat="birth_date">)
            bdate_tag = tr.find("td", {"data-stat": "birth_date"})
            bdate_str = bdate_tag.get_text().strip() if bdate_tag else ""
            age = calculate_age(bdate_str, target_year=SEASON_YEAR - 1)

            roster_records.append({
                "team_id": team_code,
                "player_id": p_id,
                "player_name": raw_name,
                "age": int(age),
                "position": str(pos),
                "jersey": str(jersey)
            })
            team_count += 1

        print(f"   [{idx:02d}/30] {team_code}: {team_count} oyuncu basariyla cekildi. (ID'ler dogrudan URL'den alindi)")
        time.sleep(3.2)  # B-Ref rate limit koruması

    except Exception as e:
        print(f"   [{idx}/30] [HATA] {team_code} cekilirken hata: {e}")
        time.sleep(3.2)

df_bref_roster = pd.DataFrame(roster_records).drop_duplicates(subset=["player_id", "team_id"]).reset_index(drop=True)

if df_bref_roster.empty:
    raise Exception("[HATA] Hicbir oyuncu verisi alinamadi!")

print(f"\n[BASARILI] Toplam {len(df_bref_roster)} oyuncu 30 takim icin Basketball-Reference'tan kazindi.")

# ---------------------------------------------------------
# 2. BigQuery `dim_current_rosters_2026_27` Tablosunu Güncelle
# ---------------------------------------------------------
print("\n2. BigQuery 'dim_current_rosters_2026_27' tablosu guncelleniyor...")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

table_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27"
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    autodetect=True
)

client.load_table_from_dataframe(df_bref_roster, table_ref, job_config=job_config).result()
print(f"[BASARILI] BigQuery '{table_ref}' tablosu Basketball-Reference kadrolari ile yenilendi!")

# Charlotte Hornets (CHO) Kontrol Çıktısı
print("\n" + "="*80)
print("  CHARLOTTE HORNETS (CHO) GUNCEL KADROSU (BASKETBALL-REFERENCE)")
print("="*80)
print(df_bref_roster[df_bref_roster["team_id"] == "CHO"][["player_name", "player_id", "position", "age", "jersey"]].to_string(index=False))