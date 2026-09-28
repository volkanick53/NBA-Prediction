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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

TEAMS = [
    "ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
]

SEASONS = [2024, 2025, 2026]

# Belirttiğin 7 DIV ID ve Sütun Çakışmalarını Önleyen Önekler
DIV_TABLE_CONFIGS = [
    {"div_id": "div_per_game_stats", "prefix": ""},
    {"div_id": "div_per_minute_stats", "prefix": "p36_"},
    {"div_id": "div_per_poss", "prefix": "p100_"},
    {"div_id": "div_advanced", "prefix": "adv_"},
    {"div_id": "div_shooting", "prefix": "sht_"},
    {"div_id": "div_adj_shooting", "prefix": "adj_"},
    {"div_id": "div_pbp_stats", "prefix": "pbp_"}
]

def clean_col_name(name):
    """BigQuery standartlarına uygun geçerli sütun ismi üretir."""
    name = str(name).strip().lower()
    name = name.replace("%", "_pct").replace("+", "_plus").replace("/", "_per_").replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

def fetch_html_uncommented(url, max_retries=4):
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            resp = session.get(url, headers=HEADERS, timeout=(10, 35))
            if resp.status_code == 200:
                # Tüm HTML yorum taglerini kaldırarak gizli tabloları doğrudan DOM'a açıyoruz
                raw_html = resp.text.replace("<!--", "").replace("-->", "")
                return raw_html
            elif resp.status_code == 429:
                print(f"   [!] Rate limit (429). 30 sn bekleniyor... (Deneme {attempt}/{max_retries})")
                time.sleep(30)
            elif resp.status_code == 404:
                return None
            else:
                time.sleep(4)
        except Exception:
            time.sleep(attempt * 5)
    return None

def parse_div_table(soup, div_id, prefix=""):
    div_elem = soup.find("div", {"id": div_id})
    if not div_elem:
        return pd.DataFrame()

    table_elem = div_elem.find("table")
    if not table_elem or not table_elem.find("tbody"):
        return pd.DataFrame()

    rows = []
    for tr in table_elem.find("tbody").find_all("tr"):
        classes = tr.get("class", [])
        if any(cls in ["thead", "over_header", "sub_header"] for cls in classes):
            continue

        # Oyuncu ID ve İsim Tespiti
        th_player = tr.find(["th", "td"], {"data-stat": "player"}) or tr.find(["th", "td"], {"data-stat": "player_name"})
        player_id = ""
        player_name = ""

        if th_player:
            player_name = th_player.text.strip()
            player_id = th_player.get("data-append-csv", "")
            if not player_id:
                link = th_player.find("a")
                if link and "href" in link.attrs:
                    m = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", link["href"])
                    if m:
                        player_id = m.group(1)

        if not player_id:
            # Satırdaki herhangi bir linkten oyuncu ID'si bulmayı dene
            link = tr.find("a", href=re.compile(r"/players/[a-z]/[a-z0-9]+\.html"))
            if link:
                m = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", link["href"])
                if m:
                    player_id = m.group(1)
                    if not player_name:
                        player_name = link.text.strip()

        if not player_id:
            continue

        row_dict = {"player_id": player_id, "player_name": player_name}

        for td in tr.find_all("td"):
            stat = td.get("data-stat")
            if stat and stat not in ["player", "player_name"]:
                col_name = f"{prefix}{stat}" if prefix and stat not in ["age", "g", "gs", "mp", "pos"] else stat
                row_dict[col_name] = td.text.strip()

        rows.append(row_dict)

    return pd.DataFrame(rows)

def scrape_team_tables(team_code, year):
    url = f"https://www.basketball-reference.com/teams/{team_code}/{year}.html"
    season_code = f"{year-1}-{str(year)[2:]}"

    html = fetch_html_uncommented(url)
    if not html:
        print(f"[{season_code} - {team_code}] Sayfa alinamadi (404 veya Baglanti Hatasi)")
        return []

    soup = BeautifulSoup(html, "html.parser")
    merged_df = None
    tables_found = 0

    for cfg in DIV_TABLE_CONFIGS:
        d_id = cfg["div_id"]
        pref = cfg["prefix"]

        df_tbl = parse_div_table(soup, d_id, prefix=pref)
        if not df_tbl.empty:
            tables_found += 1
            if merged_df is None:
                merged_df = df_tbl
            else:
                overlap = [c for c in df_tbl.columns if c in merged_df.columns and c not in ["player_id", "player_name"]]
                df_to_merge = df_tbl.drop(columns=overlap)
                merged_df = merged_df.merge(df_to_merge, on="player_id", how="outer")
                if "player_name_y" in merged_df.columns:
                    merged_df["player_name"] = merged_df["player_name_x"].combine_first(merged_df["player_name_y"])
                    merged_df = merged_df.drop(columns=["player_name_x", "player_name_y"])

    if merged_df is None or merged_df.empty:
        print(f"[{season_code} - {team_code}] Tablo bulunamadi.")
        return []

    merged_df["team_id"] = team_code
    merged_df["season"] = season_code
    print(f"[{season_code} - {team_code}] OK ({tables_found}/7 Div Tablosu, {len(merged_df)} Oyuncu)")
    return merged_df.to_dict("records")

# ---------------------------------------------------------
# Tüm Takımları ve Sezonları Topla
# ---------------------------------------------------------
all_records = []
for yr in SEASONS:
    for tm in TEAMS:
        recs = scrape_team_tables(tm, yr)
        all_records.extend(recs)
        time.sleep(3.2)  # BRef rate limit koruması

df_all = pd.DataFrame(all_records)

if df_all.empty:
    print("\n[HATA] Hicbir veri cekilemedi!")
    exit(1)

# Kolon isimlerini BigQuery standartlarına temizle
df_all.columns = [clean_col_name(c) for c in df_all.columns]

# İsim Eşlemeleri
col_rename = {
    "g": "games",
    "gs": "games_started",
    "mp_per_g": "mp_per_g",
    "pts_per_g": "pts_per_g",
    "trb_per_g": "trb_per_g",
    "ast_per_g": "ast_per_g",
    "stl_per_g": "stl_per_g",
    "blk_per_g": "blk_per_g",
    "tov_per_g": "tov_per_g",
    "adv_bpm": "bpm",
    "adv_vorp": "vorp",
    "adv_ws_per_48": "ws_per_48",
    "adv_ws": "ws",
    "adv_usg_pct": "usg_pct",
    "adv_ts_pct": "ts_pct",
    "adv_per": "per"
}
df_all = df_all.rename(columns={k: v for k, v in col_rename.items() if k in df_all.columns})

# Sayısal Değer Dönüşümleri
for c in df_all.columns:
    if c not in ["player_id", "player_name", "team_id", "season", "pos"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

# Sezonluk Toplam Dakika (mp = games * mp_per_g)
if "games" in df_all.columns and "mp_per_g" in df_all.columns:
    df_all["mp"] = (df_all["games"].fillna(0) * df_all["mp_per_g"].fillna(0)).round(1)

df_all.to_csv("dim_player_season_profiles.csv", index=False)
print("\n" + "="*80)
print(f" [BASARILI] Toplam {len(df_all)} oyuncu-sezon kaydi ve {len(df_all.columns)} sutun alindi.")
print("="*80)

# BigQuery Güncelle
if os.path.exists(KEY_PATH) and not df_all.empty:
    print("BigQuery 'dim_player_season_profiles' tablosu guncelleniyor...")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles"
    client.load_table_from_dataframe(df_all, t_ref, job_config=job_config).result()
    print(f"[BASARILI] BigQuery '{t_ref}' tablosu 7 div tablosunun tum metrikleriyle guncellendi!")