import os
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

TEAM_NAME_TO_CODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

def parse_record(val_str):
    if not val_str or "-" not in str(val_str):
        return 0, 0, 0.0
    parts = str(val_str).strip().split("-")
    try:
        w, l = int(parts[0]), int(parts[1])
        pct = round(w / (w + l), 3) if (w + l) > 0 else 0.0
        return w, l, pct
    except Exception:
        return 0, 0, 0.0

def fetch_html_with_retry(url, max_retries=4):
    """SSL ve Timeout hatalarına karşı kademeli bekleme (backoff) ile istek atar."""
    for attempt in range(1, max_retries + 1):
        try:
            session = requests.Session()
            resp = session.get(url, headers=HEADERS, timeout=(10, 35))
            
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                print(f"   [!] Rate limit (429) alindi. 30 saniye bekleniyor... (Deneme {attempt}/{max_retries})")
                time.sleep(30)
            else:
                print(f"   [!] HTTP {resp.status_code} alindi. 5 saniye sonra tekrar deneniyor...")
                time.sleep(5)
        except (requests.exceptions.RequestException, Exception) as e:
            wait_sec = attempt * 6
            print(f"   [!] Baglanti/Timeout Hatasi ({e.__class__.__name__}). {wait_sec} sn bekleniyor... (Deneme {attempt}/{max_retries})")
            time.sleep(wait_sec)
            
    return None

def scrape_expanded_standings_for_season(season_year=2026):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season_year}_standings.html"
    print(f"[{season_year}] Expanded Standings verisi cekiliyor: {url}")

    html_content = fetch_html_with_retry(url)
    if not html_content:
        print(f"[{season_year}] Sayfa verisi alinamadi, atlaniyor.")
        return []

    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", {"id": "expanded_standings"})

    if not table:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if 'id="expanded_standings"' in c:
                c_soup = BeautifulSoup(c, "html.parser")
                table = c_soup.find("table", {"id": "expanded_standings"})
                if table:
                    break

    if not table or not table.find("tbody"):
        print(f"[{season_year}] 'expanded_standings' tablosu HTML icerisinde bulunamadi!")
        return []

    season_code = f"{season_year-1}-{str(season_year)[2:]}"
    rows = []

    for tr in table.find("tbody").find_all("tr"):
        classes = tr.get("class", [])
        if any(cls in ["thead", "over_header"] for cls in classes):
            continue

        th_rank = tr.find("th", {"data-stat": "ranker"})
        th_team = tr.find("td", {"data-stat": "team_name"}) or tr.find("th", {"data-stat": "team_name"})

        if not th_team:
            continue

        team_name = th_team.text.strip()
        rank_val = int(th_rank.text.strip()) if th_rank and th_rank.text.strip().isdigit() else None

        team_id = TEAM_NAME_TO_CODE.get(team_name, "")
        if not team_id:
            for k, v in TEAM_NAME_TO_CODE.items():
                if k in team_name:
                    team_id = v
                    break

        def extract_stat(stat_aliases):
            for alias in stat_aliases:
                td = tr.find("td", {"data-stat": alias})
                if td and td.text.strip():
                    return td.text.strip()
            return ""

        ov_w, ov_l, ov_pct = parse_record(extract_stat(["overall", "Overall"]))
        hm_w, hm_l, hm_pct = parse_record(extract_stat(["home", "Home"]))
        rd_w, rd_l, rd_pct = parse_record(extract_stat(["road", "Road"]))

        e_w, e_l, e_pct = parse_record(extract_stat(["conf_e", "e", "E", "east"]))
        w_w, w_l, w_pct = parse_record(extract_stat(["conf_w", "w", "W", "west"]))

        a_w, a_l, a_pct = parse_record(extract_stat(["div_a", "a", "A", "atlantic"]))
        c_w, c_l, c_pct = parse_record(extract_stat(["div_c", "c", "C", "central"]))
        se_w, se_l, se_pct = parse_record(extract_stat(["div_se", "se", "SE", "southeast"]))
        nw_w, nw_l, nw_pct = parse_record(extract_stat(["div_nw", "nw", "NW", "northwest"]))
        p_w, p_l, p_pct = parse_record(extract_stat(["div_p", "p", "P", "pacific"]))
        sw_w, sw_l, sw_pct = parse_record(extract_stat(["div_sw", "sw", "SW", "southwest"]))

        pre_w, pre_l, pre_pct = parse_record(extract_stat(["pre", "Pre", "pre_all_star"]))
        post_w, post_l, post_pct = parse_record(extract_stat(["post", "Post", "post_all_star"]))

        le3_w, le3_l, le3_pct = parse_record(extract_stat(["margin_le_3", "le_3", "Margin <=3", "margin_3_or_less"]))
        ge10_w, ge10_l, ge10_pct = parse_record(extract_stat(["margin_ge_10", "ge_10", "Margin >=10", "margin_10_or_more"]))

        oct_w, oct_l, oct_pct = parse_record(extract_stat(["oct", "Oct", "october"]))
        nov_w, nov_l, nov_pct = parse_record(extract_stat(["nov", "Nov", "november"]))
        dec_w, dec_l, dec_pct = parse_record(extract_stat(["dec", "Dec", "december"]))
        jan_w, jan_l, jan_pct = parse_record(extract_stat(["jan", "Jan", "january"]))
        feb_w, feb_l, feb_pct = parse_record(extract_stat(["feb", "Feb", "february"]))
        mar_w, mar_l, mar_pct = parse_record(extract_stat(["mar", "Mar", "march"]))
        apr_w, apr_l, apr_pct = parse_record(extract_stat(["apr", "Apr", "april"]))

        rows.append({
            "season": season_code,
            "team_id": team_id,
            "team_name": team_name,
            "rank": rank_val,
            "overall_w": ov_w, "overall_l": ov_l, "overall_pct": ov_pct,
            "home_w": hm_w, "home_l": hm_l, "home_pct": hm_pct,
            "road_w": rd_w, "road_l": rd_l, "road_pct": rd_pct,
            "vs_east_w": e_w, "vs_east_l": e_l, "vs_east_pct": e_pct,
            "vs_west_w": w_w, "vs_west_l": w_l, "vs_west_pct": w_pct,
            "vs_atlantic_w": a_w, "vs_atlantic_l": a_l, "vs_atlantic_pct": a_pct,
            "vs_central_w": c_w, "vs_central_l": c_l, "vs_central_pct": c_pct,
            "vs_southeast_w": se_w, "vs_southeast_l": se_l, "vs_southeast_pct": se_pct,
            "vs_northwest_w": nw_w, "vs_northwest_l": nw_l, "vs_northwest_pct": nw_pct,
            "vs_pacific_w": p_w, "vs_pacific_l": p_l, "vs_pacific_pct": p_pct,
            "vs_southwest_w": sw_w, "vs_southwest_l": sw_l, "vs_southwest_pct": sw_pct,
            "pre_as_w": pre_w, "pre_as_l": pre_l, "pre_as_pct": pre_pct,
            "post_as_w": post_w, "post_as_l": post_l, "post_as_pct": post_pct,
            "close_game_le3_w": le3_w, "close_game_le3_l": le3_l, "close_game_le3_pct": le3_pct,
            "blowout_ge10_w": ge10_w, "blowout_ge10_l": ge10_l, "blowout_ge10_pct": ge10_pct,
            "oct_w": oct_w, "oct_l": oct_l, "oct_pct": oct_pct,
            "nov_w": nov_w, "nov_l": nov_l, "nov_pct": nov_pct,
            "dec_w": dec_w, "dec_l": dec_l, "dec_pct": dec_pct,
            "jan_w": jan_w, "jan_l": jan_l, "jan_pct": jan_pct,
            "feb_w": feb_w, "feb_l": feb_l, "feb_pct": feb_pct,
            "mar_w": mar_w, "mar_l": mar_l, "mar_pct": mar_pct,
            "apr_w": apr_w, "apr_l": apr_l, "apr_pct": apr_pct,
            "empirical_hca_diff": round(hm_pct - rd_pct, 3),
            "clutch_luck_bias": round(le3_pct - ov_pct, 3)
        })

    return rows

all_standings = []
for season_year in [2024, 2025, 2026]:
    season_data = scrape_expanded_standings_for_season(season_year)
    all_standings.extend(season_data)
    time.sleep(4.5)

df_all = pd.DataFrame(all_standings)
df_all.to_csv("dim_team_expanded_standings.csv", index=False)
print(f"\n[BASARILI] Toplam {len(df_all)} satir Expanded Standings verisi toplandi.")

if os.path.exists(KEY_PATH) and not df_all.empty:
    print("BigQuery 'dim_team_expanded_standings' tablosu guncelleniyor...")
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True
    )
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.dim_team_expanded_standings"
    client.load_table_from_dataframe(df_all, t_ref, job_config=job_config).result()
    print(f"[BASARILI] BigQuery '{t_ref}' tablosu olusturuldu ve guncellendi!")