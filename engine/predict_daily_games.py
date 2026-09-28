import os
import re
import json
import argparse
import unicodedata
from datetime import datetime, timezone, timedelta
import dateutil.parser
import requests
import numpy as np
import pandas as pd
from scipy.stats import norm
from difflib import SequenceMatcher

BASE_SIGMA = 12.2
LEAGUE_PACE_BASE = 99.5
HCA = 2.60  # Ev Sahibi Avantajı Puanı

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

def clean_name(name):
    """İsimleri normalize eder: Aksanları (Jokić -> Jokic), Jr./III eklerini temizler."""
    if not name: return ""
    nfkd = unicodedata.normalize("NFKD", name)
    name = "".join([c for c in nfkd if not unicodedata.combining(c)])
    name = re.sub(r"\b(Jr\.?|Sr\.?|II|III|IV)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[\.\'\-]", "", name)
    return " ".join(name.lower().split())

# ---------------------------------------------------------
# 1. Yerel Oyuncu Puan Havuzunu Yükle (BigQuery'ye Gitmez)
# ---------------------------------------------------------
print("1. Oyuncu projeksiyon referans havuzu yukleniyor...")

CSV_PATH = "proj_player_stats_2026_27.csv"
if os.path.exists(CSV_PATH):
    df_players_pool = pd.read_csv(CSV_PATH)
else:
    # Eğer CSV yoksa BigQuery'den tek seferlik çekip CSV olarak kaydeder
    print("   -> Yerel CSV bulunamadi, BigQuery'den havuz indiriliyor...")
    from google.cloud import bigquery
    from google.oauth2 import service_account
    KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    bq_client = bigquery.Client(credentials=credentials, project="nba-analytics-503718")
    df_players_pool = bq_client.query("SELECT * FROM `nba-analytics-503718.nba_analytics.proj_player_stats_2026_27`").to_dataframe()
    df_players_pool.to_csv(CSV_PATH, index=False)

# Hızlı arama için oyuncu isim haritası oluştur
player_ref_dict = {}
for _, r in df_players_pool.iterrows():
    c_name = clean_name(r.get("player_name", ""))
    player_ref_dict[c_name] = {
        "player_id": r.get("player_id"),
        "pts36": float(r.get("proj_p36_pts", 12.0) or 12.0),
        "trb36": float(r.get("proj_p36_trb", 4.0) or 4.0),
        "ast36": float(r.get("proj_p36_ast", 2.5) or 2.5),
        "fg3m": float(r.get("proj_fg3m", 1.0) or 1.0),
        "spg": float(r.get("proj_spg", 0.7) or 0.7),
        "bpg": float(r.get("proj_bpg", 0.4) or 0.4),
        "bpm": float(r.get("proj_bpm", -1.0) or -1.0),
        "mpg": float(r.get("proj_mpg", 20.0) or 20.0),
        "age": int(r.get("age", 25) or 25)
    }

print(f"   -> {len(player_ref_dict)} oyuncunun temel metrikleri hazir.")

# ---------------------------------------------------------
# 2. ESPN API'den Canlı Kadroları ve Sakatlıkları Çekme
# ---------------------------------------------------------
session = requests.Session()
session.headers.update(HEADERS)

def get_live_team_roster(team_espn_id):
    """ESPN API'den takimin O ANKI gercek kadrosunu ceker."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_espn_id}/roster"
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200: return []
        data = res.json()
        athletes_raw = data.get("athletes", [])
        
        roster = []
        for item in athletes_raw:
            ath_list = item.get("items", [item]) if isinstance(item, dict) and "items" in item else [item]
            for ath in ath_list:
                name = ath.get("fullName") or ath.get("displayName", "")
                if name:
                    roster.append({
                        "espn_id": ath.get("id"),
                        "player_name": name,
                        "position": ath.get("position", {}).get("abbreviation", "G") if isinstance(ath.get("position"), dict) else "G",
                        "jersey": ath.get("jersey", "")
                    })
        return roster
    except Exception as e:
        print(f"   [HATA] Kadro cekilemedi ({team_espn_id}): {e}")
        return []

def get_league_injuries():
    """ESPN'den lig genelindeki anlik sakatlik raporunu ceker."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    out_players = set()
    try:
        res = session.get(url, timeout=10)
        if res.status_code == 200:
            for t in res.json().get("injuries", []):
                for p in t.get("injuries", []):
                    st = str(p.get("status", "")).lower()
                    if any(term in st for term in ["out", "doubtful", "suspended"]):
                        p_name = p.get("athlete", {}).get("displayName", "")
                        out_players.add(clean_name(p_name))
    except Exception as e:
        print(f"   [UYARI] Sakatlik raporu alinamadi: {e}")
    return out_players

# ---------------------------------------------------------
# 3. 240-Dakika Rotasyon ve Oyuncu Tahmin Motoru
# ---------------------------------------------------------
def project_team(live_roster, team_abbr, opp_abbr, out_set, match_pace):
    # Sakat oyuncuları canlı kadrodan çıkar
    active_roster = []
    for p in live_roster:
        c_name = clean_name(p["player_name"])
        if c_name not in out_set:
            # Oyuncu havuzumuzda var mı kontrol et
            if c_name in player_ref_dict:
                stats = player_ref_dict[c_name]
            else:
                # Fuzzy Arama
                matched_stats = None
                for k, v in player_ref_dict.items():
                    if SequenceMatcher(None, c_name, k).ratio() >= 0.88:
                        matched_stats = v
                        break
                # Havuzda hiç yoksa (yeni çaylak) taban istatistik ata
                stats = matched_stats if matched_stats else {
                    "player_id": f"new_{c_name[:6]}", "pts36": 10.0, "trb36": 3.5,
                    "ast36": 2.0, "fg3m": 0.8, "spg": 0.5, "bpg": 0.3,
                    "bpm": -2.0, "mpg": 14.0, "age": 22
                }

            active_roster.append({
                "player_name": p["player_name"],
                "position": p["position"],
                **stats
            })

    if not active_roster:
        return 105.0, []

    # En kaliteli oyunculara göre sırala
    active_roster = sorted(active_roster, key=lambda x: (x["bpm"], x["mpg"]), reverse=True)

    # 240 Dakikayı Canlı Kadrodaki Sağlıklı İlk 10 Oyuncuya Paylaştır
    ROTATION = [34.0, 33.0, 31.0, 29.0, 27.0, 23.0, 21.0, 18.0, 14.0, 10.0]
    total_assigned = sum(ROTATION[:len(active_roster)])
    
    player_preds = []
    team_pts = 0.0
    pace_factor = match_pace / LEAGUE_PACE_BASE

    for i, p in enumerate(active_roster):
        raw_min = ROTATION[i] if i < len(ROTATION) else 0.0
        if raw_min == 0.0: continue

        game_min = round((raw_min / total_assigned) * 240.0, 1)
        min_ratio = (game_min / 36.0) * pace_factor

        pts = round(p["pts36"] * min_ratio, 1)
        reb = round(p["trb36"] * min_ratio, 1)
        ast = round(p["ast36"] * min_ratio, 1)
        fg3m = round(p["fg3m"] * (game_min / 30.0) * pace_factor, 1)
        stl = round(p["spg"] * (game_min / 30.0), 1)
        blk = round(p["bpg"] * (game_min / 30.0), 1)

        team_pts += pts

        player_preds.append({
            "player_name": p["player_name"],
            "team_id": team_abbr,
            "position": p["position"],
            "projected_mins": game_min,
            "pts": pts, "reb": reb, "ast": ast,
            "fg3m": fg3m, "stl": stl, "blk": blk
        })

    team_pts = round(team_pts + 5.0, 1)  # Serbest atış/faul düzeltmesi
    return team_pts, player_preds

# ---------------------------------------------------------
# 4. Ana Tahmin Akışı (Günün Maçları)
# ---------------------------------------------------------
def run(target_date_str):
    print(f"\n2. ESPN'den {target_date_str} fiksturu cekiliyor...")
    sb_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={target_date_str}"
    res = session.get(sb_url, timeout=12)
    events = res.json().get("events", []) if res.status_code == 200 else []

    if not events:
        print(f"[UYARI] {target_date_str} tarihinde oynanacak mac bulunamadi.")
        return

    out_players = get_league_injuries()
    print(f"   -> Toplam {len(out_players)} sakat/cezali oyuncu tespit edildi.")

    game_cards = []
    all_player_props = []

    print("\n" + "=" * 105)
    print(f"  NBA MAC VE OYUNCU TAHMINLERI (TARIH: {target_date_str} - TAMAMEN CANLI KADROLARLA)")
    print("=" * 105)

    for ev in events:
        game_id = ev.get("id")
        comp = ev.get("competitions", [])[0]
        competitors = comp.get("competitors", [])

        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        h_id, h_abbr = home.get("id"), home.get("team", {}).get("abbreviation")
        a_id, a_abbr = away.get("id"), away.get("team", {}).get("abbreviation")
        h_logo = home.get("team", {}).get("logo")
        a_logo = away.get("team", {}).get("logo")

        # Türkiye Saati
        utc_dt = dateutil.parser.parse(ev.get("date"))
        tr_time = (utc_dt.astimezone(timezone.utc).replace(tzinfo=None) + timedelta(hours=3)).strftime("%d.%m.%Y %H:%M")

        # ESPN'den O ANKİ Canlı Kadroları Çek (BigQuery yok!)
        h_roster = get_live_team_roster(h_id)
        a_roster = get_live_team_roster(a_id)

        match_pace = 100.0  # Ortalama lig temposu
        h_score, h_players = project_team(h_roster, h_abbr, a_abbr, out_players, match_pace)
        a_score, a_players = project_team(a_roster, a_abbr, h_abbr, out_players, match_pace)

        # Ev Sahibi Avantajı
        h_score = round(h_score + (HCA / 2.0), 1)
        a_score = round(a_score - (HCA / 2.0), 1)

        spread = round(h_score - a_score, 1)
        total = round(h_score + a_score, 1)
        h_win_prob = round(float(norm.cdf(spread / BASE_SIGMA) * 100.0), 1)
        a_win_prob = round(100.0 - h_win_prob, 1)
        winner = h_abbr if spread >= 0 else a_abbr

        # Terminale Yazdır
        print(f"\n[{game_id}] {a_abbr} @ {h_abbr} | {tr_time} (TSİ)")
        print(f"   -> SKOR TAHMINI: {a_abbr} {a_score} - {h_score} {h_abbr}")
        print(f"   -> SPREAD: {h_abbr} {('-' if spread > 0 else '+')}{abs(spread)} | TOPLAM SAYI: {total}")
        print(f"   -> KAZANMA IHTIMALI: %{a_win_prob} {a_abbr} | %{h_win_prob} {h_abbr} (Favori: {winner})")
        
        # Maçın öne çıkan oyuncu tahminleri
        top_stars = sorted(h_players + a_players, key=lambda x: x["pts"], reverse=True)[:3]
        for s in top_stars:
            print(f"      * {s['player_name']} ({s['team_id']}): {s['pts']} Pts | {s['reb']} Reb | {s['ast']} Ast | {s['fg3m']} 3PM ({s['projected_mins']} Dk)")

        # Web sitesi için kart objesi
        game_cards.append({
            "game_id": game_id,
            "game_date": target_date_str,
            "match_time_tr": tr_time,
            "home_team": h_abbr,
            "away_team": a_abbr,
            "home_logo": h_logo,
            "away_logo": a_logo,
            "proj_home_score": h_score,
            "proj_away_score": a_score,
            "proj_spread": spread,
            "proj_total": total,
            "home_win_prob": h_win_prob,
            "away_win_prob": a_win_prob,
            "predicted_winner": winner
        })
        for p in (h_players + a_players):
            p["game_id"] = game_id
            all_player_props.append(p)

    # ---------------------------------------------------------
    # 5. Web Sitesi / API İçin JSON Çıktısı
    # ---------------------------------------------------------
    os.makedirs("output", exist_ok=True)
    out_file = f"output/predictions_{target_date_str}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"games": game_cards, "player_props": all_player_props}, f, ensure_ascii=False, indent=2)

    # Son tahminleri web dashboard'un hemen okuyacağı dosya olarak kopyala
    with open("output/daily_predictions_latest.json", "w", encoding="utf-8") as f:
        json.dump({"games": game_cards, "player_props": all_player_props}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 105)
    print(f"[BASARILI] {len(game_cards)} macin tahminleri 'output/daily_predictions_latest.json' dosyasina yazildi!")
    print("=" * 105)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default="20261020", help="Format: YYYYMMDD")
    args = parser.parse_args()
    run(args.date)