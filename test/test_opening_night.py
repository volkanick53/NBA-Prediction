import requests
import json
from datetime import datetime, timezone
import dateutil.parser

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

# 2026-27 Sezonu Açılış Tarihi (YYYYMMDD)
# Tek gün: "20261020" | Açılış haftası aralığı: "20261020-20261023"
OPENING_DATES = "20261020-20261022"

print("=" * 90)
print(f"  NBA 2026-27 SEZONU ACILIS MACLARI & VERI TESTI (Tarih: {OPENING_DATES})")
print("=" * 90)

# ---------------------------------------------------------
# 1. Açılış Maçlarını Çek (/scoreboard)
# ---------------------------------------------------------
scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={OPENING_DATES}"

try:
    resp = requests.get(scoreboard_url, headers=HEADERS, timeout=12)
    if resp.status_code != 200:
        print(f"[HATA] Fikstur alinamadi: HTTP {resp.status_code}")
        exit()

    data = resp.json()
    events = data.get("events", [])
    print(f"\n[BASARILI] Toplam {len(events)} acilis maci takvime yerlestirildi.\n")

    parsed_games = []
    active_teams_in_opening = set()

    for ev in events:
        ev_id = ev.get("id")
        utc_date_str = ev.get("date")
        
        # UTC Tarihini Türkiye Saatine (UTC+3) Çevir
        dt_utc = dateutil.parser.parse(utc_date_str)
        dt_tr = dt_utc.astimezone(timezone.utc).replace(tzinfo=None)
        # UTC+3 ekle
        from datetime import timedelta
        dt_tr = dt_tr + timedelta(hours=3)
        tr_time_formatted = dt_tr.strftime("%d.%m.%Y - Saat %H:%M")

        comp = ev.get("competitions", [])[0]
        competitors = comp.get("competitors", [])

        home_obj = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away_obj = next((c for c in competitors if c.get("homeAway") == "away"), {})

        home_team = home_obj.get("team", {})
        away_team = away_obj.get("team", {})

        home_abbr = home_team.get("abbreviation")
        away_abbr = away_team.get("abbreviation")
        home_name = home_team.get("displayName")
        away_name = away_team.get("displayName")
        home_logo = home_team.get("logo")
        away_logo = away_team.get("logo")

        active_teams_in_opening.add(home_abbr)
        active_teams_in_opening.add(away_abbr)

        venue = comp.get("venue", {}).get("fullName", "Bilinmeyen Salon")
        city = comp.get("venue", {}).get("address", {}).get("city", "")

        # Yayın kanalı
        broadcasts = comp.get("broadcasts", [])
        tv_channel = broadcasts[0].get("names", ["-"])[0] if broadcasts else "Yerel / NBA TV"

        parsed_games.append({
            "game_id": ev_id,
            "matchup": f"{away_abbr} @ {home_abbr}",
            "home_team": home_name,
            "away_team": away_name,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_logo": home_logo,
            "away_logo": away_logo,
            "match_time_tr": tr_time_formatted,
            "venue": f"{venue} ({city})" if city else venue,
            "tv": tv_channel
        })

    # Terminal Çıktısı (Maç Kartları Önizleme)
    print("-" * 90)
    print(f"{'MAC ID':<12} | {'ESLEŞME':<12} | {'TARIH & TSİ':<22} | {'SALON & SEHIR':<26} | {'YAYIN'}")
    print("-" * 90)
    for g in parsed_games:
        print(f"{g['game_id']:<12} | {g['matchup']:<12} | {g['match_time_tr']:<22} | {g['venue'][:25]:<26} | {g['tv']}")
    print("-" * 90)

except Exception as e:
    print(f"[BAGLANTI HATASI] {e}")
    exit()

# ---------------------------------------------------------
# 2. Sakatlık Raporu (/injuries) Filtreleme
# ---------------------------------------------------------
print("\n" + "=" * 90)
print("  ACILIS GECESI OYNAYACAK TAKIMLARIN SAKATLIK DURUMLARI")
print("=" * 90)

injuries_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
try:
    r_inj = requests.get(injuries_url, headers=HEADERS, timeout=12)
    if r_inj.status_code == 200:
        inj_data = r_inj.json().get("injuries", [])
        
        found_injuries = 0
        for team_inj in inj_data:
            t_abbr = team_inj.get("team", {}).get("abbreviation")
            t_name = team_inj.get("team", {}).get("displayName")

            # Sadece açılış gecesi/haftası maçı olan takımların sakatlıklarını filtrele
            if t_abbr in active_teams_in_opening:
                items = team_inj.get("injuries", [])
                if items:
                    print(f"\n* {t_name} ({t_abbr}):")
                    for p in items:
                        p_name = p.get("athlete", {}).get("displayName")
                        status = p.get("status", "Bilinmiyor")  # Out, Questionable, Day-to-Day
                        comment = p.get("shortComment", p.get("details", {}).get("detail", "Aciklama yok"))
                        print(f"   [-] {p_name:<24} -> Durum: [{status:<12}] | Not: {comment}")
                        found_injuries += 1

        if found_injuries == 0:
            print("\n   -> Bu takımlarda şu an için resmi sakatlık raporu bulunmuyor.")
    else:
        print(f"[UYARI] Sakatlık endpointi HTTP {r_inj.status_code}")
except Exception as e:
    print(f"[HATA] Sakatlık verisi alınamadı: {e}")

print("\n" + "=" * 90)
print("  VERI FORMATI ONAYLANDI: Model girdisi ve Web kartlari icin saf veri hazir!")
print("=" * 90)