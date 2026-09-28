import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

SIMULATION_ROUNDS = 10000
BASE_PLAYOFF_SIGMA = 11.6  # Playoff sertliğinde daha düşük varyans
LEAGUE_PACE_BASE = 99.5

# Konferans Eşlemeleri (15 Doğu, 15 Batı)
CONFERENCE_MAP = {
    # Doğu Konferansı (Eastern Conference)
    "BOS": "East", "NYK": "East", "PHI": "East", "CLE": "East", "MIL": "East",
    "IND": "East", "ORL": "East", "MIA": "East", "ATL": "East", "CHI": "East",
    "TOR": "East", "BRK": "East", "DET": "East", "CHO": "East", "WAS": "East",
    
    # Batı Konferansı (Western Conference)
    "OKC": "West", "DEN": "West", "MIN": "West", "DAL": "West", "HOU": "West",
    "MEM": "West", "PHO": "West", "SAC": "West", "GSW": "West", "LAL": "West",
    "LAC": "West", "NOP": "West", "SAS": "West", "POR": "West", "UTA": "West"
}

# ---------------------------------------------------------
# 1. BigQuery Tablolarını Yükle
# ---------------------------------------------------------
print("1. BigQuery verileri yukleniyor...")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

df_team_proj = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.proj_team_wins_2026_27`").to_dataframe()
df_roster = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27`").to_dataframe()
df_player_profiles = client.query(f"SELECT player_id, bpm, ws_per_48, season, mp FROM `{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles`").to_dataframe()

for col in ["bpm", "ws_per_48", "mp"]:
    if col in df_player_profiles.columns:
        df_player_profiles[col] = pd.to_numeric(df_player_profiles[col], errors="coerce")

# ---------------------------------------------------------
# 2. Playoff Rotasyon Gücü Hesabı (7-Oyunculu Daralma)
# ---------------------------------------------------------
print("2. Playoff rotasyon daralmasi (7-Man Rotation) hesaplaniyor...")

# Oyuncu ratinglerini hesapla
df_p_clean = df_player_profiles.sort_values(by=["player_id", "season", "mp"], ascending=[True, True, False]).drop_duplicates(subset=["player_id", "season"])
df_p_clean["composite"] = df_p_clean["bpm"].fillna(0.0) * 0.60 + ((df_p_clean["ws_per_48"].fillna(0.100) - 0.100) * 30.0) * 0.40

weights = {"2025-26": 0.65, "2024-25": 0.25, "2023-24": 0.10}
df_p_clean["w"] = df_p_clean["season"].map(weights).fillna(0.05)

p_ratings = {}
for p_id, grp in df_p_clean.groupby("player_id"):
    w_sum = grp["w"].sum()
    p_ratings[p_id] = float((grp["composite"] * grp["w"]).sum() / w_sum) if w_sum > 0 else -1.5

df_roster["p_rating"] = df_roster["player_id"].map(p_ratings).fillna(-1.5)

# Playoff 7-Oyunculu Rotasyon Ağırlıkları (Yıldız yoğunluğu)
# İlk 3 oyuncu = %66 ağırlık, ilk 5 = %90, ilk 7 = %100
PLAYOFF_ROTATION_WEIGHTS = [0.26, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04]

team_playoff_net_ratings = {}
team_hca_map = df_team_proj.set_index("team_id")["empirical_hca"].to_dict()
team_reg_net = df_team_proj.set_index("team_id")["net_rating_proj"].to_dict()
team_pace_map = df_team_proj.set_index("team_id")["proj_pace"].to_dict()
team_exp_wins = df_team_proj.set_index("team_id")["exp_wins"].to_dict()
team_win_std = df_team_proj.set_index("team_id")["win_std_dev"].to_dict()

for t_id, grp in df_roster.groupby("team_id"):
    sorted_p = grp.sort_values(by="p_rating", ascending=False).reset_index(drop=True)
    raw_playoff_p = sum(
        (sorted_p.iloc[i]["p_rating"] if i < len(sorted_p) else -1.5) * w
        for i, w in enumerate(PLAYOFF_ROTATION_WEIGHTS)
    )
    # Normal sezon Net Rating ile playoff süperstar gücünü harmanla (%50 Reg Net + %50 Playoff Top-7)
    reg_val = team_reg_net.get(t_id, 0.0)
    team_playoff_net_ratings[t_id] = round(reg_val * 0.50 + raw_playoff_p * 0.50, 2)

# ---------------------------------------------------------
# 3. Maç & Seri Simülasyon Fonksiyonları
# ---------------------------------------------------------
def sim_game(team_home, team_away, is_game_7=False):
    """Tekil bir playoff maçını simüle eder."""
    h_net = team_playoff_net_ratings.get(team_home, 0.0)
    a_net = team_playoff_net_ratings.get(team_away, 0.0)
    
    # HCA + Game 7 ekstra ev sahibi avantajı
    hca = team_hca_map.get(team_home, 2.60)
    if is_game_7:
        hca += 0.70  # 7. maç ev sahibi psikolojik üstünlüğü

    # Playoff tempo daralması (-2.5 pace)
    h_pace = team_pace_map.get(team_home, LEAGUE_PACE_BASE) - 2.5
    a_pace = team_pace_map.get(team_away, LEAGUE_PACE_BASE) - 2.5
    avg_pace = (h_pace + a_pace) / 2.0
    
    sigma = BASE_PLAYOFF_SIGMA * np.sqrt(avg_pace / 100.0)
    spread = (h_net + hca) - a_net
    home_win_prob = norm.cdf(spread / sigma)

    return 1 if np.random.rand() < home_win_prob else 0

def sim_series(team_high, team_low):
    """2-2-1-1-1 formatında Best-of-7 serisini simüle eder."""
    high_wins = 0
    low_wins = 0
    # 2-2-1-1-1 Saha Dağılımı (True: team_high evinde, False: team_low evinde)
    home_pattern = [True, True, False, False, True, False, True]

    for g_num in range(7):
        is_g7 = (g_num == 6)
        if home_pattern[g_num]:
            res = sim_game(team_high, team_low, is_game_7=is_g7)
            if res == 1: high_wins += 1
            else: low_wins += 1
        else:
            res = sim_game(team_low, team_high, is_game_7=is_g7)
            if res == 1: low_wins += 1
            else: high_wins += 1

        if high_wins == 4:
            return team_high, high_wins, low_wins
        if low_wins == 4:
            return team_low, low_wins, high_wins

    return team_high, 4, 3

# ---------------------------------------------------------
# 4. 10.000 Sezonluk Monte Carlo Playoff Motoru
# ---------------------------------------------------------
print(f"3. 10.000 Playoff ve Sampiyonluk Braketi simulasyonu basliyor...")

stats_tracker = {
    t: {
        "make_playin": 0, "make_playoffs": 0, "reach_semis": 0,
        "reach_conf_finals": 0, "reach_finals": 0, "champion": 0,
        "total_playoff_wins": 0
    }
    for t in team_playoff_net_ratings.keys()
}

teams_list = list(team_playoff_net_ratings.keys())

for sim_i in range(SIMULATION_ROUNDS):
    # 1. Normal Sezon Galibiyet Çekilişi & Konferans Sıralamaları
    season_wins = {}
    for t in teams_list:
        mean_w = team_exp_wins.get(t, 41.0)
        std_w = team_win_std.get(t, 5.0)
        # Normal sezon galibiyeti zar atışı
        season_wins[t] = np.random.normal(mean_w, std_w)

    conf_standings = {"East": [], "West": []}
    for t, w in season_wins.items():
        conf = CONFERENCE_MAP.get(t, "East")
        conf_standings[conf].append((t, w))

    # Sıralamayı yap
    for conf in ["East", "West"]:
        conf_standings[conf].sort(key=lambda x: x[1], reverse=True)

    playoff_seeds = {"East": {}, "West": {}}

    # 2. Play-In Turnuvası (Doğu ve Batı)
    for conf in ["East", "West"]:
        st = [x[0] for x in conf_standings[conf]]
        
        # 1-6 Direkt Playoff
        for s_idx in range(6):
            playoff_seeds[conf][s_idx + 1] = st[s_idx]
            stats_tracker[st[s_idx]]["make_playoffs"] += 1

        t7, t8, t9, t10 = st[6], st[7], st[8], st[9]
        for pt in [t7, t8, t9, t10]:
            stats_tracker[pt]["make_playin"] += 1

        # Maç 1: 7 vs 8 (7 evinde) -> Kazanan #7 Seed
        res_7v8 = sim_game(t7, t8)
        win_7v8 = t7 if res_7v8 == 1 else t8
        los_7v8 = t8 if res_7v8 == 1 else t7

        playoff_seeds[conf][7] = win_7v8
        stats_tracker[win_7v8]["make_playoffs"] += 1

        # Maç 2: 9 vs 10 (9 evinde) -> Kaybeden elenir
        res_9v10 = sim_game(t9, t10)
        win_9v10 = t9 if res_9v10 == 1 else t10

        # Maç 3: Final (Loser 7v8 evinde vs Winner 9v10) -> Kazanan #8 Seed
        res_final_pi = sim_game(los_7v8, win_9v10)
        win_pi_8 = los_7v8 if res_final_pi == 1 else win_9v10

        playoff_seeds[conf][8] = win_pi_8
        stats_tracker[win_pi_8]["make_playoffs"] += 1

    # 3. Konferans Elemeleri (Round 1 -> Semis -> Conf Finals)
    conf_champs = {}

    for conf in ["East", "West"]:
        seeds = playoff_seeds[conf]
        
        # 1. Tur (Best-of-7)
        w_1v8, w1_w, l1_w = sim_series(seeds[1], seeds[8])
        w_4v5, w4_w, l4_w = sim_series(seeds[4], seeds[5])
        w_3v6, w3_w, l3_w = sim_series(seeds[3], seeds[6])
        w_2v7, w2_w, l2_w = sim_series(seeds[2], seeds[7])

        stats_tracker[seeds[1]]["total_playoff_wins"] += (w1_w if w_1v8 == seeds[1] else l1_w)
        stats_tracker[seeds[8]]["total_playoff_wins"] += (w1_w if w_1v8 == seeds[8] else l1_w)
        stats_tracker[seeds[4]]["total_playoff_wins"] += (w4_w if w_4v5 == seeds[4] else l4_w)
        stats_tracker[seeds[5]]["total_playoff_wins"] += (w4_w if w_4v5 == seeds[5] else l4_w)
        stats_tracker[seeds[3]]["total_playoff_wins"] += (w3_w if w_3v6 == seeds[3] else l3_w)
        stats_tracker[seeds[6]]["total_playoff_wins"] += (w3_w if w_3v6 == seeds[6] else l3_w)
        stats_tracker[seeds[2]]["total_playoff_wins"] += (w2_w if w_2v7 == seeds[2] else l2_w)
        stats_tracker[seeds[7]]["total_playoff_wins"] += (w2_w if w_2v7 == seeds[7] else l2_w)

        for semi_t in [w_1v8, w_4v5, w_3v6, w_2v7]:
            stats_tracker[semi_t]["reach_semis"] += 1

        # 2. Tur (Konferans Yarı Finalleri)
        # HCA: Normal sezonda daha çok kazanan
        t_high_1 = w_1v8 if season_wins[w_1v8] >= season_wins[w_4v5] else w_4v5
        t_low_1 = w_4v5 if t_high_1 == w_1v8 else w_1v8
        w_semi_1, ws1_w, ls1_w = sim_series(t_high_1, t_low_1)

        t_high_2 = w_2v7 if season_wins[w_2v7] >= season_wins[w_3v6] else w_3v6
        t_low_2 = w_3v6 if t_high_2 == w_2v7 else w_2v7
        w_semi_2, ws2_w, ls2_w = sim_series(t_high_2, t_low_2)

        stats_tracker[t_high_1]["total_playoff_wins"] += (ws1_w if w_semi_1 == t_high_1 else ls1_w)
        stats_tracker[t_low_1]["total_playoff_wins"] += (ws1_w if w_semi_1 == t_low_1 else ls1_w)
        stats_tracker[t_high_2]["total_playoff_wins"] += (ws2_w if w_semi_2 == t_high_2 else ls2_w)
        stats_tracker[t_low_2]["total_playoff_wins"] += (ws2_w if w_semi_2 == t_low_2 else ls2_w)

        for cf_t in [w_semi_1, w_semi_2]:
            stats_tracker[cf_t]["reach_conf_finals"] += 1

        # Konferans Finali
        t_high_cf = w_semi_1 if season_wins[w_semi_1] >= season_wins[w_semi_2] else w_semi_2
        t_low_cf = w_semi_2 if t_high_cf == w_semi_1 else w_semi_1
        conf_champ, wcf_w, lcf_w = sim_series(t_high_cf, t_low_cf)

        stats_tracker[t_high_cf]["total_playoff_wins"] += (wcf_w if conf_champ == t_high_cf else lcf_w)
        stats_tracker[t_low_cf]["total_playoff_wins"] += (wcf_w if conf_champ == t_low_cf else lcf_w)

        conf_champs[conf] = conf_champ
        stats_tracker[conf_champ]["reach_finals"] += 1

    # 4. NBA Finalleri (East Champion vs West Champion)
    east_c = conf_champs["East"]
    west_c = conf_champs["West"]

    t_high_finals = east_c if season_wins[east_c] >= season_wins[west_c] else west_c
    t_low_finals = west_c if t_high_finals == east_c else east_c

    nba_champion, wf_w, lf_w = sim_series(t_high_finals, t_low_finals)

    stats_tracker[t_high_finals]["total_playoff_wins"] += (wf_w if nba_champion == t_high_finals else lf_w)
    stats_tracker[t_low_finals]["total_playoff_wins"] += (wf_w if nba_champion == t_low_finals else lf_w)
    stats_tracker[nba_champion]["champion"] += 1

# ---------------------------------------------------------
# 5. Raporlama ve BigQuery Kayıt
# ---------------------------------------------------------
summary_rows = []
for t in teams_list:
    st = stats_tracker[t]
    summary_rows.append({
        "team_id": t,
        "conference": CONFERENCE_MAP.get(t, "East"),
        "season": "2026-27",
        "playoff_net_rating": team_playoff_net_ratings.get(t, 0.0),
        "make_playin_pct": round((st["make_playin"] / SIMULATION_ROUNDS) * 100, 1),
        "make_playoffs_pct": round((st["make_playoffs"] / SIMULATION_ROUNDS) * 100, 1),
        "reach_conf_semis_pct": round((st["reach_semis"] / SIMULATION_ROUNDS) * 100, 1),
        "reach_conf_finals_pct": round((st["reach_conf_finals"] / SIMULATION_ROUNDS) * 100, 1),
        "reach_nba_finals_pct": round((st["reach_finals"] / SIMULATION_ROUNDS) * 100, 1),
        "win_championship_pct": round((st["champion"] / SIMULATION_ROUNDS) * 100, 2),
        "exp_playoff_wins": round(st["total_playoff_wins"] / SIMULATION_ROUNDS, 1)
    })

df_playoffs = pd.DataFrame(summary_rows).sort_values(by=["win_championship_pct", "reach_nba_finals_pct"], ascending=False).reset_index(drop=True)

print("\n" + "="*125)
print("  2026-27 NBA PLAYOFF & SAMPIYONLUK PROJEKSIYONLARI (10.000 MONTE CARLO)")
print("="*125)
print(df_playoffs.to_string(index=False))

df_playoffs.to_csv("proj_playoffs_2026_27.csv", index=False)

if os.path.exists(KEY_PATH):
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True)
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.proj_playoffs_2026_27"
    client.load_table_from_dataframe(df_playoffs, t_ref, job_config=job_config).result()
    print(f"\n[BASARILI] BigQuery '{t_ref}' tablosu guncellendi!")