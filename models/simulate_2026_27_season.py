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
SINGLE_GAME_SIGMA = 12.4

TEAM_HCA_MAP = {
    "DEN": 4.10, "UTA": 3.50, "BOS": 3.30, "NYK": 3.20, "PHI": 3.10,
    "OKC": 3.10, "GSW": 3.00, "CLE": 2.90, "MIN": 2.90, "MIA": 2.90,
    "MIL": 2.85, "DAL": 2.80, "SAC": 2.80, "IND": 2.75, "HOU": 2.75,
    "MEM": 2.70, "LAL": 2.70, "NOP": 2.65, "TOR": 2.65, "PHO": 2.60,
    "ORL": 2.60, "SAS": 2.60, "CHI": 2.55, "POR": 2.50, "ATL": 2.40,
    "DET": 2.20, "CHO": 2.10, "LAC": 1.90, "BRK": 1.85, "WAS": 1.80
}

TIMEZONE_MAP = {
    "BOS": "ET", "NYK": "ET", "BRK": "ET", "PHI": "ET", "TOR": "ET",
    "WAS": "ET", "CLE": "ET", "DET": "ET", "IND": "ET", "CHO": "ET",
    "ATL": "ET", "MIA": "ET", "ORL": "ET",
    "CHI": "CT", "MIL": "CT", "MIN": "CT", "MEM": "CT", "NOP": "CT",
    "OKC": "CT", "DAL": "CT", "HOU": "CT", "SAS": "CT",
    "DEN": "MT", "UTA": "MT",
    "PHO": "MT", "POR": "PT", "SAC": "PT", "GSW": "PT", "LAC": "PT", "LAL": "PT"
}
TZ_ORDER = {"ET": 3, "CT": 2, "MT": 1, "PT": 0}

# ---------------------------------------------------------
# 1. Verileri Çek
# ---------------------------------------------------------
print("1. BigQuery verileri yukleniyor...")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

df_sched = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_schedule_2026_27`").to_dataframe()
df_roster = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27`").to_dataframe()
df_profiles_raw = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles`").to_dataframe()

for col in ["bpm", "vorp", "ws_per_48", "mp"]:
    if col in df_profiles_raw.columns:
        df_profiles_raw[col] = pd.to_numeric(df_profiles_raw[col], errors="coerce")

df_roster["age"] = pd.to_numeric(df_roster["age"], errors="coerce").fillna(26)

# ---------------------------------------------------------
# 2. Kompozit Güç Endeksi (BPM + Win Shares / 48)
# ---------------------------------------------------------
print("2. Kompozit oyuncu gucu ve ilk 5 sinerjisi hesaplaniyor...")

# WS/48'i BPM skalasına eşle: (ws_per_48 - 0.100) * 35.0
df_profiles_raw["ws_scaled"] = (df_profiles_raw["ws_per_48"] - 0.100) * 35.0
df_profiles_raw["composite_rating"] = df_profiles_raw["bpm"] * 0.50 + df_profiles_raw["ws_scaled"] * 0.50

weights = {"2025-26": 0.65, "2024-25": 0.25, "2023-24": 0.10}
df_profiles_raw["season_w"] = df_profiles_raw["season"].map(weights).fillna(0.05)

# Takas satırlarını dakika (mp) ağırlıklı birleştir
def calc_player_rating(group):
    valid = group.dropna(subset=["composite_rating"])
    if valid.empty:
        return -2.0
    mp_w = valid["mp"].fillna(500.0) + 10.0
    tot_w = valid["season_w"] * mp_w
    if tot_w.sum() == 0:
        return -2.0
    return float((valid["composite_rating"] * tot_w).sum() / tot_w.sum())

player_ratings = df_profiles_raw.groupby("player_id").apply(calc_player_rating, include_groups=False).reset_index(name="proj_rating_base")

df_roster_ratings = df_roster.merge(player_ratings, on="player_id", how="left")
df_roster_ratings["proj_rating_base"] = df_roster_ratings["proj_rating_base"].fillna(-2.0)

def apply_aging_curve(row):
    r = row["proj_rating_base"]
    age = row["age"]
    if age <= 21: return r + 0.80
    elif age <= 23: return r + 0.40
    elif 24 <= age <= 29: return r
    elif 30 <= age <= 32: return r - 0.30
    elif 33 <= age <= 35: return r - 0.70
    else: return r - 1.20

df_roster_ratings["proj_rating"] = df_roster_ratings.apply(apply_aging_curve, axis=1)

# Dengeli 240 Dakika Dağılımı
ROTATION_WEIGHTS = [0.22, 0.18, 0.16, 0.14, 0.12, 0.08, 0.05, 0.03, 0.02]

raw_team_ratings = {}
team_avg_ages = {}
team_b2b_penalties = {}

for team, group in df_roster_ratings.groupby("team_id"):
    sorted_players = group.sort_values(by="proj_rating", ascending=False).reset_index(drop=True)
    
    t_rating = sum(
        (sorted_players.iloc[i]["proj_rating"] if i < len(sorted_players) else -2.0) * w
        for i, w in enumerate(ROTATION_WEIGHTS)
    )
    
    # Sinerji Kontrolü: İlk 5'in tamamı artı puandaysa sinerji bonusu ekle
    top5_ratings = [sorted_players.iloc[i]["proj_rating"] for i in range(min(5, len(sorted_players)))]
    if all(r > 0.8 for r in top5_ratings):
        t_rating += 0.75  # Şampiyonluk omurgası sinerji bonusu
        
    t_age = sum(
        (sorted_players.iloc[i]["age"] if i < len(sorted_players) else 26) * w
        for i, w in enumerate(ROTATION_WEIGHTS)
    )
    raw_team_ratings[team] = t_rating
    team_avg_ages[team] = round(t_age, 1)
    
    dyn_b2b = 1.30 + max(0.0, (t_age - 23.5) * 0.28)
    team_b2b_penalties[team] = round(dyn_b2b, 2)

r_series = pd.Series(raw_team_ratings)
z_scores = (r_series - r_series.mean()) / r_series.std()
team_net_ratings = (z_scores * 4.90).round(2).to_dict()

# ---------------------------------------------------------
# 3. Fikstür ve Seyahat Simülasyonu
# ---------------------------------------------------------
print("3. Fikstur dinamikleri ve seyahat matrisi hesaplaniyor...")

df_sched["game_date_dt"] = pd.to_datetime(df_sched["game_date"])
df_sched = df_sched.sort_values(by=["game_date_dt", "game_number"]).reset_index(drop=True)

team_last_date, team_last_city = {}, {}
team_consecutive_home = {t: 0 for t in team_net_ratings.keys()}
team_consecutive_road = {t: 0 for t in team_net_ratings.keys()}
match_spreads = []

for idx, row in df_sched.iterrows():
    g_date = row["game_date_dt"]
    h_team, a_team = row["home_team"], row["away_team"]
    is_neutral = row.get("is_neutral_site", False)
    g_time = str(row.get("game_time_et", ""))

    h_b2b = 1 if h_team in team_last_date and (g_date - team_last_date[h_team]).days == 1 else 0
    a_b2b = 1 if a_team in team_last_date and (g_date - team_last_date[a_team]).days == 1 else 0

    a_travel_penalty = 0.0
    if a_team in team_last_city:
        prev_city, curr_city = team_last_city[a_team], h_team
        if not ((prev_city in ["LAL", "LAC"] and curr_city in ["LAL", "LAC"]) or 
                (prev_city in ["NYK", "BRK"] and curr_city in ["NYK", "BRK"])):
            prev_tz = TZ_ORDER.get(TIMEZONE_MAP.get(prev_city, "ET"), 0)
            curr_tz = TZ_ORDER.get(TIMEZONE_MAP.get(curr_city, "ET"), 0)
            if curr_tz > prev_tz:
                a_travel_penalty += 0.45 * (curr_tz - prev_tz)

    h_net = team_net_ratings.get(h_team, 0.0)
    if not is_neutral:
        h_net += TEAM_HCA_MAP.get(h_team, 2.70)
        if team_consecutive_home.get(h_team, 0) >= 2: h_net += 0.40
        if any(pt in g_time for pt in ["7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM", "9:30 PM", "10:00 PM", "10:30 PM"]):
            h_net += 0.30

    if h_b2b: h_net -= team_b2b_penalties.get(h_team, 2.50)

    a_net = team_net_ratings.get(a_team, 0.0)
    if a_b2b:
        a_dyn_b2b = team_b2b_penalties.get(a_team, 2.50)
        if team_consecutive_road.get(a_team, 0) >= 2: a_dyn_b2b += 0.70
        a_net -= a_dyn_b2b

    if team_consecutive_road.get(a_team, 0) >= 3: a_net -= 0.65
    a_net -= a_travel_penalty

    if g_date.month in [3, 4]:
        if h_net < -3.5: h_net -= 2.2
        if a_net < -3.5: a_net -= 2.2

    spread = h_net - a_net
    match_spreads.append(spread)

    team_last_date[h_team] = g_date
    team_last_date[a_team] = g_date
    team_last_city[h_team] = h_team
    team_last_city[a_team] = h_team

    team_consecutive_home[h_team] = team_consecutive_home.get(h_team, 0) + 1
    team_consecutive_road[h_team] = 0
    team_consecutive_road[a_team] = team_consecutive_road.get(a_team, 0) + 1
    team_consecutive_home[a_team] = 0

win_probs = norm.cdf(np.array(match_spreads) / SINGLE_GAME_SIGMA)
df_sched["home_win_prob"] = win_probs

# ---------------------------------------------------------
# 4. 10.000 Sezonluk Monte Carlo Simülasyonu
# ---------------------------------------------------------
print(f"4. Monte Carlo Simulasyonu basliyor ({SIMULATION_ROUNDS:,} Sezon)...")

teams = list(team_net_ratings.keys())
n_games = len(df_sched)
prob_matrix = np.tile(win_probs, (SIMULATION_ROUNDS, 1))

random_draws = np.random.rand(SIMULATION_ROUNDS, n_games)
home_wins = (random_draws < prob_matrix).astype(int)
away_wins = 1 - home_wins

sim_results = {t: np.zeros(SIMULATION_ROUNDS, dtype=int) for t in teams}
home_team_arr = df_sched["home_team"].values
away_team_arr = df_sched["away_team"].values

for g_idx in range(n_games):
    h, a = home_team_arr[g_idx], away_team_arr[g_idx]
    if h in sim_results: sim_results[h] += home_wins[:, g_idx]
    if a in sim_results: sim_results[a] += away_wins[:, g_idx]

# ---------------------------------------------------------
# 5. Raporlama & BigQuery Kayıt
# ---------------------------------------------------------
final_rows = []
for t in teams:
    wins_dist = sim_results[t]
    exp_wins = round(float(np.mean(wins_dist)), 1)
    final_rows.append({
        "team_id": t,
        "season": "2026-27",
        "net_rating_proj": team_net_ratings.get(t, 0.0),
        "rot_avg_age": team_avg_ages.get(t, 26.0),
        "dyn_b2b_penalty": team_b2b_penalties.get(t, 2.50),
        "exp_wins": exp_wins,
        "exp_losses": round(82.0 - exp_wins, 1),
        "win_std_dev": round(float(np.std(wins_dist)), 2),
        "playoff_prob_pct": round(float(np.mean(wins_dist >= 42.0) * 100), 1),
        "fifty_plus_win_pct": round(float(np.mean(wins_dist >= 50.0) * 100), 1)
    })

df_proj = pd.DataFrame(final_rows).sort_values(by="exp_wins", ascending=False).reset_index(drop=True)

print("\n" + "="*100)
print("  2026-27 NBA SEZON PROJEKSIYONLARI (KOMPOZIT WS/BPM & SINERJI KALIBRELI)")
print("="*100)
print(df_proj.to_string(index=False))

df_proj.to_csv("proj_team_wins_2026_27.csv", index=False)

if os.path.exists(KEY_PATH):
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True)
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.proj_team_wins_2026_27"
    client.load_table_from_dataframe(df_proj, t_ref, job_config=job_config).result()
    print(f"\n[BASARILI] BigQuery '{t_ref}' tablosu guncellendi!")