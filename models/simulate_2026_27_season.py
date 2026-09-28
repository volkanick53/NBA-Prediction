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
BASE_SIGMA = 12.4
MIN_HEALTHY_MINUTES = 900.0

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

vegas_map = {
    "ATL": 44.5, "BOS": 49.5, "BRK": 24.5, "CHO": 37.5, "CHI": 29.5,
    "CLE": 47.5, "DAL": 35.5, "DEN": 48.5, "DET": 49.5, "GSW": 39.5,
    "HOU": 46.5, "IND": 43.5, "LAC": 28.5, "LAL": 45.5, "MEM": 28.5,
    "MIA": 45.5, "MIL": 24.5, "MIN": 48.5, "NOP": 28.5, "NYK": 51.5,
    "OKC": 60.5, "ORL": 44.5, "PHI": 50.5, "PHO": 39.5, "POR": 43.5,
    "SAC": 21.5, "SAS": 60.5, "TOR": 45.5, "UTA": 36.5, "WAS": 35.5
}

# ---------------------------------------------------------
# 1. BigQuery Tablolarını Yükle
# ---------------------------------------------------------
print("1. BigQuery tablolari yukleniyor...")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

df_sched = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_schedule_2026_27`").to_dataframe()
df_roster = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27`").to_dataframe()
df_profiles_raw = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles`").to_dataframe()
df_expanded = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_team_expanded_standings`").to_dataframe()
df_team_prof = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_team_season_profiles`").to_dataframe()

for col in ["bpm", "vorp", "ws_per_48", "mp", "age"]:
    if col in df_profiles_raw.columns:
        df_profiles_raw[col] = pd.to_numeric(df_profiles_raw[col], errors="coerce")

df_roster["age"] = pd.to_numeric(df_roster["age"], errors="coerce").fillna(26)

# ---------------------------------------------------------
# 2. Takas Stint Temizliği & Kompozit Puanlama
# ---------------------------------------------------------
print("2. Kadro verileri ve ampirik degiskenler isleniyor...")

df_profiles_raw = df_profiles_raw.sort_values(by=["player_id", "season", "mp"], ascending=[True, True, False])
df_profiles_clean = df_profiles_raw.drop_duplicates(subset=["player_id", "season"], keep="first").copy()

df_profiles_clean["ws_scaled"] = (df_profiles_clean["ws_per_48"] - 0.100) * 35.0
df_profiles_clean["composite_rating"] = df_profiles_clean["bpm"] * 0.55 + df_profiles_clean["ws_scaled"] * 0.45

weights = {"2025-26": 0.65, "2024-25": 0.25, "2023-24": 0.10}
df_profiles_clean["season_w"] = df_profiles_clean["season"].map(weights).fillna(0.05)

def calc_player_rating(group):
    valid = group.dropna(subset=["composite_rating"]).copy()
    if valid.empty:
        return -2.0

    healthy_seasons = valid[valid["mp"] >= MIN_HEALTHY_MINUTES]
    peak_rating = healthy_seasons["composite_rating"].max() if not healthy_seasons.empty else valid["composite_rating"].max()

    valid["dyn_w"] = valid["season_w"] * np.clip(valid["mp"].fillna(500) / MIN_HEALTHY_MINUTES, 0.25, 1.0)
    tot_w = valid["dyn_w"].sum()
    if tot_w == 0:
        return -2.0

    weighted_rating = float((valid["composite_rating"] * valid["dyn_w"]).sum() / tot_w)
    
    if peak_rating >= 3.5:
        weighted_rating = max(weighted_rating, peak_rating * 0.85)

    return float(weighted_rating)

player_ratings = df_profiles_clean.groupby("player_id").apply(calc_player_rating, include_groups=False).reset_index(name="proj_rating_base")

df_roster_ratings = df_roster.merge(player_ratings, on="player_id", how="left")
df_roster_ratings["proj_rating_base"] = df_roster_ratings["proj_rating_base"].fillna(-2.0)

def apply_aging_curve(row):
    r = row["proj_rating_base"]
    age = row["age"]
    if age <= 21: return r + 0.60
    elif age <= 23: return r + 0.30
    elif 24 <= age <= 29: return r
    elif 30 <= age <= 32: return r - 0.25
    elif 33 <= age <= 35: return r - 0.60
    else: return r - 1.10

df_roster_ratings["proj_rating"] = df_roster_ratings.apply(apply_aging_curve, axis=1)

# ---------------------------------------------------------
# 3. Takım Net Rating Hesabı (Z-Score Ölçekleme)
# ---------------------------------------------------------
print("3. Takim guc indeksleri hesaplaniyor...")

ROTATION_WEIGHTS = [0.20, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03]

raw_team_ratings = {}
team_avg_ages = {}
team_b2b_penalties = {}

for team, group in df_roster_ratings.groupby("team_id"):
    sorted_players = group.sort_values(by="proj_rating", ascending=False).reset_index(drop=True)
    
    t_rating = sum(
        (sorted_players.iloc[i]["proj_rating"] if i < len(sorted_players) else -2.0) * w
        for i, w in enumerate(ROTATION_WEIGHTS)
    )
    
    top5_ratings = [sorted_players.iloc[i]["proj_rating"] for i in range(min(5, len(sorted_players)))]
    if all(r > 0.8 for r in top5_ratings):
        t_rating += 0.60
        
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
team_net_ratings = (z_scores * 4.80).round(2).to_dict()

# ---------------------------------------------------------
# 4. Fikstür Dinamikleri & Maç Olasılıkları
# ---------------------------------------------------------
print("4. Fikstur dinamikleri ve seyahat yorgunluklari hesaplaniyor...")

team_pace_map = {}
if "pace" in df_team_prof.columns:
    df_team_prof["pace"] = pd.to_numeric(df_team_prof["pace"], errors="coerce")
    for t_id, grp in df_team_prof.groupby("team_id"):
        s26 = grp[grp["season"] == "2025-26"]["pace"].values
        s25 = grp[grp["season"] == "2024-25"]["pace"].values
        p26 = s26[0] if len(s26) > 0 and pd.notnull(s26[0]) else 99.5
        p25 = s25[0] if len(s25) > 0 and pd.notnull(s25[0]) else 99.5
        team_pace_map[t_id] = round(p26 * 0.65 + p25 * 0.35, 1)

team_hca_map = {}
for team_id, group in df_expanded.groupby("team_id"):
    s26 = group[group["season"] == "2025-26"]
    s25 = group[group["season"] == "2024-25"]
    h26 = s26["empirical_hca_diff"].values[0] if not s26.empty else 0.20
    h25 = s25["empirical_hca_diff"].values[0] if not s25.empty else 0.20
    team_hca_map[team_id] = round(float(np.clip(1.90 + (h26 * 0.65 + h25 * 0.35) * 4.0, 1.80, 4.10)), 2)

df_sched["game_date_dt"] = pd.to_datetime(df_sched["game_date"])
df_sched = df_sched.sort_values(by=["game_date_dt", "game_number"]).reset_index(drop=True)

unique_game_dates = pd.Series(sorted(df_sched["game_date_dt"].unique()))
feb_dates = unique_game_dates[unique_game_dates.dt.month == 2]
feb_gaps = feb_dates.diff().dt.days

asb_gap_indices = feb_gaps[feb_gaps >= 5].index
all_star_end_date = unique_game_dates.iloc[asb_gap_indices[0]] if not asb_gap_indices.empty else pd.to_datetime("2027-02-18")

team_last_date, team_last_city = {}, {}
team_consecutive_home = {t: 0 for t in team_net_ratings.keys()}
team_consecutive_road = {t: 0 for t in team_net_ratings.keys()}

match_spreads = []
match_sigmas = []

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

    h_net_base = team_net_ratings.get(h_team, 0.0)
    a_net_base = team_net_ratings.get(a_team, 0.0)
    h_net, a_net = h_net_base, a_net_base

    if g_date >= all_star_end_date:
        if h_net_base >= 2.0: h_net += 1.00
        elif h_net_base >= -0.5: h_net += 0.40
        elif h_net_base < -3.5: h_net -= 1.20

        if a_net_base >= 2.0: a_net += 1.00
        elif a_net_base >= -0.5: a_net += 0.40
        elif a_net_base < -3.5: a_net -= 1.20

    if not is_neutral:
        h_net += team_hca_map.get(h_team, 2.70)
        if team_consecutive_home.get(h_team, 0) >= 2: h_net += 0.40
        if any(pt in g_time for pt in ["7:30 PM", "8:00 PM", "8:30 PM", "9:00 PM", "9:30 PM"]):
            h_net += 0.30

    if h_b2b: h_net -= team_b2b_penalties.get(h_team, 2.50)

    if a_b2b:
        a_dyn_b2b = team_b2b_penalties.get(a_team, 2.50)
        if team_consecutive_road.get(a_team, 0) >= 2: a_dyn_b2b += 0.70
        a_net -= a_dyn_b2b

    if team_consecutive_road.get(a_team, 0) >= 3: a_net -= 0.65
    a_net -= a_travel_penalty

    h_pace = team_pace_map.get(h_team, 99.5)
    a_pace = team_pace_map.get(a_team, 99.5)
    game_sigma = BASE_SIGMA * np.sqrt(((h_pace + a_pace) / 2.0) / 100.0)

    spread = h_net - a_net
    match_spreads.append(spread)
    match_sigmas.append(game_sigma)

    team_last_date[h_team] = g_date
    team_last_date[a_team] = g_date
    team_last_city[h_team] = h_team
    team_last_city[a_team] = h_team

    team_consecutive_home[h_team] = team_consecutive_home.get(h_team, 0) + 1
    team_consecutive_road[h_team] = 0
    team_consecutive_road[a_team] = team_consecutive_road.get(a_team, 0) + 1
    team_consecutive_home[a_team] = 0

win_probs = norm.cdf(np.array(match_spreads) / np.array(match_sigmas))
df_sched["home_win_prob"] = win_probs

# ---------------------------------------------------------
# 5. 10.000 Sezonluk Monte Carlo Simülasyonu
# ---------------------------------------------------------
print(f"5. Monte Carlo Simulasyonu basliyor ({SIMULATION_ROUNDS:,} Sezon)...")

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
# 6. Raporlama & BigQuery Kayıt
# ---------------------------------------------------------
final_rows = []
for t in teams:
    wins_dist = sim_results[t]
    exp_wins = round(float(np.mean(wins_dist)), 1)
    
    # Vegas eklentisi
    v_line = vegas_map.get(t, 41.0)
    v_diff = round(exp_wins - v_line, 1)

    final_rows.append({
        "team_id": t,
        "season": "2026-27",
        "exp_wins": exp_wins,
        "vegas_total": v_line,
        "vegas_diff": v_diff,
        "net_rating_proj": team_net_ratings.get(t, 0.0),
        "proj_pace": team_pace_map.get(t, 99.5),
        "empirical_hca": team_hca_map.get(t, 2.70),
        "rot_avg_age": team_avg_ages.get(t, 26.0),
        "dyn_b2b_penalty": team_b2b_penalties.get(t, 2.50),
        "exp_losses": round(82.0 - exp_wins, 1),
        "win_std_dev": round(float(np.std(wins_dist)), 2),
        "playoff_prob_pct": round(float(np.mean(wins_dist >= 42.0) * 100), 1),
        "fifty_plus_win_pct": round(float(np.mean(wins_dist >= 50.0) * 100), 1)
    })

df_proj = pd.DataFrame(final_rows).sort_values(by="exp_wins", ascending=False).reset_index(drop=True)

print("\n" + "="*125)
print("  2026-27 NBA SEZON PROJEKSIYONLARI (STABIL ORIJINAL VERSİYON + VEGAS)")
print("="*125)
display_cols = ["team_id", "exp_wins", "vegas_total", "vegas_diff", "net_rating_proj", "playoff_prob_pct"]
print(df_proj[display_cols].to_string(index=False))

df_proj.to_csv("proj_team_wins_2026_27.csv", index=False)

if os.path.exists(KEY_PATH):
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True)
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.proj_team_wins_2026_27"
    client.load_table_from_dataframe(df_proj, t_ref, job_config=job_config).result()
    print(f"\n[BASARILI] BigQuery '{t_ref}' tablosu guncellendi!")