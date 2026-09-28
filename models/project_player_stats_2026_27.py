import os
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

KEY_PATH = "nba-analytics-503718-9f3bbd399bc1.json"
PROJECT_ID = "nba-analytics-503718"
DATASET_ID = "nba_analytics"

LEAGUE_PACE_BASE = 99.5
SHRINKAGE_MINUTES = 1200.0  # MARCELS lig ortalamasına regresyon sözde-dakikası

# ---------------------------------------------------------
# 1. BigQuery Tablolarını Yükle
# ---------------------------------------------------------
print("1. BigQuery verileri yukleniyor...")
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

df_roster = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_current_rosters_2026_27`").to_dataframe()
df_profiles_raw = client.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.dim_player_season_profiles`").to_dataframe()
df_team_proj = client.query(f"SELECT team_id, proj_pace FROM `{PROJECT_ID}.{DATASET_ID}.proj_team_wins_2026_27`").to_dataframe()

df_roster["age"] = pd.to_numeric(df_roster["age"], errors="coerce").fillna(25)

count_cols = [
    "pts", "trb", "ast", "stl", "blk", "tov", "pf", "orb", "drb",
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"
]

# Eğer ham toplamlar yoksa maç başı ortalamalardan toplam üret
for col in count_cols + ["mp", "games", "mpg", "bpm", "ws_per_48", "usg_pct", "age"]:
    if col in df_profiles_raw.columns:
        df_profiles_raw[col] = pd.to_numeric(df_profiles_raw[col], errors="coerce")

# Sayma istatistiklerinin toplamlarını garantiye al
for base, per_g in [("pts", "ppg"), ("trb", "rpg"), ("ast", "apg"), ("stl", "spg"), 
                     ("blk", "bpg"), ("tov", "topg"), ("fgm", "fgm_per_g"), 
                     ("fga", "fga_per_g"), ("fg3m", "fg3m_per_g"), ("fg3a", "fg3a_per_g"), 
                     ("ftm", "ftm_per_g"), ("fta", "fta_per_g")]:
    if base not in df_profiles_raw.columns and per_g in df_profiles_raw.columns:
        df_profiles_raw[base] = df_profiles_raw[per_g] * df_profiles_raw["games"]

# Takas olan oyuncuların en yüksek dakikalı ana satırını al
df_profiles_raw = df_profiles_raw.sort_values(by=["player_id", "season", "mp"], ascending=[True, True, False])
df_profiles = df_profiles_raw.drop_duplicates(subset=["player_id", "season"], keep="first").copy()
df_profiles = df_profiles[df_profiles["mp"] > 50].copy()

# ---------------------------------------------------------
# 2. Lig Geneli Per-36 Ortalamaları (Regresyon Havuzu)
# ---------------------------------------------------------
print("2. Lig geneli Per-36 referans havuzu hesaplaniyor...")

league_tot_mp = df_profiles[df_profiles["season"] == "2025-26"]["mp"].sum()
league_per36 = {}

for col in count_cols:
    if col in df_profiles.columns:
        tot_stat = df_profiles[df_profiles["season"] == "2025-26"][col].sum()
        league_per36[col] = (tot_stat / league_tot_mp) * 36.0 if league_tot_mp > 0 else 0.0

# ---------------------------------------------------------
# 3. 3-Yıllık MARCELS Per-36 Ağırlıklandırması ve Shrinkage
# ---------------------------------------------------------
print("3. MARCELS 3-Yillik agirliklandirma ve lig ortalamasina regresyon calisiyor...")

season_weights = {"2025-26": 5.0, "2024-25": 3.0, "2023-24": 1.0}
df_profiles["s_w"] = df_profiles["season"].map(season_weights).fillna(0.5)

player_marcels_per36 = {}

for p_id, grp in df_profiles.groupby("player_id"):
    tot_pseudo_mp = (grp["mp"] * grp["s_w"]).sum()
    if tot_pseudo_mp == 0:
        continue

    avg_hist_games = (grp["games"] * grp["s_w"]).sum() / grp["s_w"].sum()
    avg_hist_mpg = (grp["mp"] * grp["s_w"]).sum() / (grp["games"] * grp["s_w"]).sum()
    
    p_p36 = {
        "hist_games": avg_hist_games,
        "hist_mpg": avg_hist_mpg,
        "tot_pseudo_mp": tot_pseudo_mp,
        "bpm": (grp["bpm"].fillna(0.0) * grp["s_w"]).sum() / grp["s_w"].sum() if "bpm" in grp.columns else 0.0,
        "ws_per_48": (grp["ws_per_48"].fillna(0.09) * grp["s_w"]).sum() / grp["s_w"].sum() if "ws_per_48" in grp.columns else 0.09,
        "usg_pct": (grp["usg_pct"].fillna(18.0) * grp["s_w"]).sum() / grp["s_w"].sum() if "usg_pct" in grp.columns else 18.0
    }

    for col in count_cols:
        if col in grp.columns:
            # Oyuncunun 3 yıllık ağırlıklı ham Per 36 değeri
            raw_player_stat = (grp[col] * grp["s_w"]).sum()
            raw_p36 = (raw_player_stat / tot_pseudo_mp) * 36.0
            
            # MARCELS Shrinkage: Oyuncu Puanı + 1200 Dk Lig Ortalaması
            lg_stat36 = league_per36.get(col, 0.0)
            shrunk_p36 = (raw_p36 * tot_pseudo_mp + lg_stat36 * SHRINKAGE_MINUTES) / (tot_pseudo_mp + SHRINKAGE_MINUTES)
            p_p36[col] = shrunk_p36
        else:
            p_p36[col] = league_per36.get(col, 0.0)

    player_marcels_per36[p_id] = p_p36

# ---------------------------------------------------------
# 4. Yaş Eğrisi, Dinamik Maç Sayısı ve Maç Başına Dönüşüm
# ---------------------------------------------------------
print(f"4. {len(df_roster)} oyuncu icin 2026-27 Per-36 ve Mac Basi projeksiyonlari olusturuluyor...")

team_pace_dict = df_team_proj.set_index("team_id")["proj_pace"].to_dict() if not df_team_proj.empty else {}
final_players = []

for idx, row in df_roster.iterrows():
    p_id = row["player_id"]
    p_name = row["player_name"]
    team_id = row["team_id"]
    age = int(row["age"])

    t_pace = team_pace_dict.get(team_id, LEAGUE_PACE_BASE)
    pace_factor = t_pace / LEAGUE_PACE_BASE

    if p_id in player_marcels_per36:
        m = player_marcels_per36[p_id]
        hist_games = m["hist_games"]
        base_mpg = m["hist_mpg"]
        base_bpm = m["bpm"]
        base_ws48 = m["ws_per_48"]
        base_usg = m["usg_pct"]
    else:
        # Çaylak / Yeni Giren Oyuncular
        hist_games = 78.0 if age <= 23 else 68.0
        base_mpg = 16.0
        base_bpm = -1.5
        base_ws48 = 0.050
        base_usg = 16.0
        m = {col: league_per36.get(col, 0.0) * 0.85 for col in count_cols}

    # A. Dinamik Maç Sayısı (proj_games)
    is_superstar = (base_usg >= 27.0 or base_bpm >= 3.5 or m.get("pts", 10.0) >= 22.0)

    if age < 25:
        # Gençler dinlendirilmez (80-82 Maç)
        exp_games = min(82, int(round(hist_games + 5.0))) if hist_games >= 68 else min(80, int(round(hist_games + 8.0)))
    elif 25 <= age <= 29:
        # Prime dönem (78-82 Maç)
        exp_games = min(82, int(round(hist_games + 3.0))) if hist_games >= 72 else int(round(hist_games + 4.0))
    elif 30 <= age <= 33:
        exp_games = int(np.clip(round(hist_games), 58, 74)) if is_superstar else int(np.clip(round(hist_games + 2.0), 64, 80))
    else:
        exp_games = int(np.clip(round(hist_games - 3.0), 50, 68)) if is_superstar else int(np.clip(round(hist_games), 55, 75))

    exp_games = int(np.clip(exp_games, 25, 82))

    # B. MARCELS Ampirik Yaş Eğrisi (Peak: 27 Yaş)
    if age <= 22:
        age_rate_mult = 1.06
        mp_adj = 2.0
    elif age <= 24:
        age_rate_mult = 1.03
        mp_adj = 1.0
    elif 25 <= age <= 28:
        age_rate_mult = 1.00
        mp_adj = 0.0
    elif 29 <= age <= 31:
        age_rate_mult = 0.98
        mp_adj = -0.8
    elif 32 <= age <= 34:
        age_rate_mult = 0.94
        mp_adj = -2.0
    else:
        age_rate_mult = 0.88
        mp_adj = -3.5

    proj_mpg = round(float(np.clip(base_mpg + mp_adj, 5.0, 37.0)), 1)
    min_ratio = (proj_mpg / 36.0) * pace_factor

    # C. Per-36 ve Maç Başı İstatistik Üretimi
    p36_pts = m.get("pts", 0.0) * age_rate_mult
    p36_trb = m.get("trb", 0.0) * (age_rate_mult if age <= 28 else 0.98)
    p36_ast = m.get("ast", 0.0) * (1.02 if age <= 29 else 0.98)
    p36_stl = m.get("stl", 0.0) * (age_rate_mult if age <= 27 else 0.95)
    p36_blk = m.get("blk", 0.0) * (age_rate_mult if age <= 27 else 0.93)
    p36_tov = m.get("tov", 0.0) * (0.97 if age > 24 else 1.03)

    p36_fga = m.get("fga", 0.0) * age_rate_mult
    p36_fgm = m.get("fgm", 0.0) * age_rate_mult
    p36_fg3a = m.get("fg3a", 0.0) * (1.03 if age <= 26 else 1.0)
    p36_fg3m = m.get("fg3m", 0.0) * (1.03 if age <= 26 else 1.0)
    p36_fta = m.get("fta", 0.0) * age_rate_mult
    p36_ftm = m.get("ftm", 0.0) * age_rate_mult

    # Maç Başı Değerler (Per Game)
    ppg = round(p36_pts * min_ratio, 1)
    rpg = round(p36_trb * min_ratio, 1)
    apg = round(p36_ast * min_ratio, 1)
    spg = round(p36_stl * min_ratio, 1)
    bpg = round(p36_blk * min_ratio, 1)
    topg = round(p36_tov * min_ratio, 1)

    fga = round(p36_fga * min_ratio, 1)
    fgm = round(p36_fgm * min_ratio, 1)
    fg3a = round(p36_fg3a * min_ratio, 1)
    fg3m = round(p36_fg3m * min_ratio, 1)
    fta = round(p36_fta * min_ratio, 1)
    ftm = round(p36_ftm * min_ratio, 1)

    # Yüzdeler (Bileşenlerden Hesaplanır)
    fg_pct = round(float(np.clip(fgm / fga if fga > 0 else 0.450, 0.350, 0.720)), 3)
    fg3_pct = round(float(np.clip(fg3m / fg3a if fg3a > 0 else 0.340, 0.220, 0.450)), 3)
    ft_pct = round(float(np.clip(ftm / fta if fta > 0 else 0.750, 0.500, 0.950)), 3)

    efg_pct = round(float(np.clip((fgm + 0.5 * fg3m) / fga if fga > 0 else 0.510, 0.380, 0.750)), 3)
    ts_denom = 2 * (fga + 0.44 * fta)
    ts_pct = round(float(np.clip(ppg / ts_denom if ts_denom > 0 else 0.540, 0.450, 0.700)), 3)

    proj_bpm = round(base_bpm + (0.4 if age <= 23 else (-0.5 if age >= 33 else 0.0)), 1)
    proj_ws48 = round(float(np.clip(base_ws48 * (1.02 if age <= 25 else 0.98), -0.050, 0.320)), 3)

    final_players.append({
        "player_id": p_id,
        "player_name": p_name,
        "team_id": team_id,
        "season": "2026-27",
        "age": age,
        "proj_games": exp_games,
        "proj_mpg": proj_mpg,
        "proj_ppg": ppg,
        "proj_rpg": rpg,
        "proj_apg": apg,
        "proj_spg": spg,
        "proj_bpg": bpg,
        "proj_topg": topg,
        "proj_fgm": fgm,
        "proj_fga": fga,
        "proj_fg_pct": fg_pct,
        "proj_fg3m": fg3m,
        "proj_fg3a": fg3a,
        "proj_fg3_pct": fg3_pct,
        "proj_ftm": ftm,
        "proj_fta": fta,
        "proj_ft_pct": ft_pct,
        "proj_efg_pct": efg_pct,
        "proj_ts_pct": ts_pct,
        "proj_p36_pts": round(p36_pts, 1),
        "proj_p36_trb": round(p36_trb, 1),
        "proj_p36_ast": round(p36_ast, 1),
        "proj_bpm": proj_bpm,
        "proj_ws_per_48": proj_ws48
    })

df_final = pd.DataFrame(final_players).sort_values(by=["proj_ppg", "proj_mpg"], ascending=False).reset_index(drop=True)

# ---------------------------------------------------------
# 5. Raporlama ve BigQuery Kayıt
# ---------------------------------------------------------
print("\n" + "="*125)
print("  2026-27 NBA MARCELS (PER 36 TEMELLI) OYUNCU PROJEKSIYONLARI (TOP 35)")
print("="*125)
display_cols = ["player_name", "team_id", "age", "proj_games", "proj_mpg", "proj_ppg", "proj_p36_pts", "proj_rpg", "proj_apg", "proj_ts_pct", "proj_bpm"]
print(df_final[display_cols].head(35).to_string(index=False))

df_final.to_csv("proj_player_stats_2026_27.csv", index=False)

if os.path.exists(KEY_PATH):
    job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True)
    t_ref = f"{PROJECT_ID}.{DATASET_ID}.proj_player_stats_2026_27"
    client.load_table_from_dataframe(df_final, t_ref, job_config=job_config).result()
    print(f"\n[BASARILI] BigQuery '{t_ref}' tablosu Per-36 MARCELS modeli ile guncellendi!")