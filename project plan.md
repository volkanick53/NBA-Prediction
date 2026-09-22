# 🏀 NBA Machine Learning Prediction Platform & Web Dashboard — Master Project Spec

## 1. Project Role & Objective
You are a Senior Data Engineer, Machine Learning Specialist, and Full-Stack Architect.
Your task is to build an end-to-end, production-grade NBA daily game & player proposition prediction platform from scratch. 
The system does NOT rely on external betting odds (Vegas, etc.); it calculates 100% proprietary predictions using a multi-stage Machine Learning (XGBoost) architecture, ingests live daily rosters/injuries via ESPN APIs, and serves all predictions on a modern, interactive web dashboard.

---

## 2. Available Data Assets & Environment

1. **Historical Boxscore Database (GCP BigQuery):**
   - **Table:** `nba-analytics-503718.nba_analytics.fact_player_boxscores`
   - **Data Volume:** Past 5 complete NBA seasons (~150,000+ player-game rows).
   - **Credentials:** Service account JSON located at project root: `nba-analytics-503718-9f3bbd399bc1.json`
   - **Key Schema Columns:** `game_id`, `game_date`, `season`, `player_id`, `player_name`, `team_id`, `opponent_id`, `is_home`, `minutes`, `pts`, `trb`, `ast`, `stl`, `blk`, `tov`, `fga`, `fgm`, `fg3`, `fg3a`, `fta`, `ftm`, `orb`, `drb`, `plus_minus`, `ts_pct`, `usg_pct`, `off_rtg`, `def_rtg`, `bpm`.

2. **Live Data & Ingestion Specs (`basketball.md`):**
   - We have the complete public ESPN API specifications saved locally in `@basketball.md`.
   - Do NOT use web scrapers or HTML parsing. Exclusively use ESPN Site API endpoints:
     - `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` (Daily matchups & schedule)
     - `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries` (League-wide real-time injury status)
     - `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{id}/roster` (Live official active team rosters)
     - `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={id}` (Post-game verification & boxscore)

---

## 3. Core Modeling Philosophy & Non-Negotiable Rules

1. **NO Static Dictionaries or Clamping:**
   - Never hardcode static team ratings, pace, or ORtg/DRtg dictionaries in code.
   - Never artificially clamp or clamp-force game scores (e.g. forcing 110–115). All scores must emerge organically from team pace, offensive efficiency, opponent defense, and rest days.
2. **Zero Data Leakage:**
   - When engineering rolling features (last 5/10 games), strictly use chronological ordering and `.shift(1)`. The current game's statistics must NEVER be visible to the feature row.
3. **Dean Oliver Possession Formulation:**
   - Team possessions (Pace) must strictly follow:
     $$\text{Possessions} = \text{FGA} - \text{ORB} + \text{TOV} + (0.44 \times \text{FTA})$$
4. **240-Minute & USG% Conservation Law:**
   - A standard NBA game has strictly 240 team minutes (5 players × 48 mins).
   - Filter out players listed as "Out" or "Doubtful" from the live ESPN roster.
   - Dynamically allocate the 240 minutes to the remaining active rotation (approx. top 9–10 players).
   - Player point projections must NOT be calculated as unconstrained Per-36 sums (which leads to unrealistic 140+ point blowouts). Instead, calibrate scoring through a two-stage approach:
     - **Stage 1:** Predict Game Pace and Team Total Points via XGBoost.
     - **Stage 2:** Distribute expected points and props to active players using their projected minutes, rolling USG%, and efficiency relative to the predicted game tempo.

---

## 4. Multi-Stage Machine Learning Pipeline (Architecture B)

The prediction pipeline operates in two correlated stages:

### Stage 1: Match Level Models (XGBoost Regressors)
- **Model 1A (Game Pace):** Predicts total possessions in 48 minutes based on home/away rolling pace (last 5 & 10 games) and rest fatigue.
- **Model 1B (Team Points / Spread):** Predicts expected points for Home and Away teams based on rolling ORtg, opponent rolling DRtg, home court advantage, and net rest days.
- **Output:** Predicted Home Score, Predicted Away Score, Spread, Total Points (Over/Under), Home Win Probability (via Normal CDF on point difference).

### Stage 2: Player Prop Models (Specialized Regressors)
Using the Stage 1 predicted game pace and team total points as inputs:
- Predict: **PTS (Points)**, **TRB (Rebounds)**, **AST (Assists)**, **3PM (3-Pointers Made)**, **STL (Steals)**, **BLK (Blocks)**.
- Input Features: Player projected minutes, rolling 5/10-game per-minute rate, rolling USG%, rolling TS%, opponent defensive rating, and predicted game pace.

---

## 5. End-to-End Implementation Roadmap

### Phase 1: Feature Engineering (`etl/`)
- Connect to BigQuery using `nba-analytics-503718-9f3bbd399bc1.json`.
- Extract 5 seasons of `fact_player_boxscores` and cache locally in `data/raw_boxscores_5seasons.parquet`.
- Compute game-level team stats, pace, ORtg, and DRtg.
- Build leakage-free rolling features (EWMA / rolling mean for 5 and 10 games) + rest days for both teams and players.
- Save clean training datasets: `data/team_training_data.parquet` and `data/player_training_data.parquet`.

### Phase 2: Model Training & Evaluation (`models/`)
- Train XGBoost models for:
  1. `team_pace_model.json`
  2. `team_points_model.json`
  3. `player_pts_model.json`, `player_reb_model.json`, `player_ast_model.json`, `player_3pm_model.json`
- Perform time-series split cross-validation (train on earlier seasons, validate on the most recent season).
- Output evaluation metrics (RMSE, MAE, R²).

### Phase 3: Daily Live Inference Engine (`engine/`)
- CLI command: `python -m engine.predict_daily --date YYYYMMDD`
- Query ESPN API (`scoreboard`, `injuries`, `roster` as specified in `basketball.md`).
- Match live players with model registry IDs.
- Run Stage 1 & Stage 2 inference.
- Output clean structured JSON to `output/predictions_latest.json` and persist to BigQuery prediction history.

### Phase 4: Backend API (`api/`)
- Develop a high-performance **FastAPI** service:
  - `GET /api/games?date=YYYYMMDD`: Returns matchup cards, predicted score, win probabilities, spread, total.
  - `GET /api/game/{game_id}`: Returns in-depth matchup details, injury report, and player prop tables.
  - `GET /api/projections/season`: Returns full season team win projections and player season profiles.

### Phase 5: Modern Web Dashboard (`frontend/`)
- Modern, clean dashboard (Next.js / React with Tailwind CSS, Lucide icons, Dark Mode):
  - **Date Picker / Header:** Switch between yesterday's results (hit rate verification), today's live predictions, and upcoming schedule.
  - **Matchup Cards:** Team logos, match start time (converted to UTC+3 / Istanbul time), Predicted Score, Model Spread, Total Over/Under, Win Probability Bar.
  - **Match Detail Drawer / Modal:**
    - Live Injury Badge list (Out, Doubtful, Questionable).
    - Player Props Table (Name, Pos, Projected Mins, PTS, REB, AST, 3PM, STL, BLK).
    - Pace & Efficiency Breakdown.

---

## 6. How to Begin

Please review this specification and:
1. Confirm your understanding of the architecture, data sources, and constraints.
2. Outline the repository folder structure you plan to set up.
3. Start executing **Phase 1: Feature Engineering Pipeline**, writing clean, modular, and well-documented Python scripts.