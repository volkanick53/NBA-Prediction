# 🏀 NBA Analytics & Predictive Modeling Suite (End-to-End GCP Pipeline)

An end-to-end data engineering and predictive modeling pipeline built on **Google Cloud Platform (GCP)** and **BigQuery**. The system ingests, models, and analyzes historical NBA data across 6,600+ games and 100,000+ player records to simulate team standings and project player performance metrics for the 2026-27 season.

---

## ⚡ Pipeline & Data Flow

1. **Data Ingestion:** Automated scraping pipelines parse box scores, game metrics, and player absence/injury data from Basketball-Reference.
2. **Serverless ETL (Cloud Run Jobs):** Containerized Python microservices run on GCP Cloud Run Jobs with automatic retry and rate-limiting controls.
3. **Data Warehouse (BigQuery):** Data is modeled into date-partitioned and clustered Fact tables to optimize analytical query costs and speed.
4. **Feature Engineering (SQL Views):** Analytical SQL window functions (`LAG`, `ROWS BETWEEN`) generate dynamic rolling metrics and rest day context without data leakage.
5. **Predictive Modeling:** Multi-season weighted projections and Net Rating simulations produce player-level projections and conference win/loss standings.

---

## 🛠️ Tech Stack

* **Cloud:** Google Cloud Platform (GCP)
* **Compute & Containerization:** Cloud Run Jobs, Cloud Build, Artifact Registry, Docker
* **Data Warehouse:** Google BigQuery (Partitioned & Clustered Fact Tables)
* **Languages & Frameworks:** Python 3.11, Pandas, NumPy, BeautifulSoup4, Requests
* **Data Modeling:** BigQuery SQL, Analytical Window Functions
* **Machine Learning / Statistics:** Weighted Time-Series Decay (MARCEL/Rolling Projection), Pythagorean Win Expectation

---

## 🗄️ BigQuery Data Warehouse Schema

### `fact_player_boxscores`
Tracks individual player box score metrics per game (Basic & Advanced).
* `player_id`, `player_name`, `team_id`, `game_id`, `game_date`, `season`, `is_home`
* `minutes`, `pts`, `trb`, `ast`, `stl`, `blk`, `tov`, `fg3`
* `ts_pct` (True Shooting %), `usg_pct` (Usage Rate %), `off_rtg`, `def_rtg`

### `fact_game_summaries`
Aggregated game metrics, pace, and team efficiency ratings.
* `game_id`, `game_date`, `season`, `home_team`, `away_team`, `home_score`, `away_score`
* `pace` (NBA Possessions per 48 min), `home_off_rtg`, `away_off_rtg`

### `fact_player_absences`
Tracks player availability, injuries, rest management, and coach decisions (DNP).
* `game_id`, `game_date`, `season`, `team_id`, `player_id`, `player_name`, `status`, `reason`

### `v_player_features` (Analytical View)
Prevents data leakage by leveraging SQL analytical windowing for prior performance:
* Dynamic rolling metrics (`pts_avg_last_3`, `pts_avg_last_5`, `pts_avg_last_10`, `usg_avg_last_5`)
* Rest schedule & fatigue tracking (`rest_days`, `is_back_to_back`)

---

## 🚀 Predictive Engines

1. **Top 100 Player Season Projections (`models/project_top_100_players.py`):**
   * Multi-season weighted decay model predicting `PTS`, `REB`, `AST`, `3PM`, `TS%`, and `USG%` for regular starters and high-usage stars.
2. **Team Win/Loss Simulator (`models/project_team_standings.py`):**
   * Net Rating trajectory and conference standings simulation with an 82-game zero-sum league constraint (1,230 total regular season wins).

---

