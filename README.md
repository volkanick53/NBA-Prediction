# 🏀 NBA Analytics & Predictive Modeling Suite (End-to-End GCP Pipeline)

An enterprise-grade, serverless data engineering and predictive modeling pipeline built on **Google Cloud Platform (GCP)** and **BigQuery**. The system ingests, processes, and stores historical NBA data (2022–2026 seasons) across 6,600+ games and 100,000+ player records to simulate team standings and project player performance metrics.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    A[Basketball-Reference Data Source] -->|Python Scraping & Parsing| B[Docker Containerized ETL]
    B -->|Serverless Job Orchestration| C[GCP Cloud Run Jobs]
    C -->|Batch Upload| D[(Google BigQuery Data Warehouse)]
    D -->|Partitioned & Clustered Tables| E[Fact Tables: Boxscores, Games, Absences]
    E -->|SQL Window Functions & Feature Engineering| F[v_player_features View]
    F -->|ML Regression & Time-Series Projections| G[Predictive Suite: Top 100 & Team Standings]




🛠️ Tech Stack & Tooling
Cloud Platform: Google Cloud Platform (GCP)

Compute / ETL: GCP Cloud Run Jobs, Cloud Build, Google Artifact Registry, Docker

Data Warehouse: Google BigQuery (Partitioned by game_date, Clustered by team_id / player_id)

Languages & Libraries: Python 3.11, Pandas, BeautifulSoup4, Requests, NumPy

Data Modeling & SQL: BigQuery SQL, Analytical Window Functions (LAG, ROWS BETWEEN), DDL Views

Predictive Models: Weighted Time-Series Decay (MARCEL/Rolling Projection), Pythagorean Win Expectation

🗄️ Data Warehouse Schema (BigQuery)
1. fact_player_boxscores
Individual player performance per game (Basic & Advanced Box Scores).

player_id (STRING), player_name (STRING), team_id (STRING)

game_id (STRING), game_date (DATE), season (STRING), is_home (BOOL)

minutes (FLOAT), pts, trb, ast, stl, blk, tov, fg3

ts_pct (True Shooting %), usg_pct (Usage Rate %), off_rtg, def_rtg

2. fact_game_summaries
Aggregated game metrics, team scores, offensive ratings, and game pace.

game_id (STRING), game_date (DATE), season (STRING)

home_team (STRING), away_team (STRING), home_score (INT64), away_score (INT64)

pace (FLOAT - Standardized NBA Possessions per 48 min), home_off_rtg, away_off_rtg

3. fact_player_absences
Tracks player availability, injuries, rest management, and coach decisions.

game_id (STRING), game_date (DATE), player_id (STRING), status (DNP / INACTIVE), reason (STRING)

4. v_player_features (Analytical View)
Prevents data leakage by leveraging SQL analytical windowing for prior performance:

Dynamic rolling metrics (pts_avg_last_3, pts_avg_last_5, pts_avg_last_10, usg_avg_last_5)

Rest schedule & fatigue tracking (rest_days, is_back_to_back)

🚀 Predictive Engines
Top 100 Player Season Projections (models/project_top_100_players.py):

Multi-season weighted decay model predicting PTS, REB, AST, 3PM, TS%, and USG% for the 2026-27 season.

Team Win/Loss Simulator (models/project_team_standings.py):

Net Rating trajectory and conference standing simulation with an 82-game zero-sum league constraint (1,230 total regular season wins).