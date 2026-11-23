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

## 🚀 Daily Game & Player Prediction Engine (v3)

```bash
python -m engine.live_engine                  # today's slate (US Eastern date)
python -m engine.live_engine --days-ahead 1   # tomorrow's slate
python models/train.py                        # retrain + full validation report
```

**Data:** BigQuery `fact_player_boxscores` (5 completed seasons) + the current season from the
ESPN API, written back to BigQuery (`espn_player_boxscores`, `espn_game_results`,
`dim_player_id_map`, unified view `v_player_games`). No betting lines or third-party
projections are used anywhere.

**Model** (`engine/`):
1. *Player state* (`features.py`) – for every player, per-minute production of each box stat
   (incl. on-court plus-minus), exponentially weighted over his own games across seasons and
   shrunk toward an "unknown player" prior; projected minutes = EWMA of minutes played.
2. *Tonight's lineups* – ESPN roster minus Out/Doubtful players; expected minutes =
   P(plays) × minutes-when-playing, fitted to 240 (bench absorbs any excess) →
   projected team box score.
3. *Team ratings* (`team_ratings.py`) – opponent-adjusted offense/defense/pace by
   recency-weighted ridge regression; prior = last season's final rating regressed 50%.
4. *Game model* (`game_model.py`) – ridge regressions fitted on real final scores:
   margin and total from lineup box-score differences/sums + ratings + back-to-backs;
   in-season scoring-level correction for totals; P(win) = Φ(margin / σ).
5. *Player props* – gradient boosting on rates × projected minutes + team/opponent context
   (or plain rate × minutes where boosting did not win on the validation season).

**Validation** (`models/train.py`, report in `models/evaluation_report.txt`): strictly
walk-forward (features use only earlier games), hyper-parameters chosen on 2024-25,
final fit on 2022-25, tested on the untouched 2025-26 season and on 2026-27 to date.

**Tracking:** each run upserts pre-tipoff predictions into `pred_games` /
`pred_player_props`; views `v_game_prediction_results` and `v_player_prop_results`
grade them against real results as games finish.

> Season-long projections (`models/simulate_2026_27_season.py`, `/api/projections/season`)
> are a separate, older pipeline and still use Vegas win totals as calibration.
