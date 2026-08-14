CREATE TABLE IF NOT EXISTS \
ba-analytics-503718.nba_analytics.fact_game_summaries\ (
    game_id STRING,
    game_date DATE,
    season STRING,
    season_type STRING,
    home_team STRING,
    away_team STRING,
    home_score INT64,
    away_score INT64,
    pace FLOAT64,
    home_off_rtg FLOAT64,
    away_off_rtg FLOAT64
)
PARTITION BY game_date
CLUSTER BY home_team, away_team;
