CREATE TABLE IF NOT EXISTS \
ba-analytics-503718.nba_analytics.fact_player_absences\ (
    game_id STRING,
    game_date DATE,
    season STRING,
    team_id STRING,
    player_id STRING,
    player_name STRING,
    status STRING,
    reason STRING
)
PARTITION BY game_date
CLUSTER BY team_id, status;
