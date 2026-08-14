CREATE TABLE IF NOT EXISTS \
ba-analytics-503718.nba_analytics.fact_player_boxscores\ (
    player_id STRING,
    player_name STRING,
    team_id STRING,
    game_id STRING,
    game_date DATE,
    season STRING,
    season_type STRING,
    is_home BOOL,
    minutes FLOAT64,
    pts INT64,
    trb INT64,
    ast INT64,
    stl INT64,
    blk INT64,
    tov INT64,
    fg3 INT64,
    ts_pct FLOAT64,
    usg_pct FLOAT64,
    off_rtg INT64,
    def_rtg INT64
)
PARTITION BY game_date
CLUSTER BY team_id, player_id;
