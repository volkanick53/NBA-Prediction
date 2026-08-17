CREATE TABLE IF NOT EXISTS `nba-analytics-503718.nba_analytics.dim_player_season_profiles` (
    player_id STRING,
    player_name STRING,
    team_id STRING,
    season STRING,
    games_played INT64,
    minutes_played INT64,
    
    -- Advanced Metrics
    per FLOAT64,
    ts_pct FLOAT64,
    fg3a_per_fga_pct FLOAT64,
    fta_per_fga_pct FLOAT64,
    orb_pct FLOAT64,
    drb_pct FLOAT64,
    trb_pct FLOAT64,
    ast_pct FLOAT64,
    stl_pct FLOAT64,
    blk_pct FLOAT64,
    tov_pct FLOAT64,
    usg_pct FLOAT64,
    ows FLOAT64,
    dws FLOAT64,
    ws FLOAT64,
    ws_per_48 FLOAT64,
    obpm FLOAT64,
    dbpm FLOAT64,
    bpm FLOAT64,
    vorp FLOAT64,

    -- Per 36 Minutes
    pts_per_36 FLOAT64,
    trb_per_36 FLOAT64,
    ast_per_36 FLOAT64,
    stl_per_36 FLOAT64,
    blk_per_36 FLOAT64,
    tov_per_36 FLOAT64,
    fg3_per_36 FLOAT64,

    -- Per 100 Possessions Individual Ratings
    ind_off_rtg FLOAT64,
    ind_def_rtg FLOAT64,

    -- Shooting & Shot Creation Splits
    avg_dist_ft FLOAT64,
    pct_fga_2p FLOAT64,
    pct_fga_0_3ft FLOAT64,
    pct_fga_3_10ft FLOAT64,
    pct_fga_10_16ft FLOAT64,
    pct_fga_16_3pft FLOAT64,
    pct_fga_3p FLOAT64,
    fg_pct_0_3ft FLOAT64,
    fg_pct_3_10ft FLOAT64,
    fg_pct_10_16ft FLOAT64,
    fg_pct_16_3pft FLOAT64,
    pct_ast_2p FLOAT64,
    pct_ast_3p FLOAT64,
    pct_dunk_fga FLOAT64,
    pct_corner3_3pa FLOAT64,
    corner3_pct FLOAT64,

    -- Play-by-Play & Position Distribution
    pct_pg FLOAT64,
    pct_sg FLOAT64,
    pct_sf FLOAT64,
    pct_pf FLOAT64,
    pct_c FLOAT64,
    on_off_net_per_100 FLOAT64,
    tov_bad_pass INT64,
    tov_lost_ball INT64,
    shooting_fouls_drawn INT64
)
CLUSTER BY team_id, player_id;