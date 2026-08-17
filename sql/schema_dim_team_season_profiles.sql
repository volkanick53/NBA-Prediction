CREATE TABLE IF NOT EXISTS `nba-analytics-503718.nba_analytics.dim_team_season_profiles` (
    team_id STRING,
    season STRING,
    wins INT64,
    losses INT64,
    pace FLOAT64,
    off_rtg FLOAT64,
    def_rtg FLOAT64,
    net_rtg FLOAT64,
    efg_pct FLOAT64,
    tov_pct FLOAT64,
    orb_pct FLOAT64,
    ft_rate FLOAT64,
    opp_efg_pct FLOAT64,
    opp_tov_pct FLOAT64,
    opp_drb_pct FLOAT64,
    opp_ft_rate FLOAT64,
    arena_name STRING,
    attendance INT64
)
CLUSTER BY team_id;