CREATE OR REPLACE VIEW \
ba-analytics-503718.nba_analytics.v_player_features\ AS
WITH player_game_ordered AS (
    SELECT 
        p.player_id,
        p.player_name,
        p.team_id,
        p.game_id,
        p.game_date,
        p.season,
        p.season_type,
        p.is_home,
        p.minutes,
        p.pts,
        p.trb,
        p.ast,
        p.ts_pct,
        p.usg_pct,
        g.pace,
        LAG(p.game_date) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id) AS prev_game_date,
        ROUND(AVG(p.pts) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 2) AS pts_avg_last_3,
        ROUND(AVG(p.pts) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 2) AS pts_avg_last_5,
        ROUND(AVG(p.pts) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING), 2) AS pts_avg_last_10,
        ROUND(AVG(p.trb) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 2) AS trb_avg_last_5,
        ROUND(AVG(p.ast) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 2) AS ast_avg_last_5,
        ROUND(AVG(p.minutes) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 2) AS min_avg_last_5,
        ROUND(AVG(p.usg_pct) OVER(PARTITION BY p.player_id ORDER BY p.game_date, p.game_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 2) AS usg_avg_last_5
    FROM \
ba-analytics-503718.nba_analytics.fact_player_boxscores\ p
    LEFT JOIN \
ba-analytics-503718.nba_analytics.fact_game_summaries\ g ON p.game_id = g.game_id
)
SELECT 
    player_id, player_name, team_id, game_id, game_date, season, is_home,
    COALESCE(DATE_DIFF(game_date, prev_game_date, DAY), 7) AS rest_days,
    CASE WHEN DATE_DIFF(game_date, prev_game_date, DAY) = 1 THEN 1 ELSE 0 END AS is_back_to_back,
    pts_avg_last_3, pts_avg_last_5, pts_avg_last_10, trb_avg_last_5, ast_avg_last_5, min_avg_last_5, usg_avg_last_5,
    COALESCE(pace, 100.0) AS game_pace,
    pts AS target_pts, trb AS target_trb, ast AS target_ast
FROM player_game_ordered
WHERE pts_avg_last_5 IS NOT NULL;
