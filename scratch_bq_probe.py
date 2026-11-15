from google.cloud import bigquery
from google.oauth2 import service_account

cr = service_account.Credentials.from_service_account_file("nba-analytics-503718-9f3bbd399bc1.json")
c = bigquery.Client(credentials=cr, project="nba-analytics-503718")
ds = "nba-analytics-503718.nba_analytics"
for q in [
    f"select season, season_type, count(*) n, count(distinct game_id) g, min(game_date) a, max(game_date) b from `{ds}.fact_player_boxscores` group by 1,2 order by 1,2",
    f"select player_id, player_name, team_id, game_date, minutes, pts, trb, ast from `{ds}.fact_player_boxscores` where player_name like 'Nikola Joki%' order by game_date desc limit 3",
    f"select typeof_min, count(*) n from (select cast(minutes as string) typeof_min from `{ds}.fact_player_boxscores` limit 1000) group by 1 order by 2 desc limit 3",
]:
    for r in c.query(q).result():
        print(dict(r))
    print("--")
