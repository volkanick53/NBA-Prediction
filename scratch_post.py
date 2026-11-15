import db_dtypes  # noqa: F401  (registers 'dbdate' dtype)
import pandas as pd

df = pd.read_parquet("data/raw_boxscores_5seasons.parquet")
print(df["season_type"].value_counts().to_dict())
g = df[["game_id", "game_date", "season", "team_id"]].drop_duplicates()
g["game_date"] = pd.to_datetime(g["game_date"])
g = g.sort_values(["team_id", "game_date"])
g["idx"] = g.groupby(["team_id", "season"]).cumcount() + 1
print(g.groupby(["season", "team_id"])["idx"].max().groupby("season").describe()[["min", "50%", "max"]])
post = g[g["idx"] > 82]
print("postseason team-games:", len(post), "of", len(g))
# minutes of top players: regular vs post
m = df.merge(g[["game_id", "team_id", "idx"]], on=["game_id", "team_id"])
m["post"] = m["idx"] > 82
top = m[m["minutes"] > 25].groupby("post")[["minutes", "pts"]].mean()
print(top)
print(m.groupby("post")[["minutes"]].count())
print(g.groupby("season")["game_date"].apply(lambda s: s[g.loc[s.index, "idx"] == 82].max()))
