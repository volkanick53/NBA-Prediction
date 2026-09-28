import json
with open("output/predictions_20261020.json", encoding="utf-8") as f:
    d = json.load(f)

print("=== MACLAR ===")
for g in d["games"]:
    print(
        g["away_team"], "@", g["home_team"],
        "|", g["match_time_tr"],
        "| Ev:", g["proj_home_score"], "Dep:", g["proj_away_score"],
        "| Total:", g["proj_total"],
        "| HWin:", g["home_win_prob"], "%",
        "| Kazanan:", g["predicted_winner"]
    )

print()
print("=== OYUNCU PROPS (ilk 20) ===")
print(f"{'Oyuncu':<25} {'Takim':<5} {'Pos':<3} {'Min':>4} | {'PTS':>5} {'REB':>4} {'AST':>4} {'3PM':>4} {'STL':>4} {'BLK':>4}")
print("-" * 75)
for p in d["player_props"][:20]:
    print(
        f"{p['player_name']:<25} {p['team_id']:<5} {p['position']:<3} {p['projected_mins']:>4.0f} |"
        f" {p['proj_pts']:>5.1f} {p['proj_trb']:>4.1f} {p['proj_ast']:>4.1f}"
        f" {p['proj_fg3']:>4.1f} {p['proj_stl']:>4.1f} {p['proj_blk']:>4.1f}"
    )

print()
print(f"Toplam mac: {len(d['games'])} | Toplam oyuncu: {len(d['player_props'])}")
