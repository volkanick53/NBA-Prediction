const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface GameCard {
  event_id: string;
  home_team_id: number;
  away_team_id: number;
  home_team_name: string;
  away_team_name: string;
  home_team_abbr: string;
  away_team_abbr: string;
  home_team_logo: string;
  away_team_logo: string;
  game_date: string;
  status: string;
  pred_home_score: number;
  pred_away_score: number;
  pred_spread: number;
  pred_total: number;
  home_win_prob: number;
  pred_game_pace: number;
}

export interface PlayerProp {
  player_id: number;
  name: string;
  position: string;
  proj_minutes: number;
  proj_pts: number;
  proj_trb: number;
  proj_ast: number;
  proj_fg3: number;
  proj_stl: number;
  proj_blk: number;
}

export interface InjuryEntry {
  player_id: number;
  name: string;
  position: string;
  status: string;
  description: string;
  is_excluded: boolean;
}

export interface GameDetail extends GameCard {
  home_player_props: PlayerProp[];
  away_player_props: PlayerProp[];
  home_injuries: InjuryEntry[];
  away_injuries: InjuryEntry[];
}

export interface GamesResponse {
  date: string;
  count: number;
  games: GameCard[];
}

export async function fetchGames(date: string): Promise<GamesResponse> {
  const res = await fetch(`${API_BASE}/api/games?date=${date}`, {
    next: { revalidate: 300 }, // 5 dk cache
  });
  if (!res.ok) throw new Error(`Maçlar alınamadı: ${res.status}`);
  return res.json();
}

export async function fetchGameDetail(eventId: string): Promise<GameDetail> {
  const res = await fetch(`${API_BASE}/api/game/${eventId}`, {
    next: { revalidate: 120 },
  });
  if (!res.ok) throw new Error(`Maç detayı alınamadı: ${res.status}`);
  return res.json();
}
