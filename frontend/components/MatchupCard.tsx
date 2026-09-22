'use client';

import { GameCard } from '@/lib/api';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface Props {
  game: GameCard;
  onClick: () => void;
}

function formatTime(isoDate: string): string {
  try {
    const d = new Date(isoDate);
    // UTC+3 (Istanbul)
    return d.toLocaleTimeString('tr-TR', {
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'Europe/Istanbul',
    });
  } catch {
    return '--:--';
  }
}

function WinProbBar({ homeProb }: { homeProb: number }) {
  const homePct = Math.round(homeProb * 100);
  const awayPct = 100 - homePct;
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1" style={{ color: '#94a3b8' }}>
        <span style={{ color: '#1D9BF0', fontWeight: 600 }}>{homePct}%</span>
        <span style={{ fontSize: 10, color: '#64748b' }}>WIN PROB</span>
        <span style={{ color: '#F7A230', fontWeight: 600 }}>{awayPct}%</span>
      </div>
      <div style={{
        height: 5,
        borderRadius: 3,
        background: 'rgba(255,255,255,0.08)',
        overflow: 'hidden',
      }}>
        <div style={{
          width: `${homePct}%`,
          height: '100%',
          background: 'linear-gradient(90deg, #1D9BF0, #3b82f6)',
          borderRadius: 3,
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  );
}

function TeamDisplay({
  logo,
  abbr,
  name,
  score,
  align,
}: {
  logo: string;
  abbr: string;
  name: string;
  score: number;
  align: 'left' | 'right';
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: align === 'left' ? 'row' : 'row-reverse',
        alignItems: 'center',
        gap: 12,
        flex: 1,
      }}
    >
      {/* Team Logo */}
      <div style={{
        width: 52,
        height: 52,
        borderRadius: '50%',
        background: 'rgba(255,255,255,0.06)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        flexShrink: 0,
        border: '1px solid rgba(255,255,255,0.1)',
      }}>
        {logo ? (
          <img src={logo} alt={abbr} width={40} height={40} style={{ objectFit: 'contain' }} />
        ) : (
          <span style={{ fontSize: 16, fontWeight: 800, color: '#94a3b8' }}>{abbr}</span>
        )}
      </div>

      {/* Team Name + Score */}
      <div style={{ textAlign: align }}>
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {abbr}
        </div>
        <div style={{ fontSize: 13, color: '#cbd5e1', fontWeight: 500, marginTop: 1 }}>
          {name.replace('Los Angeles', 'LA').replace('Golden State', 'GS')}
        </div>
        <div style={{
          fontSize: 26,
          fontWeight: 800,
          color: '#f1f5f9',
          lineHeight: 1.1,
          marginTop: 2,
          fontVariantNumeric: 'tabular-nums',
        }}>
          {score.toFixed(0)}
        </div>
      </div>
    </div>
  );
}

export default function MatchupCard({ game, onClick }: Props) {
  const spread = game.pred_spread;
  const isHomePositive = spread > 0;

  return (
    <button
      onClick={onClick}
      className="glass-card"
      style={{
        width: '100%',
        textAlign: 'left',
        padding: '20px',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      {/* Header: Time + Status */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, color: '#64748b', fontWeight: 500 }}>
          🕐 {formatTime(game.game_date)} (TR)
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {game.status === 'STATUS_IN_PROGRESS' && <div className="live-dot" />}
          <span style={{
            fontSize: 11,
            fontWeight: 600,
            color: game.status === 'STATUS_FINAL' ? '#4ade80' : game.status === 'STATUS_IN_PROGRESS' ? '#22c55e' : '#64748b',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
          }}>
            {game.status === 'STATUS_FINAL' ? 'Final' : game.status === 'STATUS_IN_PROGRESS' ? 'Live' : 'Scheduled'}
          </span>
        </div>
      </div>

      {/* Teams + Scores */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <TeamDisplay
          logo={game.home_team_logo}
          abbr={game.home_team_abbr}
          name={game.home_team_name}
          score={game.pred_home_score}
          align="left"
        />

        {/* VS Divider */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: '#475569', letterSpacing: '0.1em' }}>VS</span>
          <Activity size={14} color="#334155" />
        </div>

        <TeamDisplay
          logo={game.away_team_logo}
          abbr={game.away_team_abbr}
          name={game.away_team_name}
          score={game.pred_away_score}
          align="right"
        />
      </div>

      {/* Win Probability */}
      <WinProbBar homeProb={game.home_win_prob} />

      {/* Stats Row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr',
        gap: 8,
        paddingTop: 12,
        borderTop: '1px solid rgba(255,255,255,0.06)',
      }}>
        {/* Spread */}
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
            Spread
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 3 }}>
            {isHomePositive
              ? <TrendingUp size={12} color="#4ade80" />
              : <TrendingDown size={12} color="#f87171" />}
            <span style={{
              fontSize: 15,
              fontWeight: 700,
              color: isHomePositive ? '#4ade80' : '#f87171',
            }}>
              {spread > 0 ? '+' : ''}{spread.toFixed(1)}
            </span>
          </div>
          <div style={{ fontSize: 10, color: '#334155', marginTop: 2 }}>HOME</div>
        </div>

        {/* Total */}
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
            O/U Total
          </div>
          <span style={{ fontSize: 15, fontWeight: 700, color: '#F7A230' }}>
            {game.pred_total.toFixed(1)}
          </span>
          <div style={{ fontSize: 10, color: '#334155', marginTop: 2 }}>POINTS</div>
        </div>

        {/* Pace */}
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 10, color: '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>
            Pace
          </div>
          <span style={{ fontSize: 15, fontWeight: 700, color: '#1D9BF0' }}>
            {game.pred_game_pace.toFixed(1)}
          </span>
          <div style={{ fontSize: 10, color: '#334155', marginTop: 2 }}>POSS/48</div>
        </div>
      </div>
    </button>
  );
}
