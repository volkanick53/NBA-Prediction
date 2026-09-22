'use client';

import { GameDetail } from '@/lib/api';
import PlayerPropsTable from './PlayerPropsTable';
import InjuryReport from './InjuryReport';
import { X, Zap, BarChart2, Target } from 'lucide-react';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';

interface Props {
  game: GameDetail;
  onClose: () => void;
}

function StatBox({ label, value, color, icon: Icon }: { label: string; value: string; color: string; icon?: React.ElementType }) {
  return (
    <div style={{
      padding: '14px 16px',
      borderRadius: 12,
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.07)',
      textAlign: 'center',
    }}>
      {Icon && <Icon size={14} color={color} style={{ margin: '0 auto 4px' }} />}
      <div style={{ fontSize: 20, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.08em', marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

export default function GameDetailModal({ game, onClose }: Props) {
  const homePct = Math.round(game.home_win_prob * 100);
  const awayPct = 100 - homePct;
  const spread = game.pred_spread;

  // Radar chart data (Team efficiency comparison)
  const radarData = [
    { stat: 'O/U', home: game.pred_total / 2, away: game.pred_total / 2 },
    { stat: 'Pace', home: game.pred_game_pace, away: game.pred_game_pace },
    { stat: 'Score', home: game.pred_home_score, away: game.pred_away_score },
  ];

  return (
    <div
      className="modal-backdrop"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          background: '#0D1525',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 20,
          width: '100%',
          maxWidth: 860,
          maxHeight: '92vh',
          overflowY: 'auto',
          position: 'relative',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '20px 24px 16px',
          borderBottom: '1px solid rgba(255,255,255,0.07)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'sticky',
          top: 0,
          background: '#0D1525',
          zIndex: 10,
        }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
              {game.home_team_name} vs {game.away_team_name}
            </h2>
            <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>
              Maç Analiz Raporu • AI Tahmin
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              padding: 8,
              borderRadius: 10,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.1)',
              color: '#64748b',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <X size={18} />
          </button>
        </div>

        <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 24 }}>

          {/* Score Banner */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr auto 1fr',
            alignItems: 'center',
            gap: 16,
            padding: '20px',
            borderRadius: 14,
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.07)',
          }}>
            {/* Home */}
            <div style={{ textAlign: 'center' }}>
              {game.home_team_logo && (
                <img src={game.home_team_logo} alt={game.home_team_abbr} width={56} height={56}
                  style={{ objectFit: 'contain', marginBottom: 8 }} />
              )}
              <div style={{ fontSize: 13, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {game.home_team_abbr} <span style={{ color: '#1D9BF0', fontSize: 11 }}>• EV</span>
              </div>
              <div style={{ fontSize: 44, fontWeight: 900, color: '#1D9BF0', fontVariantNumeric: 'tabular-nums' }}>
                {game.pred_home_score.toFixed(0)}
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#22c55e' }}>{homePct}% Kazanma</div>
            </div>

            {/* Divider */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
              <Zap size={20} color="#F7A230" />
              <span style={{ fontSize: 13, fontWeight: 700, color: '#475569', letterSpacing: '0.15em' }}>VS</span>
            </div>

            {/* Away */}
            <div style={{ textAlign: 'center' }}>
              {game.away_team_logo && (
                <img src={game.away_team_logo} alt={game.away_team_abbr} width={56} height={56}
                  style={{ objectFit: 'contain', marginBottom: 8 }} />
              )}
              <div style={{ fontSize: 13, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {game.away_team_abbr} <span style={{ color: '#F7A230', fontSize: 11 }}>• DEP</span>
              </div>
              <div style={{ fontSize: 44, fontWeight: 900, color: '#F7A230', fontVariantNumeric: 'tabular-nums' }}>
                {game.pred_away_score.toFixed(0)}
              </div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#F7A230' }}>{awayPct}% Kazanma</div>
            </div>
          </div>

          {/* Key Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            <StatBox label="Spread (Home)" value={`${spread > 0 ? '+' : ''}${spread.toFixed(1)}`} color={spread >= 0 ? '#4ade80' : '#f87171'} icon={Target} />
            <StatBox label="Over/Under" value={game.pred_total.toFixed(1)} color="#F7A230" icon={BarChart2} />
            <StatBox label="Game Pace" value={`${game.pred_game_pace.toFixed(1)}`} color="#1D9BF0" icon={Zap} />
          </div>

          {/* Injury Reports */}
          {(game.home_injuries?.length > 0 || game.away_injuries?.length > 0) && (
            <div>
              <h3 style={{ fontSize: 13, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                🏥 Sakatlık Raporu
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <InjuryReport injuries={game.home_injuries} title={`${game.home_team_abbr} Sakatlıklar`} />
                <InjuryReport injuries={game.away_injuries} title={`${game.away_team_abbr} Sakatlıklar`} />
              </div>
            </div>
          )}

          {/* Player Props */}
          <div>
            <h3 style={{ fontSize: 13, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 16 }}>
              📊 Oyuncu Prop Tahminleri
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <PlayerPropsTable
                players={game.home_player_props}
                title={`${game.home_team_name} (Ev Sahibi)`}
                accentColor="#1D9BF0"
              />
              <PlayerPropsTable
                players={game.away_player_props}
                title={`${game.away_team_name} (Deplasman)`}
                accentColor="#F7A230"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
