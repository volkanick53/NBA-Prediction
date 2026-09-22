'use client';

import { PlayerProp } from '@/lib/api';

interface Props {
  players: PlayerProp[];
  title: string;
  accentColor?: string;
}

const STAT_COLS = [
  { key: 'proj_pts',     label: 'PTS',  color: '#F7A230' },
  { key: 'proj_trb',     label: 'REB',  color: '#1D9BF0' },
  { key: 'proj_ast',     label: 'AST',  color: '#a78bfa' },
  { key: 'proj_fg3',     label: '3PM',  color: '#34d399' },
  { key: 'proj_stl',     label: 'STL',  color: '#fb923c' },
  { key: 'proj_blk',     label: 'BLK',  color: '#f472b6' },
  { key: 'proj_minutes', label: 'MIN',  color: '#64748b' },
];

function StatCell({ value, color }: { value: number; color: string }) {
  return (
    <td style={{ padding: '10px 10px', borderTop: '1px solid rgba(255,255,255,0.04)', textAlign: 'center' }}>
      <span style={{ fontSize: 14, fontWeight: 600, color, fontVariantNumeric: 'tabular-nums' }}>
        {value.toFixed(1)}
      </span>
    </td>
  );
}

export default function PlayerPropsTable({ players, title, accentColor = '#F7A230' }: Props) {
  if (!players || players.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: '24px', color: '#475569', fontSize: 14 }}>
        Oyuncu verisi bulunamadı.
      </div>
    );
  }

  // Dk'ya göre sırala (en fazla oynayan üstte)
  const sorted = [...players].sort((a, b) => b.proj_minutes - a.proj_minutes);

  return (
    <div>
      {/* Section Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{ width: 3, height: 18, borderRadius: 2, background: accentColor }} />
        <h3 style={{ fontSize: 13, fontWeight: 700, color: '#cbd5e1', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {title}
        </h3>
      </div>

      <div style={{ overflowX: 'auto', borderRadius: 10, border: '1px solid rgba(255,255,255,0.07)' }}>
        <table className="props-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.03)' }}>
              <th style={{ padding: '10px 12px', textAlign: 'left', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#475569' }}>
                OYUNCU
              </th>
              <th style={{ padding: '10px 8px', textAlign: 'center', fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#475569' }}>
                POS
              </th>
              {STAT_COLS.map(col => (
                <th
                  key={col.key}
                  style={{
                    padding: '10px 10px',
                    textAlign: 'center',
                    fontSize: 11,
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    color: col.color,
                    opacity: 0.8,
                  }}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((player, idx) => (
              <tr
                key={player.player_id}
                style={{
                  background: idx % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)',
                  transition: 'background 0.15s',
                }}
              >
                {/* Name */}
                <td style={{
                  padding: '10px 12px',
                  borderTop: '1px solid rgba(255,255,255,0.04)',
                  fontSize: 13,
                  fontWeight: 600,
                  color: '#e2e8f0',
                  whiteSpace: 'nowrap',
                }}>
                  {player.name}
                </td>

                {/* Position */}
                <td style={{
                  padding: '10px 8px',
                  borderTop: '1px solid rgba(255,255,255,0.04)',
                  textAlign: 'center',
                }}>
                  <span style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    background: 'rgba(255,255,255,0.07)',
                    color: '#94a3b8',
                    letterSpacing: '0.05em',
                  }}>
                    {player.position || '—'}
                  </span>
                </td>

                {/* Stats */}
                {STAT_COLS.map(col => (
                  <StatCell
                    key={col.key}
                    value={(player as unknown as Record<string, number>)[col.key] ?? 0}
                    color={col.color}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
