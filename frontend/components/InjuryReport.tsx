'use client';

import { InjuryEntry } from '@/lib/api';
import { AlertTriangle, AlertCircle, HelpCircle, CheckCircle } from 'lucide-react';

interface Props {
  injuries: InjuryEntry[];
  title: string;
}

function getStatusStyle(status: string) {
  const s = status.toLowerCase();
  if (s.includes('out') || s.includes('reserve') || s.includes('suspend') || s.includes('season')) {
    return { className: 'badge-out', Icon: AlertCircle, label: 'Out' };
  }
  if (s.includes('doubtful')) {
    return { className: 'badge-doubtful', Icon: AlertTriangle, label: 'Doubtful' };
  }
  if (s.includes('question')) {
    return { className: 'badge-question', Icon: HelpCircle, label: 'Questionable' };
  }
  return { className: 'badge-active', Icon: CheckCircle, label: status };
}

export default function InjuryReport({ injuries, title }: Props) {
  if (!injuries || injuries.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px', color: '#4ade80', fontSize: 13 }}>
        <CheckCircle size={14} />
        <span>Aktif sakatlık yok</span>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <div style={{ width: 3, height: 16, borderRadius: 2, background: '#f87171' }} />
        <h4 style={{ fontSize: 12, fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {title}
        </h4>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {injuries.map(inj => {
          const { className, Icon, label } = getStatusStyle(inj.status);
          return (
            <div
              key={inj.player_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '8px 12px',
                borderRadius: 8,
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              {/* Player Info */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '2px 5px',
                  borderRadius: 4,
                  background: 'rgba(255,255,255,0.07)',
                  color: '#64748b',
                }}>
                  {inj.position || '—'}
                </span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: inj.is_excluded ? '#f87171' : '#e2e8f0' }}>
                    {inj.name}
                  </div>
                  {inj.description && (
                    <div style={{ fontSize: 11, color: '#475569', marginTop: 1, maxWidth: 200 }}>
                      {inj.description.slice(0, 60)}{inj.description.length > 60 ? '…' : ''}
                    </div>
                  )}
                </div>
              </div>

              {/* Status Badge */}
              <div
                className={className}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '3px 8px',
                  borderRadius: 6,
                  fontSize: 11,
                  fontWeight: 700,
                  whiteSpace: 'nowrap',
                  flexShrink: 0,
                }}
              >
                <Icon size={10} />
                {label}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
