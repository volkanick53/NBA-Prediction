'use client';

import { useState, useCallback } from 'react';
import { GameCard, GameDetail, fetchGames, fetchGameDetail } from '@/lib/api';
import MatchupCard from '@/components/MatchupCard';
import GameDetailModal from '@/components/GameDetailModal';
import { Calendar, ChevronLeft, ChevronRight, RefreshCw, TrendingUp } from 'lucide-react';

function formatDisplayDate(dateStr: string): string {
  const y = parseInt(dateStr.slice(0, 4));
  const m = parseInt(dateStr.slice(4, 6)) - 1;
  const d = parseInt(dateStr.slice(6, 8));
  const date = new Date(y, m, d);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = Math.round((date.getTime() - today.getTime()) / 86400000);

  if (diff === 0) return 'Bugün';
  if (diff === -1) return 'Dün';
  if (diff === 1) return 'Yarın';
  return date.toLocaleDateString('tr-TR', { day: 'numeric', month: 'long', year: 'numeric' });
}

function todayStr(): string {
  const d = new Date();
  return `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
}

function offsetDate(dateStr: string, days: number): string {
  const y = parseInt(dateStr.slice(0, 4));
  const m = parseInt(dateStr.slice(4, 6)) - 1;
  const d = parseInt(dateStr.slice(6, 8));
  const date = new Date(y, m, d + days);
  return `${date.getFullYear()}${String(date.getMonth() + 1).padStart(2, '0')}${String(date.getDate()).padStart(2, '0')}`;
}

function LoadingSkeleton() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
      {[...Array(6)].map((_, i) => (
        <div key={i} className="skeleton" style={{ height: 220, borderRadius: 16 }} />
      ))}
    </div>
  );
}

export default function HomePage() {
  const [date, setDate] = useState(todayStr());
  const [games, setGames] = useState<GameCard[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedGame, setSelectedGame] = useState<GameDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const loadGames = useCallback(async (d: string) => {
    setLoading(true);
    setError(null);
    setGames(null);
    try {
      const data = await fetchGames(d);
      setGames(data.games);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDateChange = (d: string) => {
    setDate(d);
    loadGames(d);
  };

  const openDetail = async (game: GameCard) => {
    setDetailLoading(true);
    try {
      const detail = await fetchGameDetail(game.event_id);
      setSelectedGame(detail);
    } catch {
      // Fallback: show card data without props
      setSelectedGame({ ...game, home_player_props: [], away_player_props: [], home_injuries: [], away_injuries: [] });
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <header style={{
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'rgba(13, 21, 37, 0.8)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}>
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36,
            height: 36,
            borderRadius: 10,
            background: 'linear-gradient(135deg, #F7A230, #1D9BF0)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 18,
          }}>
            🏀
          </div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, color: '#f1f5f9', lineHeight: 1.1 }}>
              NBA<span style={{ color: '#F7A230' }}>Predict</span>
            </div>
            <div style={{ fontSize: 10, color: '#475569', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              AI · XGBoost · ML
            </div>
          </div>
        </div>

        {/* Date Picker */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            onClick={() => handleDateChange(offsetDate(date, -1))}
            style={{
              padding: '8px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#94a3b8',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <ChevronLeft size={16} />
          </button>

          <button
            onClick={() => handleDateChange(date)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              padding: '8px 16px',
              borderRadius: 8,
              background: 'rgba(247, 162, 48, 0.12)',
              border: '1px solid rgba(247, 162, 48, 0.25)',
              color: '#F7A230',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 600,
              transition: 'all 0.2s',
            }}
          >
            <Calendar size={14} />
            {formatDisplayDate(date)}
          </button>

          <button
            onClick={() => handleDateChange(offsetDate(date, 1))}
            style={{
              padding: '8px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#94a3b8',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <ChevronRight size={16} />
          </button>

          <button
            onClick={() => handleDateChange(date)}
            title="Yenile"
            style={{
              padding: '8px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: '#64748b',
              cursor: 'pointer',
            }}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <TrendingUp size={14} color="#4ade80" />
          <span style={{ fontSize: 12, color: '#4ade80', fontWeight: 600 }}>Live AI Predictions</span>
        </div>
      </header>

      {/* Hero Banner */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(29,155,240,0.08) 0%, rgba(247,162,48,0.06) 100%)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        padding: '32px 24px',
        textAlign: 'center',
      }}>
        <h1 style={{ fontSize: 32, fontWeight: 900, color: '#f1f5f9', margin: 0, lineHeight: 1.2 }}>
          NBA{' '}
          <span style={{
            background: 'linear-gradient(135deg, #F7A230, #1D9BF0)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}>
            Tahmin Platformu
          </span>
        </h1>
        <p style={{ fontSize: 14, color: '#64748b', marginTop: 8, maxWidth: 500, margin: '8px auto 0' }}>
          XGBoost ML modelleri ile günlük NBA maç skoru, spread, over/under ve oyuncu prop tahminleri.
          Tüm tahminler %100 proprietary — bahis şirketlerine bağımlılık yok.
        </p>

        {/* CTA — Load Today's Games */}
        {!games && !loading && (
          <button
            className="btn-orange"
            style={{ marginTop: 20 }}
            onClick={() => loadGames(todayStr())}
          >
            🏀 Bugünün Tahminlerini Yükle
          </button>
        )}
      </div>

      {/* Main Content */}
      <main style={{ flex: 1, padding: '24px', maxWidth: 1200, margin: '0 auto', width: '100%' }}>

        {/* Error State */}
        {error && (
          <div style={{
            padding: '16px 20px',
            borderRadius: 12,
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            color: '#f87171',
            marginBottom: 20,
          }}>
            <strong>Hata:</strong> {error}
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
              Backend'in çalıştığından emin olun: <code>uvicorn api.main:app --reload</code>
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && <LoadingSkeleton />}

        {/* Games Grid */}
        {games !== null && !loading && (
          <>
            {games.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '60px 20px' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>🏀</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#475569' }}>
                  Bu tarih için tahmin bulunamadı
                </div>
                <div style={{ fontSize: 13, color: '#334155', marginTop: 8 }}>
                  Önce <code style={{ color: '#F7A230' }}>python -m engine.predict_daily --date {date}</code> çalıştırın
                </div>
              </div>
            ) : (
              <>
                {/* Count Badge */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                  <div style={{
                    padding: '4px 12px',
                    borderRadius: 20,
                    background: 'rgba(247, 162, 48, 0.12)',
                    border: '1px solid rgba(247, 162, 48, 0.2)',
                    fontSize: 12,
                    fontWeight: 600,
                    color: '#F7A230',
                  }}>
                    {games.length} MAÇ
                  </div>
                  <span style={{ fontSize: 13, color: '#475569' }}>
                    {formatDisplayDate(date)} tahminleri
                  </span>
                </div>

                {/* Grid */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
                  gap: 16,
                }}>
                  {games.map(game => (
                    <MatchupCard
                      key={game.event_id}
                      game={game}
                      onClick={() => openDetail(game)}
                    />
                  ))}
                </div>
              </>
            )}
          </>
        )}

        {/* Detail Loading Overlay */}
        {detailLoading && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(7, 11, 20, 0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 999,
          }}>
            <div style={{
              padding: '20px 32px',
              borderRadius: 14,
              background: '#0D1525',
              border: '1px solid rgba(255,255,255,0.1)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              fontSize: 14,
              color: '#94a3b8',
            }}>
              <RefreshCw size={16} style={{ animation: 'spin 1s linear infinite', color: '#F7A230' }} />
              Maç detayı yükleniyor...
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid rgba(255,255,255,0.06)',
        padding: '16px 24px',
        textAlign: 'center',
        fontSize: 12,
        color: '#334155',
      }}>
        NBA Tahmin Platformu • XGBoost ML • ESPN API • BigQuery • 2025-26 Sezonu
      </footer>

      {/* Game Detail Modal */}
      {selectedGame && (
        <GameDetailModal
          game={selectedGame}
          onClose={() => setSelectedGame(null)}
        />
      )}
    </div>
  );
}
