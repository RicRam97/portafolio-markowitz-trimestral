'use client';

import { BADGES } from '@/lib/badges';

interface UserBadge {
    badge_id: string;
    fecha_obtenido: string;
}

interface Props {
    userBadges: UserBadge[];
}

function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString('es-MX', {
        day: '2-digit', month: 'short', year: 'numeric',
    });
}

export default function MisLogros({ userBadges }: Props) {
    const allBadgeIds = Object.keys(BADGES);
    const earnedIds = new Set(userBadges.map((b) => b.badge_id));
    const earnedMap = Object.fromEntries(userBadges.map((b) => [b.badge_id, b.fecha_obtenido]));

    return (
        <div className="glass-panel overflow-hidden">
            <div style={{
                padding: '16px 20px 12px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-light)',
            }}>
                <h4 style={{ fontFamily: 'var(--font-display)', fontSize: '0.95rem', fontWeight: 700 }}>
                    Mis Logros
                </h4>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    {userBadges.length} de {allBadgeIds.length}
                </span>
            </div>

            {allBadgeIds.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No hay logros configurados.
                </div>
            ) : (
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                    gap: '12px',
                    padding: '16px 20px',
                }}>
                    {allBadgeIds.map((badgeId) => {
                        const badge = BADGES[badgeId];
                        const earned = earnedIds.has(badgeId);
                        return (
                            <div
                                key={badgeId}
                                style={{
                                    padding: '16px',
                                    borderRadius: '12px',
                                    background: earned
                                        ? 'rgba(37,99,235,0.08)'
                                        : 'rgba(255,255,255,0.02)',
                                    border: earned
                                        ? '1px solid rgba(37,99,235,0.25)'
                                        : '1px solid rgba(255,255,255,0.06)',
                                    opacity: earned ? 1 : 0.5,
                                    transition: 'all 0.2s ease',
                                }}
                            >
                                <div style={{ fontSize: '1.8rem', marginBottom: '8px' }}>
                                    {earned ? badge.icon : '\uD83D\uDD12'}
                                </div>
                                <div style={{
                                    fontSize: '0.85rem',
                                    fontWeight: 700,
                                    marginBottom: '4px',
                                    color: earned ? 'var(--text-main)' : 'var(--text-muted)',
                                }}>
                                    {badge.name}
                                </div>
                                <div style={{
                                    fontSize: '0.75rem',
                                    color: 'var(--text-muted)',
                                    lineHeight: 1.4,
                                    marginBottom: earned ? '8px' : '0',
                                }}>
                                    {badge.description}
                                </div>
                                {earned && earnedMap[badgeId] && (
                                    <div style={{
                                        fontSize: '0.68rem',
                                        color: 'var(--accent-primary)',
                                        fontFamily: 'var(--font-mono)',
                                    }}>
                                        Obtenido el {formatDate(earnedMap[badgeId])}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
