'use client';

import { useState, useEffect } from 'react';

export default function DashboardPreview() {
    const [activeTab, setActiveTab] = useState<'distribution' | 'growth'>('distribution');
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    if (!mounted) return null; // Avoid hydration mismatch for random values if any

    return (
        <div className="preview-container glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '24px', width: '100%' }}>
                <div className="mode-toggle" style={{ width: 'auto' }}>
                    <button
                        className={`btn-toggle ${activeTab === 'distribution' ? 'active' : ''}`}
                        onClick={() => setActiveTab('distribution')}
                    >
                        Distribución
                    </button>
                    <button
                        className={`btn-toggle ${activeTab === 'growth' ? 'active' : ''}`}
                        onClick={() => setActiveTab('growth')}
                    >
                        Evolución a 10 años
                    </button>
                </div>
            </div>

            {/* Layout Container */}
            <div style={{ display: 'flex', flexDirection: 'row', width: '100%', gap: '32px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
                {/* Left Side: Metrics */}
                <div className="preview-metrics" style={{ display: 'flex', flexDirection: 'column', gap: '16px', flex: '1', minWidth: '250px', marginBottom: 0 }}>
                    <div className="preview-metric">
                        <span className="preview-metric-label">Rendimiento Esperado</span>
                        <span className="preview-metric-value positive">+24.7%</span>
                    </div>
                    <div className="preview-metric">
                        <span className="preview-metric-label">Volatilidad</span>
                        <span className="preview-metric-value neutral">18.3%</span>
                    </div>
                </div>

                {/* Right Side: Chart Area */}
                <div className="preview-chart-area" style={{ flex: '2', minWidth: '300px', height: '300px', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 0 }}>
                    {activeTab === 'distribution' ? (
                        <div className="fade-in" style={{ width: '200px', height: '200px', borderRadius: '50%', border: '40px solid var(--accent-primary)', borderRightColor: 'var(--success)', borderTopColor: '#f59e0b', opacity: 0.8 }} />
                    ) : (
                        <div className="canvas-wrapper preview-bar-chart fade-in" style={{ display: 'flex', alignItems: 'flex-end', height: '250px', width: '100%', gap: '10px' }}>
                            {[30, 45, 25, 60, 40, 80, 55, 95, 70, 100, 85, 120].map((val, i) => (
                                <div key={i} className="preview-bar" style={{
                                    height: `${Math.min(100, (val / 120) * 100)}%`,
                                    flex: 1,
                                    backgroundColor: 'var(--accent-primary)',
                                    borderRadius: '4px 4px 0 0',
                                    opacity: 0.8 + (i * 0.02)
                                }}>
                                    <div className="preview-bar-pct" style={{ position: 'absolute', top: '-20px', left: '50%', transform: 'translateX(-50%)', fontSize: '0.75rem', opacity: 0 }}></div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <div className="preview-disclaimer">
                ⚠️ Valores ilustrativos de ejemplo — no son resultados reales ni alertas de operación.
            </div>
        </div>
    );
}
