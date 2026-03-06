'use client';

import { useState } from 'react';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';
import Link from 'next/link';

// Chart.js imports
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';
import { Doughnut, Line } from 'react-chartjs-2';

// Register Chart.js models
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    ArcElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const DEMO_PORTFOLIO = [
    { ticker: "AAPL", name: "Apple", weight: 0.15, color: "#A3AAAE" },
    { ticker: "MSFT", name: "Microsoft", weight: 0.15, color: "#00A4EF" },
    { ticker: "GOOGL", name: "Google", weight: 0.15, color: "#4285F4" },
    { ticker: "AMZN", name: "Amazon", weight: 0.10, color: "#FF9900" },
    { ticker: "META", name: "Meta", weight: 0.10, color: "#0668E1" },
    { ticker: "NVDA", name: "Nvidia", weight: 0.10, color: "#76B900" },
    { ticker: "JPM", name: "JPMorgan Chase", weight: 0.10, color: "#444444" },
    { ticker: "WMT", name: "Walmart", weight: 0.05, color: "#0071CE" },
    { ticker: "TSLA", name: "Tesla", weight: 0.05, color: "#E31937" },
    { ticker: "OXY", name: "Occidental Petroleum", weight: 0.05, color: "#004B87" }
];

export default function DemoPage() {
    const [activeTab, setActiveTab] = useState<'allocation' | 'growth'>('allocation');

    // Allocation Doughnut config
    const doughnutData = {
        labels: DEMO_PORTFOLIO.map(p => p.ticker),
        datasets: [{
            data: DEMO_PORTFOLIO.map(p => p.weight * 100),
            backgroundColor: DEMO_PORTFOLIO.map(p => p.color),
            borderWidth: 1,
            borderColor: 'rgba(15, 23, 42, 1)'
        }]
    };

    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'right' as const, labels: { color: '#94a3b8', font: { family: 'Inter' } } }
        }
    };

    // Growth Line config (simulate 10 years at 24.7%)
    const years = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const amounts = years.map(y => Math.round(10000 * Math.pow(1.247, y)));

    const lineData = {
        labels: years.map(y => `Año ${y}`),
        datasets: [{
            label: 'Crecimiento del Portafolio ($)',
            data: amounts,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.4
        }]
    };

    const lineOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { ticks: { color: '#94a3b8', callback: (val: number | string) => '$' + Number(val).toLocaleString() } },
            x: { ticks: { color: '#94a3b8' } }
        },
        plugins: {
            legend: { display: false }
        }
    };

    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="demo-view" style={{ paddingTop: '100px', minHeight: 'calc(100vh - 60px)' }}>
                <div className="about-page">
                    <div className="glass-panel" style={{ maxWidth: '1100px', margin: '0 auto', padding: '32px', borderTop: '4px solid var(--success)' }}>
                        <h1 style={{ textAlign: 'center', marginBottom: '8px', fontFamily: 'var(--font-display)', color: 'var(--text-main)', fontSize: '2rem' }}>Demo Interactiva</h1>
                        <p style={{ textAlign: 'center', color: 'var(--text-muted)', marginBottom: '32px' }}>
                            Simulación de portafolio para: AAPL, GOOGL, META, AMZN, TSLA, NVDA, MSFT, JPM, OXY, WMT
                        </p>

                        {/* Metrics Row */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px', marginBottom: '32px' }}>
                            <div className="metric-card glass-panel" style={{ background: 'rgba(15,23,42,0.8)' }}>
                                <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Retorno Anualizado Escenario Óptimo</h3>
                                <div className="metric-value positive" style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>+24.7%</div>
                            </div>
                            <div className="metric-card glass-panel" style={{ background: 'rgba(15,23,42,0.8)' }}>
                                <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Volatilidad del Conjunto</h3>
                                <div className="metric-value neutral" style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--text-main)' }}>18.3%</div>
                            </div>
                            <div className="metric-card glass-panel" style={{ background: 'rgba(15,23,42,0.8)' }}>
                                <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Sharpe Ratio</h3>
                                <div className="metric-value highlight" style={{ fontSize: '1.8rem', fontWeight: 'bold', color: 'var(--accent-primary)' }}>1.34</div>
                            </div>
                        </div>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '32px', marginBottom: '32px' }}>
                            {/* Chart Toggle */}
                            <div style={{ display: 'flex', justifyContent: 'flex-end', width: '100%' }}>
                                <div className="mode-toggle" style={{ width: 'auto' }}>
                                    <button
                                        className={`btn-toggle ${activeTab === 'allocation' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('allocation')}
                                    >
                                        Distribución Sugerida
                                    </button>
                                    <button
                                        className={`btn-toggle ${activeTab === 'growth' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('growth')}
                                    >
                                        Evolución a 10 años
                                    </button>
                                </div>
                            </div>

                            {/* Chart Container React-ChartJS */}
                            <div className="chart-card glass-panel" style={{ background: 'rgba(15,23,42,0.8)', border: '1px solid var(--border-light)', padding: '24px' }}>
                                <div className="canvas-wrapper doughnut-wrapper" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    {activeTab === 'allocation' ? (
                                        <Doughnut data={doughnutData} options={doughnutOptions} />
                                    ) : (
                                        <Line data={lineData} options={lineOptions} />
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Table */}
                        <div className="table-section glass-panel" style={{ background: 'rgba(15,23,42,0.8)' }}>
                            <h3 style={{ padding: '20px 20px 0', fontSize: '1.2rem', color: 'var(--text-main)' }}>Distribución de Capital Simulada (Inversión Inicial: $10,000 USD)</h3>
                            <div className="table-container" style={{ overflowX: 'auto', padding: '20px' }}>
                                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                                    <thead>
                                        <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                            <th style={{ padding: '12px', color: 'var(--text-muted)' }}>Activo</th>
                                            <th style={{ padding: '12px', color: 'var(--text-muted)' }}>Empresa</th>
                                            <th style={{ padding: '12px', color: 'var(--text-muted)' }}>Peso Asignado</th>
                                            <th style={{ padding: '12px', color: 'var(--text-muted)' }}>Monto ($)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {DEMO_PORTFOLIO.map(item => (
                                            <tr key={item.ticker} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                                <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--text-main)' }}>{item.ticker}</td>
                                                <td style={{ padding: '12px' }}>{item.name}</td>
                                                <td style={{ padding: '12px' }}>
                                                    <span style={{ color: item.color, fontWeight: 700 }}>{(item.weight * 100).toFixed(0)}%</span>
                                                </td>
                                                <td style={{ padding: '12px', fontFamily: 'var(--font-mono)' }}>
                                                    ${(10000 * item.weight).toLocaleString("en-US", { minimumFractionDigits: 2 })} USD
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <div style={{ textAlign: 'center', marginTop: '40px', borderTop: '1px solid var(--border-light)', paddingTop: '32px' }}>
                            <h3 style={{ marginBottom: '16px', color: 'var(--text-main)' }}>¿Quieres hacer esto con tus propias acciones?</h3>
                            <Link href="/login" className="btn btn-cta glow-effect" style={{ padding: '14px 28px', display: 'inline-flex', textDecoration: 'none' }}>
                                Crear tu Portafolio Personalizado →
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
}
