'use client';

import { useState } from 'react';
import Navbar from '@/components/layout/Navbar';
import Footer from '@/components/layout/Footer';

export default function PlanesPage() {
    const [isAnnual, setIsAnnual] = useState(true);

    return (
        <>
            <Navbar />
            <div className="page-wrapper" id="planes-view" style={{ paddingTop: '60px' }}>
                <div style={{ maxWidth: '1280px', margin: '0 auto', width: '100%', padding: '0 24px' }}>
                    <article className="flex flex-col items-center w-full" style={{ padding: '40px 0' }}>
                        <h1 className="about-title" style={{ textAlign: 'center', marginBottom: '16px', width: '100%' }}>Elige el Plan para Ti</h1>
                        <p style={{ color: 'var(--text-muted)', textAlign: 'center', fontSize: '1.1rem', marginBottom: '32px' }}>
                            Empieza gratis, mejora cuando necesites más herramientas.
                        </p>

                        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '40px', alignItems: 'center', gap: '12px', flexDirection: 'column' }}>
                            <div className="mode-toggle" style={{ width: 'auto' }}>
                                <button
                                    className={`btn-toggle ${!isAnnual ? 'active' : ''}`}
                                    onClick={() => setIsAnnual(false)}
                                >
                                    Pago Mensual
                                </button>
                                <button
                                    className={`btn-toggle ${isAnnual ? 'active' : ''}`}
                                    onClick={() => setIsAnnual(true)}
                                >
                                    Pago Anual (Ahorra)
                                </button>
                            </div>
                        </div>

                        <div className="plans-grid grid grid-cols-1 md:grid-cols-3 gap-8 lg:gap-10 mb-10 w-full">

                            {/* Plan Gratis */}
                            <div className="plan-card glass-panel" style={{ padding: '32px', display: 'flex', flexDirection: 'column', border: '1px solid var(--border-light)' }}>
                                <h3 style={{ fontSize: '1.4rem', color: 'var(--text-muted)', marginBottom: '8px', fontFamily: 'var(--font-display)' }}>Estudiante / Principiante</h3>
                                <div style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '24px' }}>
                                    $0 <span style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}>/ mes</span>
                                </div>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginBottom: '24px', minHeight: '48px' }}>Perfecto para aprender los fundamentos de diversificación.</p>
                                <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 32px 0', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Hasta 5 optimizaciones al mes</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Acceso al Directorio Global (500+ activos)</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Test de Sueños Básico</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--success)' }}>✓</span> Estrategia de Max Sharpe Ratio</li>
                                    <li style={{ display: 'flex', gap: '8px', color: 'var(--text-muted)' }}><span style={{ color: 'var(--border-light)' }}>✗</span> Sin estrategias avanzadas (HRP)</li>
                                    <li style={{ display: 'flex', gap: '8px', color: 'var(--text-muted)' }}><span style={{ color: 'var(--border-light)' }}>✗</span> Sin reportes PDF</li>
                                </ul>
                                <button className="btn btn-secondary" style={{ width: '100%' }}>Registrarme Gratis</button>
                            </div>

                            {/* Plan Pro */}
                            <div className="plan-card glass-panel primary-glow" style={{ padding: '32px', display: 'flex', flexDirection: 'column', border: '2px solid var(--accent-primary)', transform: 'scale(1.02)', position: 'relative', zIndex: 10 }}>
                                <div style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)', background: 'var(--accent-primary)', color: 'white', padding: '4px 12px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase' }}>
                                    Más Popular
                                </div>
                                <h3 style={{ fontSize: '1.4rem', color: 'var(--accent-primary)', marginBottom: '8px', fontWeight: 700, fontFamily: 'var(--font-display)' }}>Inversionista Pro</h3>
                                <div style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '4px' }}>
                                    <span>{isAnnual ? '$1,790' : '$199'}</span>
                                    <span style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}> {isAnnual ? 'MXN / año' : 'MXN / mes'}</span>
                                </div>
                                <div className="price-subtitle" style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                                    {isAnnual ? 'O $149 MXN / mes facturado anualmente' : 'Facturado mensualmente'}
                                </div>
                                <p style={{ color: 'var(--accent-primary)', fontWeight: 600, fontSize: '0.95rem', marginBottom: '8px', minHeight: '24px' }}>Diseñado para gestionar portafolios mayores a $80,000 MXN.</p>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.90rem', marginBottom: '20px', minHeight: '40px' }}>Maximiza tus rendimientos con herramientas avanzadas.</p>
                                <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 32px 0', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> Optimizaciones ilimitadas</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> Estrategia HRP (Riesgo Jerárquico)</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> Backtesting de estrés histórico</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> Exportación de Reportes PDF</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> 3 Portafolios Guardados</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--accent-primary)' }}>✓</span> Proyección Avanzada de Dividendos</li>
                                </ul>
                                <button className="btn btn-cta glow-effect" style={{ width: '100%' }}>Comenzar Prueba</button>
                            </div>

                            {/* Plan Premium */}
                            <div className="plan-card glass-panel" style={{ padding: '32px', display: 'flex', flexDirection: 'column', border: '1px solid var(--border-light)' }}>
                                <h3 style={{ fontSize: '1.4rem', color: 'var(--warning)', marginBottom: '8px', fontFamily: 'var(--font-display)' }}>Wealth Manager</h3>
                                <div style={{ fontSize: '2.5rem', fontWeight: 800, marginBottom: '4px' }}>
                                    <span>{isAnnual ? '$4,990' : '$499'}</span>
                                    <span style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 400 }}> {isAnnual ? 'MXN / año' : 'MXN / mes'}</span>
                                </div>
                                <div className="price-subtitle" style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
                                    {isAnnual ? 'O $415 MXN / mes facturado anualmente' : 'Facturado mensualmente'}
                                </div>
                                <p style={{ color: 'var(--warning)', fontWeight: 600, fontSize: '0.95rem', marginBottom: '8px', minHeight: '24px' }}>Para gestores y alto patrimonio.</p>
                                <p style={{ color: 'var(--text-muted)', fontSize: '0.90rem', marginBottom: '20px', minHeight: '40px' }}>Herramientas avanzadas para asesores.</p>
                                <ul style={{ listStyle: 'none', padding: 0, margin: '0 0 32px 0', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--warning)' }}>✓</span> Todo en el plan Pro</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--warning)' }}>✓</span> Portafolios Guardados Ilimitados</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--warning)' }}>✓</span> Inclusión de Criptomonedas y Fibras</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--warning)' }}>✓</span> Alertas de desviación por email</li>
                                    <li style={{ display: 'flex', gap: '8px' }}><span style={{ color: 'var(--warning)' }}>✓</span> Prioridad de soporte</li>
                                </ul>
                                <button className="btn btn-secondary" style={{ width: '100%' }}>Seleccionar</button>
                            </div>

                        </div>
                    </article>
                </div>
            </div>
            <Footer />
        </>
    );
}
