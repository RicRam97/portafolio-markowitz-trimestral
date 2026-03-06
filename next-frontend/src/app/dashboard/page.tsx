import type { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Dashboard — Kaudal',
    description: 'Tu dashboard de optimización de portafolios con la Teoría de Markowitz.',
};

export default function DashboardPage() {
    return (
        <div className="max-w-[1200px] mx-auto">
            {/* Welcome banner */}
            <div className="glass-panel p-6 mb-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
                style={{ borderTop: '4px solid var(--accent-primary)' }}>
                <div>
                    <h2 className="text-xl font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                        Optimizador de Portafolio
                    </h2>
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                        Selecciona tus activos, configura tu estrategia y ejecuta la optimización.
                    </p>
                </div>
                <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full"
                    style={{ background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.25)', color: 'var(--success)' }}>
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    📡 Datos actualizados diariamente
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6">
                {/* ===== CONFIG PANEL ===== */}
                <aside className="glass-panel p-5 flex flex-col gap-5">
                    {/* Mode toggle */}
                    <div>
                        <label className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                            Modo de Uso
                        </label>
                        <div className="mode-toggle">
                            <button className="btn-toggle active text-xs">Avanzado (Manual)</button>
                            <button className="btn-toggle text-xs">🌱 Principiantes</button>
                        </div>
                    </div>

                    {/* Budget */}
                    <div>
                        <label htmlFor="budget" className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                            Presupuesto de Inversión (USD)
                        </label>
                        <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold" style={{ color: 'var(--text-muted)' }}>$</span>
                            <input
                                type="number"
                                id="budget"
                                defaultValue={10000}
                                min={100}
                                step={100}
                                className="flex-1 px-3 py-2 rounded-lg text-sm outline-none"
                                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                            />
                        </div>
                    </div>

                    {/* Strategy */}
                    <div>
                        <label htmlFor="strategy" className="text-xs font-bold uppercase tracking-widest mb-2 block" style={{ color: 'var(--text-muted)' }}>
                            Estrategia de Optimización
                        </label>
                        <select
                            id="strategy"
                            className="w-full px-3 py-2 rounded-lg text-sm outline-none"
                            style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                        >
                            <option value="markowitz">Máx Sharpe (Markowitz) — Agresivo</option>
                            <option value="hrp">Paridad de Riesgo (HRP) — Defensivo</option>
                        </select>
                    </div>

                    {/* Tickers */}
                    <div className="flex-1">
                        <div className="flex items-center justify-between mb-2">
                            <label className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                                Universo (Tickers)
                            </label>
                            <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                                style={{ background: 'rgba(37,99,235,0.1)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.2)' }}>
                                0 seleccionados
                            </span>
                        </div>
                        <div className="flex gap-2 mb-3">
                            <input
                                type="text"
                                placeholder="ej. NVDA, AAPL"
                                className="flex-1 px-3 py-2 rounded-lg text-sm outline-none"
                                style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                            />
                            <button className="btn btn-secondary px-3 py-2 text-sm">+</button>
                        </div>
                        <div className="flex gap-2 mb-3">
                            <button className="btn btn-secondary flex-1 text-xs py-1.5">Seleccionar Todo</button>
                            <button className="btn btn-secondary flex-1 text-xs py-1.5">Limpiar Todo</button>
                        </div>
                        <input
                            type="search"
                            placeholder="Buscar tickers..."
                            className="w-full px-3 py-2 rounded-lg text-sm outline-none mb-3"
                            style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                        />
                        {/* Empty state */}
                        <div className="text-center py-8 text-sm rounded-lg" style={{ background: 'rgba(15,23,42,0.3)', color: 'var(--text-muted)', border: '1px dashed var(--border-light)' }}>
                            Cargando tickers del directorio...
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2">
                        <button className="btn btn-cta glow-effect flex-1 text-sm py-3">
                            Optimizar Portafolio
                        </button>
                        <button className="btn btn-secondary px-3 text-xs" title="Exportar PDF">
                            <svg viewBox="0 0 24 24" className="w-4 h-4">
                                <path fill="currentColor" d="M19,12V7H15V3H5C3.89,3 3,3.89 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V12H19M19,19H5V5H13V9H17V19M12,17L15.5,13.5H13V10H11V13.5H8.5L12,17Z" />
                            </svg>
                            PDF
                        </button>
                    </div>
                </aside>

                {/* ===== MAIN CONTENT ===== */}
                <div className="flex flex-col gap-5">
                    {/* Metric cards */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        {[
                            { label: 'Rendimiento Esperado', sub: 'Anual', value: '—', cls: 'positive', tip: 'Basado en datos históricos de 3 años.' },
                            { label: 'Volatilidad (Riesgo)', value: '—', cls: 'neutral', tip: 'Fluctuación esperada.' },
                            { label: 'Ratio de Sharpe', value: '—', cls: 'highlight', tip: 'Retorno por unidad de riesgo.' },
                            { label: 'Efectivo Restante', value: '$—', cls: 'highlight', tip: 'Capital no asignado.' },
                        ].map((m) => (
                            <div key={m.label} className="glass-panel p-4" title={m.tip}>
                                <h3 className="text-xs uppercase tracking-widest mb-1 leading-tight" style={{ color: 'var(--text-muted)' }}>
                                    {m.label} {m.sub && <span className="text-[10px] px-1.5 py-0.5 rounded ml-1" style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)' }}>{m.sub}</span>}
                                </h3>
                                <div className={`metric-value ${m.cls}`}>{m.value}</div>
                            </div>
                        ))}
                    </div>

                    {/* Chart card */}
                    <div className="glass-panel p-5">
                        <h3 className="text-sm font-semibold mb-4">Distribución de Pesos</h3>
                        <div className="flex items-center justify-center text-center py-16 rounded-xl"
                            style={{ background: 'rgba(15,23,42,0.5)', border: '1px dashed var(--border-light)' }}>
                            <div>
                                <div className="text-4xl mb-3">📊</div>
                                <p className="text-sm font-semibold mb-1">Sin datos aún</p>
                                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                                    Selecciona tickers y ejecuta la optimización para ver tu distribución de portafolio.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Table card */}
                    <div className="glass-panel overflow-hidden">
                        <h3 className="text-sm font-semibold p-5 pb-3">Asignación y Títulos del Modelo</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr style={{ borderBottom: '1px solid var(--border-light)' }}>
                                        {['Activo', 'Peso', 'Precio', 'Acciones', 'Total ($)'].map(h => (
                                            <th key={h} className="px-4 py-3 text-left text-xs uppercase tracking-widest font-semibold" style={{ color: 'var(--text-muted)' }}>{h}</th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td colSpan={5} className="px-4 py-10 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
                                            Ejecuta la optimización para ver la asignación de capital
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
