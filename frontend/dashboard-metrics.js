// ══════════════════════════════════════════════════════════════
// dashboard-metrics.js — Módulo principal del Dashboard (Vista inicial)
// ══════════════════════════════════════════════════════════════

import { getCurrentUser, getSupabase } from './session-manager.js';

export const template = `
<div class="module-container" id="dashboard-metrics-module">
    
    <!-- Banner de Bienvenida -->
    <div class="welcome-banner hidden glass-panel" id="welcome-banner" style="margin-bottom: 24px; padding: 24px; background: rgba(59, 130, 246, 0.1); border-left: 4px solid var(--accent-primary); animation: fadeIn 0.5s ease-out;">
        <h2 style="font-size: 1.5rem; margin-bottom: 8px;">¡Bienvenido a Kaudal, <span id="welcome-name"></span>! 🎉</h2>
        <p style="color: var(--text-muted); margin: 0; font-size: 1.05rem;">Tu nivel de riesgo actual es <strong id="welcome-profile" style="color: var(--text-main); text-transform: uppercase;">...</strong>. Hemos configurado el dashboard para ti.</p>
    </div>

    <!-- Header del módulo -->
    <div class="module-header" style="display:flex; justify-content:space-between; align-items:flex-end;">
        <div>
            <h2>Resumen General</h2>
            <p>Métricas de tu actividad y mercados</p>
        </div>
        <div style="color:var(--text-muted); font-size:0.85rem; padding-bottom:4px;">
            <span style="display:inline-block; width:8px; height:8px; background:var(--success); border-radius:50%; margin-right:4px;"></span>
            Sistema Operativo
        </div>
    </div>

    <!-- Grid de métricas -->
    <div class="metrics-dashboard-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 24px;">
        
        <!-- Card 1: Estrategias Activas -->
        <div class="dash-card glass-panel" style="padding: 24px; display: flex; flex-direction: column; transition: transform 0.2s;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 16px;">
                <h3 style="font-size:1.15rem; margin:0; font-weight:600;">Estrategias Activas</h3>
                <span class="icon" style="font-size:1.3rem;">🎯</span>
            </div>
            <div style="flex:1;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.95rem;">
                    <span>Usadas: <strong style="color:var(--text-main);">1</strong> de 3</span>
                    <span style="color:var(--text-muted); font-size:0.85rem; padding: 2px 8px; background: rgba(255,255,255,0.05); border-radius:12px;">Plan Básico</span>
                </div>
                <div class="progress-bar-bg" style="height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; margin-bottom: 24px; overflow:hidden;">
                    <div class="progress-bar-fill" style="height: 100%; width: 33%; background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); border-radius: 4px;"></div>
                </div>
            </div>
            <a href="#estrategias" class="btn btn-primary btn-sm" style="text-align:center; text-decoration:none;">Crear Nueva</a>
        </div>

        <!-- Card 2: Portafolios (Sparkline) -->
        <div class="dash-card glass-panel" style="padding: 24px; display: flex; flex-direction: column;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 12px;">
                <h3 style="font-size:1.15rem; margin:0; font-weight:600;">Rendimiento Global</h3>
                <span class="icon" style="font-size:1.3rem;">📈</span>
            </div>
            <div style="margin-bottom: 12px;">
                <div style="font-size: 2rem; font-family: var(--font-mono); font-weight:700; color: var(--success); letter-spacing:-0.5px;">+12.4%</div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">Acumulado YTD en estrategias</div>
            </div>
            <div style="flex:1; position:relative; min-height:60px;">
                <canvas id="dash-sparkline-chart"></canvas>
            </div>
        </div>

        <!-- Card 3: Tickers Favoritos -->
        <div class="dash-card glass-panel" style="padding: 24px; display: flex; flex-direction: column;">
            <div style="display:flex; justify-content:space-between; margin-bottom: 16px;">
                <h3 style="font-size:1.15rem; margin:0; font-weight:600;">Favoritos</h3>
                <span class="icon" style="font-size:1.3rem;">⭐</span>
            </div>
            <div class="fav-list" style="display:flex; flex-direction:column; gap:12px; flex:1;">
                <!-- Lista compacta -->
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-light); padding-bottom:8px;">
                    <div><strong style="font-family:var(--font-mono);">AAPL</strong> <span style="font-size:0.8rem; color:var(--text-muted);">EEUU</span></div>
                    <div style="text-align:right;"><div style="font-family:var(--font-mono); font-size:0.95rem;">$185.92</div><div style="font-size:0.8rem; color:var(--success); font-family:var(--font-mono);">▲ 1.2%</div></div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-light); padding-bottom:8px;">
                    <div><strong style="font-family:var(--font-mono);">VOO</strong> <span style="font-size:0.8rem; color:var(--text-muted);">EEUU</span></div>
                    <div style="text-align:right;"><div style="font-family:var(--font-mono); font-size:0.95rem;">$462.10</div><div style="font-size:0.8rem; color:var(--success); font-family:var(--font-mono);">▲ 0.4%</div></div>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><strong style="font-family:var(--font-mono);">CEMEX</strong> <span style="font-size:0.8rem; color:var(--text-muted);">BMV</span></div>
                    <div style="text-align:right;"><div style="font-family:var(--font-mono); font-size:0.95rem;">$13.45</div><div style="font-size:0.8rem; color:var(--danger); font-family:var(--font-mono);">▼ 2.1%</div></div>
                </div>
            </div>
        </div>

        <!-- Card 4: Alertas -->
        <div class="dash-card glass-panel" style="padding: 24px; display: flex; flex-direction: column;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 16px;">
                <h3 style="font-size:1.15rem; margin:0; font-weight:600;">Alertas</h3>
                <span class="badge" style="background:var(--danger); color:white; padding:4px 10px; border-radius:12px; font-size:0.8rem; font-weight:800; font-family:var(--font-mono);">2</span>
            </div>
            <div style="flex:1; display:flex; flex-direction:column; gap:12px;">
                <div style="padding:12px; background:rgba(239, 68, 68, 0.08); border-left:3px solid var(--danger); border-radius:4px;">
                    <strong style="display:block; font-size:0.95rem; margin-bottom:4px; color:white;">Desviación Detectada</strong>
                    <span style="font-size:0.85rem; color:var(--text-muted); line-height:1.4; display:block;">Tu portafolio "Jubilación" tiene un exceso de peso en Tecnología. Considera rebalancear.</span>
                </div>
                <div style="padding:12px; background:rgba(16, 185, 129, 0.08); border-left:3px solid var(--success); border-radius:4px;">
                    <strong style="display:block; font-size:0.95rem; margin-bottom:4px; color:white;">Dividendos Estimados</strong>
                    <span style="font-size:0.85rem; color:var(--text-muted); line-height:1.4; display:block;">Tienes pagos pendientes de cobro este trimestre por ~$420 MXN.</span>
                </div>
            </div>
        </div>

    </div>
</div>
`;

export async function init() {
    console.log('[Dashboard] Initializing metrics module');

    // 1. Fetch user data for personalized banner
    try {
        const user = await getCurrentUser();
        if (user) {
            const { data: profile } = await getSupabase()
                .from('profiles')
                .select('first_login_done, investor_profile, first_name')
                .eq('id', user.id)
                .single();

            if (profile && !profile.first_login_done) {
                const banner = document.getElementById('welcome-banner');
                if (banner) {
                    document.getElementById('welcome-name').textContent = profile.first_name || 'Inversor';

                    const profileMap = {
                        'conservative': 'Conservador',
                        'moderate': 'Moderado',
                        'aggressive': 'Agresivo'
                    };
                    document.getElementById('welcome-profile').textContent = profileMap[profile.investor_profile] || 'Balanceado';

                    banner.classList.remove('hidden');

                    // Fire-and-forget update
                    getSupabase()
                        .from('profiles')
                        .update({ first_login_done: true })
                        .eq('id', user.id)
                        .then(() => console.log('first_login_done updated'));
                }
            }
        }
    } catch (err) {
        console.warn('Could not load profile specific data for dashboard banner', err);
    }

    // 2. Render Sparkline using Chart.js
    const ctx = document.getElementById('dash-sparkline-chart');
    if (ctx && window.Chart) {
        // Destroy prev instance if exists (SPA safe)
        if (window.dashSparklineChart) window.dashSparklineChart.destroy();

        window.dashSparklineChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun'],
                datasets: [{
                    data: [0, 1.2, -0.4, 4.2, 7.8, 12.4],
                    borderColor: '#10B981', // var(--success)
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: {
                        target: 'origin',
                        above: 'rgba(16, 185, 129, 0.1)'
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        enabled: true,
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(15, 23, 42, 0.9)',
                        titleFont: { family: 'Inter', size: 12 },
                        bodyFont: { family: 'JetBrains Mono', size: 13 },
                        displayColors: false,
                        callbacks: {
                            label: function (context) {
                                return context.parsed.y > 0 ? `+${context.parsed.y}%` : `${context.parsed.y}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: { display: false },
                    y: { display: false, min: -2 }
                },
                layout: { padding: 0 },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
}
