// ══════════════════════════════════════════════════════════════
// config.js — Controlador del módulo de configuración
// ══════════════════════════════════════════════════════════════

import { getCurrentUser, getSupabase } from './session-manager.js';

export const template = `
<div class="module-container" id="config-module">
    <div class="module-header">
        <h2>Ajustes</h2>
        <p>Configura tu cuenta y comisiones de broker</p>
    </div>

    <div class="config-grid" style="display: grid; gap: 24px; max-width: 600px;">
        
        <!-- Tarjeta de Visualización -->
        <div class="glass-panel" style="padding: 24px;">
            <h3 style="font-size: 1.15rem; margin-bottom: 24px; font-weight:600;">Preferencias Visuales</h3>
            
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
                <div>
                    <strong style="display:block; margin-bottom:4px;">Moneda Principal</strong>
                    <span style="font-size:0.85rem; color:var(--text-muted);">Muestra los valores del dashboard</span>
                </div>
                <!-- Toggle Switch -->
                <div class="currency-toggle" style="background: rgba(0,0,0,0.3); padding:4px; border-radius:20px; display:flex; cursor:pointer;" id="toggle-currency">
                    <div id="curr-mxn" style="padding: 6px 16px; border-radius:16px; font-size:0.85rem; font-weight:700; background:var(--accent-primary); color:white; transition:all 0.3s; pointer-events:none;">MXN</div>
                    <div id="curr-usd" style="padding: 6px 16px; border-radius:16px; font-size:0.85rem; font-weight:600; color:var(--text-muted); transition:all 0.3s; pointer-events:none;">USD</div>
                </div>
            </div>

            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="display:block; margin-bottom:4px;">Notificaciones por Correo</strong>
                    <span style="font-size:0.85rem; color:var(--text-muted); max-width:240px; display:inline-block;">Alertas de rebalanceo y resúmenes semanales</span>
                </div>
                <!-- Standard CSS Toggle -->
                <label style="position:relative; display:inline-block; width:50px; height:28px;">
                    <input type="checkbox" id="toggle-email" checked>
                    <!-- Checkbox style managed in JS via class changes so we don't need excessive external CSS -->
                    <span id="email-slider-bg" style="position:absolute; cursor:pointer; top:0; left:0; right:0; bottom:0; background-color:var(--accent-primary); transition:.3s; border-radius:28px;">
                        <span id="email-slider-ball" style="position:absolute; content:''; height:22px; width:22px; left:25px; bottom:3px; background-color:white; transition:.3s; border-radius:50%;"></span>
                    </span>
                </label>
            </div>
        </div>

        <!-- Tarjeta de Comisiones -->
        <div class="glass-panel" style="padding: 24px;">
            <h3 style="font-size: 1.15rem; margin-bottom: 16px; font-weight:600;">Comisiones del Broker</h3>
            <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:24px; line-height:1.5;">Ajusta la comisión que cobra tu broker para obtener estimaciones precisas de capital restante después de operar.</p>
            
            <div style="margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; margin-bottom: 16px;">
                    <strong id="broker-preview" style="background:rgba(255,255,255,0.05); padding:4px 12px; border-radius:8px; font-weight:600; font-family:var(--font-mono); font-size:0.95rem;">GBM+ = 0.25%</strong>
                    <strong id="slider-val" style="color:var(--accent-primary); font-family:var(--font-mono); font-size:1.1rem;">0.25%</strong>
                </div>
                <!-- Range Slider -->
                <input type="range" id="commission-slider" min="0" max="2" step="0.05" value="0.25" style="width:100%; height:6px; -webkit-appearance:none; border-radius:4px; outline:none; margin-bottom:12px; cursor:pointer;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:var(--text-muted);">
                    <span>0% (Freetrade)</span>
                    <span>2% (Tradicional)</span>
                </div>
            </div>
        </div>

        <div style="text-align:center;">
            <button id="btn-save-config" class="btn btn-primary glow-effect" style="width: 100%; padding: 14px; font-size:1.05rem; display:flex; justify-content:center; align-items:center; gap:8px;">
                <span id="btn-save-text">Guardar Cambios</span>
                <span class="spinner hidden" id="save-spinner" style="width:18px;height:18px;border-width:2px;"></span>
            </button>
            <div id="config-feedback" style="margin-top: 16px; font-size:0.9rem; height:20px; transition: opacity 0.3s; opacity:0; color:var(--success); font-weight:500;">
                Cambios guardados exitosamente ✓
            </div>
        </div>

    </div>
</div>
`;

export async function init() {
    let settings = {
        currency: 'MXN',
        receive_emails: true,
        broker_fee_pct: 0.25
    };

    // DOM Elements
    const toggleCurrMain = document.getElementById('toggle-currency');
    const toggleCurrMx = document.getElementById('curr-mxn');
    const toggleCurrUsd = document.getElementById('curr-usd');

    const toggleEmail = document.getElementById('toggle-email');
    const sliderBg = document.getElementById('email-slider-bg');
    const sliderBall = document.getElementById('email-slider-ball');

    const slider = document.getElementById('commission-slider');
    const sliderVal = document.getElementById('slider-val');
    const brokerPreview = document.getElementById('broker-preview');

    const btnSave = document.getElementById('btn-save-config');
    const btnSaveText = document.getElementById('btn-save-text');
    const saveSpinner = document.getElementById('save-spinner');
    const feedback = document.getElementById('config-feedback');

    // Utility update visually for currency toggle
    function updateCurrencyVisual() {
        if (settings.currency === 'MXN') {
            toggleCurrMx.style.background = 'var(--accent-primary)';
            toggleCurrMx.style.color = 'white';
            toggleCurrMx.style.fontWeight = '700';

            toggleCurrUsd.style.background = 'transparent';
            toggleCurrUsd.style.color = 'var(--text-muted)';
            toggleCurrUsd.style.fontWeight = '500';
        } else {
            toggleCurrUsd.style.background = 'var(--accent-primary)';
            toggleCurrUsd.style.color = 'white';
            toggleCurrUsd.style.fontWeight = '700';

            toggleCurrMx.style.background = 'transparent';
            toggleCurrMx.style.color = 'var(--text-muted)';
            toggleCurrMx.style.fontWeight = '500';
        }
    }

    // Utility update visually for email toggle
    function updateEmailVisual() {
        if (settings.receive_emails) {
            sliderBg.style.backgroundColor = 'var(--accent-primary)';
            sliderBall.style.left = '25px';
            toggleEmail.checked = true;
        } else {
            sliderBg.style.backgroundColor = 'rgba(255,255,255,0.1)';
            sliderBall.style.left = '3px';
            toggleEmail.checked = false;
        }
    }

    // Utility config text prediction for Broker
    function getBrokerText(val) {
        if (val === 0) return 'Sin comisiones (RH)';
        if (val > 0 && val <= 0.15) return 'Bursanet / Flink';
        if (val > 0.15 && val <= 0.25) return 'GBM+ = 0.25%';
        if (val > 0.25 && val <= 0.5) return 'Actinver Standard';
        if (val > 0.5 && val <= 1.0) return 'Bancos Trads.';
        return 'Alta Comisión > 1%';
    }

    // Event Listeners
    toggleCurrMain.addEventListener('click', () => {
        settings.currency = settings.currency === 'MXN' ? 'USD' : 'MXN';
        updateCurrencyVisual();
    });

    toggleEmail.addEventListener('change', (e) => {
        settings.receive_emails = e.target.checked;
        updateEmailVisual();
    });

    // Support clicking the span directly as well
    sliderBg.addEventListener('click', (e) => {
        if (e.target !== toggleEmail) {
            settings.receive_emails = !settings.receive_emails;
            updateEmailVisual();
        }
    });

    function updateSliderColors() {
        const val = parseFloat(slider.value);
        settings.broker_fee_pct = val;
        sliderVal.textContent = val.toFixed(2) + '%';
        brokerPreview.textContent = getBrokerText(val);

        // Update raw input trail color
        const pct = (val / 2) * 100;
        slider.style.background = `linear-gradient(to right, var(--accent-primary) ${pct}%, rgba(255,255,255,0.1) ${pct}%)`;
    }

    slider.addEventListener('input', updateSliderColors);

    // Initial Load sequence
    try {
        const user = await getCurrentUser();
        // Here we'd fetch actual config object from Supabase (e.g., config jsonb). 
        // Emulating settings sync for now:
        const { data: profile } = await getSupabase()
            .from('profiles')
            .select('config_preferences')
            .eq('id', user?.id)
            .single();

        if (profile?.config_preferences) {
            settings = { ...settings, ...profile.config_preferences };
        }

        // Apply visual states on load
        slider.value = settings.broker_fee_pct;
        updateSliderColors();
        updateCurrencyVisual();
        updateEmailVisual();

    } catch (err) {
        console.error('Config module load error:', err);
    }

    // Save Action
    btnSave.addEventListener('click', async () => {
        saveSpinner.classList.remove('hidden');
        btnSaveText.style.opacity = '0';
        btnSave.disabled = true;
        feedback.style.opacity = '0';

        try {
            const user = await getCurrentUser();

            // Persist using generic unstructured preferences or designated columns
            const { error } = await getSupabase()
                .from('profiles')
                .update({ config_preferences: settings })
                .eq('id', user.id);

            if (error) throw error;

            // Success feedback
            feedback.textContent = 'Configuración guardada exitosamente ✓';
            feedback.style.color = 'var(--success)';
            feedback.style.opacity = '1';
        } catch (err) {
            console.error('Config save fail:', err);
            feedback.textContent = 'Error al guardar. Intenta nuevamente.';
            feedback.style.color = 'var(--danger)';
            feedback.style.opacity = '1';
        } finally {
            saveSpinner.classList.add('hidden');
            btnSaveText.style.opacity = '1';
            btnSave.disabled = false;

            // Auto hide feedback
            setTimeout(() => { feedback.style.opacity = '0'; }, 3000);
        }
    });
}
