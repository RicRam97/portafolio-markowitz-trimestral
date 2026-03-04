// ══════════════════════════════════════════════════════════════
// cuenta.js — Controlador del módulo de información de cuenta
// ══════════════════════════════════════════════════════════════

import { getCurrentUser, getSupabase } from './session-manager.js';

export const template = `
<div class="module-container" id="cuenta-module">
    <div class="module-header">
        <h2>Información de Cuenta</h2>
        <p>Gestiona los detalles de tu perfil y suscripción</p>
    </div>

    <div class="cuenta-grid" style="display: grid; gap: 24px; max-width: 600px;">
        
        <!-- Tarjeta de Perfil -->
        <div class="glass-panel" style="padding: 24px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 24px;">
                <h3 style="font-size: 1.15rem; font-weight:600; margin:0;">Perfil Personal</h3>
                <span style="font-size:1.5rem;">👤</span>
            </div>
            
            <div style="margin-bottom: 20px;">
                <label style="display:block; font-size:0.85rem; color:var(--text-muted); margin-bottom:8px;">Nombre (Editable)</label>
                <div style="display:flex; align-items:center; gap:12px; background:rgba(15,23,42,0.6); padding:4px; border-radius:8px; border:1px solid var(--border-light);">
                    <input type="text" id="input-name" style="flex:1; background:transparent; border:none; color:var(--text-main); font-size:1rem; padding:8px 12px; outline:none;" placeholder="Tu nombre">
                    <button id="btn-save-name" class="btn btn-secondary btn-sm" style="padding:6px 16px;">Guardar</button>
                    <div id="name-spinner" class="spinner hidden" style="width:16px; height:16px; margin-right:8px;"></div>
                </div>
                <div id="name-feedback" style="font-size:0.8rem; color:var(--success); margin-top:8px; opacity:0; transition:opacity 0.3s;">Nombre actualizado ✓</div>
            </div>

            <div>
                <label style="display:block; font-size:0.85rem; color:var(--text-muted); margin-bottom:8px;">Correo Electrónico</label>
                <div style="background:rgba(255,255,255,0.03); padding:12px; border-radius:8px; font-family:var(--font-mono); color:var(--text-muted); cursor:not-allowed;" id="display-email">
                    cargando...
                </div>
            </div>
        </div>

        <!-- Tarjeta de Suscripción -->
        <div class="glass-panel" style="padding: 24px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
                <h3 style="font-size: 1.15rem; font-weight:600; margin:0;">Plan Activo</h3>
                <span class="plan-badge" id="account-plan-badge" style="background:rgba(16, 185, 129, 0.15); border:1px solid rgba(16, 185, 129, 0.3); color:var(--success); padding:6px 12px; border-radius:12px; font-size:0.85rem; font-weight:700;">Básico (Free)</span>
            </div>
            
            <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:24px;">
                Tu plan actual se renueva el <strong style="color:var(--text-main);" id="plan-renewal-date">15 Dic 2026</strong>. Tienes acceso a funcionalidades limitadas.
            </p>

            <button id="btn-upgrade-plan" class="btn btn-primary glow-effect" style="width:100%; padding:12px;">Actualiza a PRO ⚡</button>
        </div>

        <!-- Tabla Historial Pagos -->
        <div class="glass-panel" style="padding: 24px;">
            <h3 style="font-size: 1.15rem; font-weight:600; margin-bottom: 16px;">Historial de Pagos</h3>
            <div style="overflow-x:auto;">
                <table style="width:100%; text-align:left; border-collapse:collapse; font-size:0.9rem;">
                    <thead>
                        <tr style="border-bottom:1px solid var(--border-light); color:var(--text-muted);">
                            <th style="padding:12px 0;">Fecha</th>
                            <th style="padding:12px 0;">Descripción</th>
                            <th style="padding:12px 0;">Monto</th>
                            <th style="padding:12px 0; text-align:right;">Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:12px 0; font-family:var(--font-mono); font-size:0.85rem;">15 Nov 2026</td>
                            <td style="padding:12px 0;">Suscripción PRO (Mensual)</td>
                            <td style="padding:12px 0; font-family:var(--font-mono);">$99.00 MXN</td>
                            <td style="padding:12px 0; text-align:right;"><span style="color:var(--success); font-size:0.8rem; background:rgba(16,185,129,0.1); padding:2px 8px; border-radius:12px;">Pagado</span></td>
                        </tr>
                        <tr>
                            <td style="padding:12px 0; font-family:var(--font-mono); font-size:0.85rem;">15 Oct 2026</td>
                            <td style="padding:12px 0;">Suscripción PRO (Mensual)</td>
                            <td style="padding:12px 0; font-family:var(--font-mono);">$99.00 MXN</td>
                            <td style="padding:12px 0; text-align:right;"><span style="color:var(--success); font-size:0.8rem; background:rgba(16,185,129,0.1); padding:2px 8px; border-radius:12px;">Pagado</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Zona de Peligro -->
        <div class="glass-panel" style="padding: 24px; border: 1px solid rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.05);">
            <h3 style="font-size: 1.15rem; font-weight:600; margin-bottom: 8px; color:var(--danger);">Zona de Peligro</h3>
            <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:20px;">
                Cerrar tu cuenta eliminará todos tus portafolios, historial y configuraciones. Esta acción no se puede deshacer.
            </p>
            
            <div id="close-account-step1">
                <button id="btn-close-account-init" class="btn btn-secondary" style="border-color:var(--danger); color:var(--danger);">Cerrar Mi Cuenta</button>
            </div>
            
            <!-- Step 2 Confirmation -->
            <div id="close-account-step2" class="hidden" style="background:rgba(15,23,42,0.8); padding:16px; border-radius:8px; border:1px solid var(--danger); margin-top:12px;">
                <strong style="color:white; display:block; margin-bottom:8px;">¿Estás absolutamente seguro?</strong>
                <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:16px;">Borraremos todo tu progreso del sistema inmediatamente.</p>
                <div style="display:flex; gap:12px;">
                    <button id="btn-close-account-cancel" class="btn btn-secondary" style="flex:1;">Cancelar</button>
                    <button id="btn-close-account-confirm" class="btn btn-primary" style="flex:1; background:var(--danger); color:white;">Sí, eliminar cuenta</button>
                </div>
            </div>
        </div>

    </div>
</div>

<!-- Plan Upgrade Modal -->
<div class="modal-overlay hidden" id="upgrade-modal" style="display:flex; justify-content:center; align-items:center;">
    <div class="modal-content glass-panel" style="max-width:400px; width:100%; animation: slideUp 0.3s ease-out;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
            <h3 style="font-size: 1.4rem; font-weight:700; margin:0;">Actualiza a PRO</h3>
            <button id="btn-close-upgrade" style="background:none; border:none; color:var(--text-muted); font-size:1.5rem; cursor:pointer;">&times;</button>
        </div>
        <p style="color:var(--text-muted); margin-bottom:24px;">Accede a estrategias ilimitadas, rebalanceo automático y más tickers internacionales.</p>
        
        <div style="background:rgba(59, 130, 246, 0.1); border:1px solid var(--accent-primary); border-radius:12px; padding:20px; text-align:center; margin-bottom:24px;">
            <div style="font-size:2rem; font-family:var(--font-mono); font-weight:800; color:white; margin-bottom:4px;">$99<span style="font-size:1rem; color:var(--text-muted);">/mes</span></div>
            <p style="font-size:0.85rem; color:var(--text-muted); margin:0;">Cancelación inmediata en cualquier momento.</p>
        </div>

        <button class="btn btn-primary glow-effect" style="width:100%; padding:14px; font-size:1.05rem;" onclick="alert('Funcionalidad de Stripe en desarrollo')">Continuar al Pago →</button>
    </div>
</div>
`;

export async function init() {
    // DOM
    const inputName = document.getElementById('input-name');
    const displayEmail = document.getElementById('display-email');
    const btnSaveName = document.getElementById('btn-save-name');
    const nameSpinner = document.getElementById('name-spinner');
    const nameFeedback = document.getElementById('name-feedback');

    const btnUpgrade = document.getElementById('btn-upgrade-plan');
    const upgradeModal = document.getElementById('upgrade-modal');
    const btnCloseUpgrade = document.getElementById('btn-close-upgrade');

    const step1Div = document.getElementById('close-account-step1');
    const step2Div = document.getElementById('close-account-step2');
    const btnCloseInit = document.getElementById('btn-close-account-init');
    const btnCloseCancel = document.getElementById('btn-close-account-cancel');
    const btnCloseConfirm = document.getElementById('btn-close-account-confirm');

    try {
        const user = await getCurrentUser();
        if (user) {
            displayEmail.textContent = user.email;

            // Fetch name
            const { data: profile } = await getSupabase()
                .from('profiles')
                .select('first_name')
                .eq('id', user.id)
                .single();

            if (profile?.first_name) {
                inputName.value = profile.first_name;
            }
        }

        // Inline Name Edit
        btnSaveName.addEventListener('click', async () => {
            const newName = inputName.value.trim();
            if (!newName) return;

            btnSaveName.style.display = 'none';
            nameSpinner.classList.remove('hidden');

            const { error } = await getSupabase()
                .from('profiles')
                .update({ first_name: newName })
                .eq('id', user.id);

            nameSpinner.classList.add('hidden');
            btnSaveName.style.display = 'block';

            if (!error) {
                // Actualizar name en header global
                const globalGreeting = document.getElementById('user-greeting');
                if (globalGreeting) globalGreeting.textContent = `Hola, ${newName}`;

                nameFeedback.style.opacity = '1';
                setTimeout(() => nameFeedback.style.opacity = '0', 3000);
            } else {
                nameFeedback.textContent = 'Error al guardar';
                nameFeedback.style.color = 'var(--danger)';
                nameFeedback.style.opacity = '1';
            }
        });

        // Upgrade Modal
        btnUpgrade.addEventListener('click', () => {
            upgradeModal.classList.remove('hidden');
        });
        btnCloseUpgrade.addEventListener('click', () => {
            upgradeModal.classList.add('hidden');
        });

        // Close Account logic
        btnCloseInit.addEventListener('click', () => {
            step1Div.classList.add('hidden');
            step2Div.classList.remove('hidden');
        });

        btnCloseCancel.addEventListener('click', () => {
            step2Div.classList.add('hidden');
            step1Div.classList.remove('hidden');
        });

        btnCloseConfirm.addEventListener('click', async () => {
            btnCloseConfirm.textContent = 'Eliminando...';
            btnCloseConfirm.disabled = true;

            // En un caso real llamaríamos a un RPC o Edge Function para borrar al user auth y base de datos.
            // Aqui emulamos con un simple toast y logout.
            setTimeout(() => {
                alert('Cuenta eliminada exitosamente en modo demostración.');
                window.location.href = './login.html';
            }, 1000);
        });

    } catch (err) {
        console.error('Account module error:', err);
    }
}
