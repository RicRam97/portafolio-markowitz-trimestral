// ══════════════════════════════════════════════════════════════
// router.js — Controlador de enrutamiento SPA para el Dashboard
// ══════════════════════════════════════════════════════════════

import { initSessionManager, getSupabase, getCurrentUser, logout } from './session-manager.js';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const contentArea = $('#dashboard-content');
const navItems = $$('.nav-item');
const mobileOverlay = $('#mobile-menu-overlay');

// Map of route modules loading functions
const MODULES = {
    dashboard: async () => {
        const { template, init } = await import('./dashboard-metrics.js');
        contentArea.innerHTML = template;
        if (init) await init();
    },
    configuracion: async () => {
        const { template, init } = await import('./config.js');
        contentArea.innerHTML = template;
        if (init) await init();
    },
    cuenta: async () => {
        const { template, init } = await import('./cuenta.js');
        contentArea.innerHTML = template;
        if (init) await init();
    },
    estrategias: async () => {
        contentArea.innerHTML = `
            <div class="module-container">
                <div class="module-header">
                    <h2>Estrategias Guardadas</h2>
                    <p>Accede rápidamente a tus portafolios previos</p>
                </div>
                <div class="glass-panel" style="padding:24px; text-align:center;">
                    <div style="font-size:3rem; margin-bottom:16px;">📚</div>
                    <h3>Próximamente</h3>
                    <p style="color:var(--text-muted);">Estamos construyendo esta funcionalidad para que puedas comparar y dar seguimiento histórico a tus optimizaciones.</p>
                </div>
            </div>
        `;
    },
    perfil: async () => {
        contentArea.innerHTML = `
            <div class="module-container">
                <div class="module-header">
                    <h2>Perfil de Inversor</h2>
                    <p>Resultados de tu cuestionario</p>
                </div>
                <!-- To be implemented with user's specific answers from onboarding -->
                <div class="glass-panel" style="padding:24px;">
                    En construcción...
                </div>
            </div>
        `;
    },
    aprende: async () => {
        contentArea.innerHTML = `
            <div class="module-container">
                <div class="module-header">
                    <h2>Aprende</h2>
                    <p>Conceptos clave de la inversión y la teoría moderna</p>
                </div>
                <div class="glass-panel" style="padding:24px;">
                    En construcción...
                </div>
            </div>
        `;
    }
};

/**
 * Loads the current active route from the URL hash.
 */
async function loadRoute() {
    let hash = window.location.hash.substring(1);

    // Default route
    if (!hash || !MODULES[hash]) {
        hash = 'dashboard';
        window.location.hash = hash;
        return; // Event listener will catch the hash change
    }

    // Determine loading state
    contentArea.innerHTML = `
        <div class="loading-state" style="display:flex; justify-content:center; align-items:center; height: 100%; min-height: 400px;">
            <div class="spinner"></div>
        </div>
    `;

    // Visual updates for Navigation
    navItems.forEach(item => {
        if (item.dataset.module === hash) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    mobileOverlay.classList.add('hidden'); // Ensure mobile menu is closed when navigating

    try {
        await MODULES[hash]();
    } catch (err) {
        console.error('[Router] Error cargando módulo:', err);
        contentArea.innerHTML = `
            <div class="module-container">
                <div class="error-toast visible" style="position:relative; transform:none; top:0; left:0; width:100%;">
                    Error al cargar este módulo. Por favor, reporta el bug si persiste.
                </div>
            </div>
        `;
    }
}

/**
 * Initialization procedure
 */
async function initRouter() {
    initSessionManager(getSupabase());

    // Auth Check
    const user = await getCurrentUser();
    if (!user) {
        window.location.href = './login.html';
        return;
    }

    // Set Greeting & Initial metadata
    const firstName = user.user_metadata?.first_name
        || (user.email ? user.email.split('@')[0] : 'Inversor');

    const greetingEl = $('#user-greeting');
    if (greetingEl) {
        greetingEl.textContent = `Hola, ${firstName}`;
    }

    // Mobile Hamburger
    const btnMobileMore = $('#btn-mobile-more');
    if (btnMobileMore) {
        btnMobileMore.addEventListener('click', () => mobileOverlay.classList.remove('hidden'));
    }

    const btnCloseMobile = $('#btn-close-mobile-menu');
    if (btnCloseMobile) {
        btnCloseMobile.addEventListener('click', () => mobileOverlay.classList.add('hidden'));
    }

    // Header Logout
    const btnLogout = $('#btn-logout');
    if (btnLogout) {
        btnLogout.addEventListener('click', async () => await logout());
    }

    // Mobile Hamburger Logout
    const btnLogoutMobile = $('#btn-logout-mobile');
    if (btnLogoutMobile) {
        btnLogoutMobile.addEventListener('click', async () => await logout());
    }

    // Hash Listening
    window.addEventListener('hashchange', loadRoute);

    // Bootstrap first load
    loadRoute();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initRouter);
} else {
    initRouter();
}
