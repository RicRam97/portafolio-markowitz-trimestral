// ══════════════════════════════════════════════════════════════
// session-manager.js — Gestión de sesión para Kaudal
// Logout por inactividad (30 min) + modal de advertencia (2 min antes)
// ══════════════════════════════════════════════════════════════

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

// ── Config ───────────────────────────────────────────────────
const INACTIVITY_TIMEOUT_MS = 30 * 60 * 1000;  // 30 minutes
const WARNING_BEFORE_MS = 2 * 60 * 1000;        // Show warning 2 min before
const COUNTDOWN_INTERVAL_MS = 1000;

// ── Supabase Client (singleton) ──────────────────────────────
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

let supabase = null;

function getSupabase() {
    if (!supabase) {
        supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    }
    return supabase;
}

// ── State ────────────────────────────────────────────────────
let inactivityTimer = null;
let warningTimer = null;
let countdownInterval = null;
let isWarningVisible = false;
let warningModal = null;
let countdownEl = null;

// ── Activity Events ──────────────────────────────────────────
const ACTIVITY_EVENTS = ['mousedown', 'mousemove', 'keydown', 'touchstart', 'scroll', 'click'];

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
export function initSessionManager(externalSupabase) {
    if (externalSupabase) supabase = externalSupabase;

    // Create warning modal if it doesn't exist
    ensureWarningModal();

    // Start tracking activity
    resetInactivityTimer();
    ACTIVITY_EVENTS.forEach(event => {
        document.addEventListener(event, onUserActivity, { passive: true });
    });

    // Listen for auth state changes
    getSupabase().auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_OUT') {
            cleanup();
        }
    });

    console.log('[Session] Manager initialized — 30 min inactivity timeout active.');
}

// ── Activity Handler ─────────────────────────────────────────
function onUserActivity() {
    if (isWarningVisible) return; // Don't reset during warning
    resetInactivityTimer();
}

// ── Timer Management ─────────────────────────────────────────
function resetInactivityTimer() {
    clearTimeout(inactivityTimer);
    clearTimeout(warningTimer);

    // Set warning timer (fires 2 min before timeout)
    warningTimer = setTimeout(() => {
        showWarningModal();
    }, INACTIVITY_TIMEOUT_MS - WARNING_BEFORE_MS);

    // Set logout timer
    inactivityTimer = setTimeout(() => {
        performAutoLogout();
    }, INACTIVITY_TIMEOUT_MS);
}

// ══════════════════════════════════════════════════════════════
// WARNING MODAL
// ══════════════════════════════════════════════════════════════
function ensureWarningModal() {
    if (document.getElementById('session-warning-modal')) {
        warningModal = document.getElementById('session-warning-modal');
        countdownEl = document.getElementById('session-countdown');
        return;
    }

    const overlay = document.createElement('div');
    overlay.id = 'session-warning-modal';
    overlay.className = 'session-warning-overlay hidden';
    overlay.innerHTML = `
    <div class="session-warning-card glass-panel">
      <div class="session-warning-icon">⏳</div>
      <h3>Tu sesión está a punto de expirar</h3>
      <p>Por seguridad, cerraremos tu sesión por inactividad en:</p>
      <div class="session-countdown" id="session-countdown">2:00</div>
      <div class="session-warning-actions">
        <button class="btn btn-cta session-btn-stay" id="btn-session-stay">
          Seguir conectado
        </button>
        <button class="btn btn-secondary session-btn-logout" id="btn-session-logout">
          Cerrar sesión
        </button>
      </div>
    </div>
  `;

    document.body.appendChild(overlay);
    warningModal = overlay;
    countdownEl = document.getElementById('session-countdown');

    // Event listeners
    document.getElementById('btn-session-stay').addEventListener('click', () => {
        hideWarningModal();
        resetInactivityTimer();
    });

    document.getElementById('btn-session-logout').addEventListener('click', () => {
        hideWarningModal();
        performAutoLogout();
    });
}

function showWarningModal() {
    isWarningVisible = true;
    warningModal.classList.remove('hidden');

    // Start countdown
    let remaining = WARNING_BEFORE_MS / 1000; // 120 seconds
    updateCountdownDisplay(remaining);

    clearInterval(countdownInterval);
    countdownInterval = setInterval(() => {
        remaining--;
        updateCountdownDisplay(remaining);
        if (remaining <= 0) {
            clearInterval(countdownInterval);
        }
    }, COUNTDOWN_INTERVAL_MS);
}

function hideWarningModal() {
    isWarningVisible = false;
    warningModal.classList.add('hidden');
    clearInterval(countdownInterval);
}

function updateCountdownDisplay(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    countdownEl.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ══════════════════════════════════════════════════════════════
// LOGOUT
// ══════════════════════════════════════════════════════════════
async function performAutoLogout() {
    hideWarningModal();
    cleanup();

    try {
        await getSupabase().auth.signOut();
    } catch (err) {
        console.error('[Session] Error during sign out:', err);
    }

    // Redirect to login with expired flag
    window.location.href = './login.html?expired=1';
}

function cleanup() {
    clearTimeout(inactivityTimer);
    clearTimeout(warningTimer);
    clearInterval(countdownInterval);

    ACTIVITY_EVENTS.forEach(event => {
        document.removeEventListener(event, onUserActivity);
    });
}

// ══════════════════════════════════════════════════════════════
// AUTH GUARD — Protect pages that require login
// ══════════════════════════════════════════════════════════════
export async function requireAuth() {
    const sb = getSupabase();
    const { data: { session } } = await sb.auth.getSession();

    if (!session) {
        window.location.href = './login.html';
        return null;
    }

    return session;
}

// ══════════════════════════════════════════════════════════════
// Get current user info
// ══════════════════════════════════════════════════════════════
export async function getCurrentUser() {
    const sb = getSupabase();
    const { data: { session } } = await sb.auth.getSession();
    return session?.user || null;
}

// ══════════════════════════════════════════════════════════════
// Logout function for nav buttons
// ══════════════════════════════════════════════════════════════
export async function logout() {
    cleanup();
    try {
        await getSupabase().auth.signOut();
    } catch (err) {
        console.error('[Session] Error during sign out:', err);
    }
    window.location.href = './login.html';
}

export { getSupabase };
