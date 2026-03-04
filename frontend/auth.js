// ══════════════════════════════════════════════════════════════
// auth.js — Sistema de autenticación completo para Kaudal
// Supabase JS SDK v2 · Vanilla JS · Mobile-first
// ══════════════════════════════════════════════════════════════

import { createClient } from 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/+esm';

// ── Supabase Client ──────────────────────────────────────────
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    console.error('[Auth] Variables VITE_SUPABASE_URL y VITE_SUPABASE_ANON_KEY son requeridas.');
}

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// ── DOM Elements ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Tabs
const tabLogin = $('#tab-login');
const tabRegister = $('#tab-register');
const formLogin = $('#form-login');
const formRegister = $('#form-register');
const authDivider = $('#auth-divider');
const btnGoogle = $('#btn-google');
const authMessage = $('#auth-message');

// Login form
const loginEmail = $('#login-email');
const loginPassword = $('#login-password');
const btnLogin = $('#btn-login');
const btnLoginText = $('#btn-login-text');
const loginSpinner = $('#login-spinner');

// Register form
const registerEmail = $('#register-email');
const registerPassword = $('#register-password');
const btnRegister = $('#btn-register');
const btnRegisterText = $('#btn-register-text');
const registerSpinner = $('#register-spinner');
const strengthBar = $('#strength-bar');
const passwordHint = $('#register-password-hint');

// Verify screen
const verifyScreen = $('#verify-screen');
const verifyEmailDisplay = $('#verify-email-display');
const btnResend = $('#btn-resend');
const btnResendText = $('#btn-resend-text');
const btnBackLogin = $('#btn-back-login');

// Legal modal
const legalModal = $('#legal-modal');
const legalCheckboxes = $$('.legal-cb');
const btnLegalContinue = $('#btn-legal-continue');
const btnLegalText = $('#btn-legal-text');
const legalSpinner = $('#legal-spinner');

// ── State ────────────────────────────────────────────────────
let currentTab = 'login';
let resendCooldown = 0;
let resendTimer = null;
let pendingUser = null; // User pending legal acceptance

// ── Error Messages Map ───────────────────────────────────────
const ERROR_MESSAGES = {
    'Invalid login credentials': 'Correo o contraseña incorrectos.',
    'Email not confirmed': 'Tu correo aún no está verificado. Revisa tu bandeja de entrada.',
    'User already registered': 'Este correo ya tiene una cuenta. Inicia sesión.',
    'Signup requires a valid password': 'La contraseña debe tener al menos 8 caracteres.',
    'Password should be at least 6 characters': 'La contraseña debe tener al menos 8 caracteres.',
    'To signup, please provide your email': 'Ingresa tu correo electrónico.',
    'Email rate limit exceeded': 'Has enviado demasiados correos. Intenta en unos minutos.',
    'For security purposes, you can only request this after': 'Espera un momento antes de intentar de nuevo.',
    'new row violates row-level security': 'Error de permisos. Intenta cerrar sesión y volver a entrar.',
};

function getErrorMessage(error) {
    if (!error) return 'Ocurrió un error inesperado.';
    const msg = error.message || error.msg || String(error);
    // Check for partial matches
    for (const [key, value] of Object.entries(ERROR_MESSAGES)) {
        if (msg.includes(key)) return value;
    }
    return msg;
}

// ── Validation ───────────────────────────────────────────────
function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function getPasswordStrength(password) {
    let score = 0;
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/[0-9]/.test(password)) score++;
    if (/[^A-Za-z0-9]/.test(password)) score++;
    return score; // 0-5
}

function showFieldError(id, message) {
    const el = $(`#${id}`);
    if (el) {
        el.textContent = message;
        el.classList.add('visible');
    }
}

function clearFieldError(id) {
    const el = $(`#${id}`);
    if (el) {
        el.textContent = '';
        el.classList.remove('visible');
    }
}

function showGlobalMessage(text, type = 'error') {
    authMessage.textContent = text;
    authMessage.className = `auth-message auth-message--${type}`;
    authMessage.classList.remove('hidden');
    // Auto-hide success messages
    if (type === 'success') {
        setTimeout(() => authMessage.classList.add('hidden'), 5000);
    }
}

function hideGlobalMessage() {
    authMessage.classList.add('hidden');
}

// ── Loading State ────────────────────────────────────────────
function setLoading(button, textEl, spinnerEl, loading) {
    if (loading) {
        button.disabled = true;
        textEl.style.opacity = '0';
        spinnerEl.classList.remove('hidden');
    } else {
        button.disabled = false;
        textEl.style.opacity = '1';
        spinnerEl.classList.add('hidden');
    }
}

// ══════════════════════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════════════════════
function switchTab(tab) {
    currentTab = tab;
    hideGlobalMessage();

    // Toggle active class
    tabLogin.classList.toggle('active', tab === 'login');
    tabRegister.classList.toggle('active', tab === 'register');

    // Toggle forms
    formLogin.classList.toggle('hidden', tab !== 'login');
    formRegister.classList.toggle('hidden', tab !== 'register');

    // Divider & Google always visible
    authDivider.classList.remove('hidden');
    btnGoogle.classList.remove('hidden');

    // Hide verify screen
    verifyScreen.classList.add('hidden');

    // Clear errors
    $$('.auth-field-error').forEach(el => { el.textContent = ''; el.classList.remove('visible'); });
}

tabLogin.addEventListener('click', () => switchTab('login'));
tabRegister.addEventListener('click', () => switchTab('register'));

// ══════════════════════════════════════════════════════════════
// REAL-TIME VALIDATION
// ══════════════════════════════════════════════════════════════
loginEmail.addEventListener('input', () => {
    clearFieldError('login-email-error');
    if (loginEmail.value && !isValidEmail(loginEmail.value)) {
        showFieldError('login-email-error', 'Ingresa un correo válido');
    }
});

loginPassword.addEventListener('input', () => {
    clearFieldError('login-password-error');
});

registerEmail.addEventListener('input', () => {
    clearFieldError('register-email-error');
    if (registerEmail.value && !isValidEmail(registerEmail.value)) {
        showFieldError('register-email-error', 'Ingresa un correo válido');
    }
});

registerPassword.addEventListener('input', () => {
    clearFieldError('register-password-error');
    const val = registerPassword.value;
    const strength = getPasswordStrength(val);

    // Update strength bar
    const pct = val.length === 0 ? 0 : (strength / 5) * 100;
    strengthBar.style.width = `${pct}%`;
    strengthBar.className = 'strength-bar';
    if (pct > 0) {
        if (strength <= 2) strengthBar.classList.add('weak');
        else if (strength <= 3) strengthBar.classList.add('medium');
        else strengthBar.classList.add('strong');
    }

    // Update hint
    if (val.length > 0 && val.length < 8) {
        passwordHint.textContent = `Faltan ${8 - val.length} caracteres`;
        passwordHint.style.color = 'var(--danger)';
    } else if (val.length >= 8) {
        passwordHint.textContent = strength >= 4 ? '¡Contraseña segura!' : 'Agrega mayúsculas, números o símbolos';
        passwordHint.style.color = strength >= 4 ? 'var(--success)' : 'var(--warning)';
    } else {
        passwordHint.textContent = 'Mínimo 8 caracteres';
        passwordHint.style.color = 'var(--text-muted)';
    }
});

// ══════════════════════════════════════════════════════════════
// PASSWORD TOGGLING
// ══════════════════════════════════════════════════════════════
function setupPasswordToggle(toggleBtn, input) {
    toggleBtn.addEventListener('click', () => {
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        toggleBtn.classList.toggle('showing', !isPassword);
    });
}
setupPasswordToggle($('#toggle-login-pass'), loginPassword);
setupPasswordToggle($('#toggle-register-pass'), registerPassword);

// ══════════════════════════════════════════════════════════════
// SIGN UP
// ══════════════════════════════════════════════════════════════
formRegister.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideGlobalMessage();

    const email = registerEmail.value.trim();
    const password = registerPassword.value;

    // Validate
    let valid = true;
    if (!email || !isValidEmail(email)) {
        showFieldError('register-email-error', 'Ingresa un correo electrónico válido');
        valid = false;
    }
    if (!password || password.length < 8) {
        showFieldError('register-password-error', 'La contraseña debe tener al menos 8 caracteres');
        valid = false;
    }
    if (!valid) return;

    setLoading(btnRegister, btnRegisterText, registerSpinner, true);

    try {
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
        });

        if (error) {
            showGlobalMessage(getErrorMessage(error));
            setLoading(btnRegister, btnRegisterText, registerSpinner, false);
            return;
        }

        // Check if email confirmation is required
        if (data.user && !data.user.confirmed_at && data.user.identities?.length > 0) {
            // User created, needs confirmation → show legal modal first, then verify screen
            pendingUser = data.user;
            showLegalModal();
        } else if (data.user && data.user.identities?.length === 0) {
            // User already exists
            showGlobalMessage('Este correo ya tiene una cuenta. Inicia sesión.', 'error');
        } else if (data.user && data.session) {
            // Auto-confirmed (e.g. email confirmation disabled)
            pendingUser = data.user;
            showLegalModal();
        }
    } catch (err) {
        showGlobalMessage('Error de conexión. Intenta de nuevo.');
    } finally {
        setLoading(btnRegister, btnRegisterText, registerSpinner, false);
    }
});

// ══════════════════════════════════════════════════════════════
// SIGN IN
// ══════════════════════════════════════════════════════════════
formLogin.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideGlobalMessage();

    const email = loginEmail.value.trim();
    const password = loginPassword.value;

    // Validate
    let valid = true;
    if (!email || !isValidEmail(email)) {
        showFieldError('login-email-error', 'Ingresa un correo electrónico válido');
        valid = false;
    }
    if (!password) {
        showFieldError('login-password-error', 'Ingresa tu contraseña');
        valid = false;
    }
    if (!valid) return;

    setLoading(btnLogin, btnLoginText, loginSpinner, true);

    try {
        const { data, error } = await supabase.auth.signInWithPassword({
            email,
            password,
        });

        if (error) {
            showGlobalMessage(getErrorMessage(error));
            setLoading(btnLogin, btnLoginText, loginSpinner, false);
            return;
        }

        // Success — redirect based on onboarding status
        await handlePostLogin(data.user);
    } catch (err) {
        showGlobalMessage('Error de conexión. Intenta de nuevo.');
        setLoading(btnLogin, btnLoginText, loginSpinner, false);
    }
});

// ══════════════════════════════════════════════════════════════
// GOOGLE OAUTH
// ══════════════════════════════════════════════════════════════
btnGoogle.addEventListener('click', async () => {
    hideGlobalMessage();

    try {
        const { error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: `${window.location.origin}/login.html?oauth_callback=1`,
            },
        });

        if (error) {
            showGlobalMessage(getErrorMessage(error));
        }
    } catch (err) {
        showGlobalMessage('Error al conectar con Google. Intenta de nuevo.');
    }
});

// ══════════════════════════════════════════════════════════════
// POST-LOGIN ROUTING
// ══════════════════════════════════════════════════════════════
async function handlePostLogin(user) {
    if (!user) return;

    try {
        // Check if user has accepted legal terms
        const { data: acceptances } = await supabase
            .from('legal_acceptances')
            .select('document')
            .eq('user_id', user.id);

        const acceptedDocs = (acceptances || []).map(a => a.document);
        const requiredDocs = ['terms', 'privacy', 'disclaimer'];
        const missingDocs = requiredDocs.filter(d => !acceptedDocs.includes(d));

        if (missingDocs.length > 0) {
            // Hasn't accepted all legal terms
            pendingUser = user;
            showLegalModal();
            return;
        }

        // Check onboarding status
        const { data: profile } = await supabase
            .from('profiles')
            .select('onboarding_complete')
            .eq('id', user.id)
            .single();

        if (profile && profile.onboarding_complete) {
            // Existing user → Dashboard
            window.location.href = './dashboard.html';
        } else {
            // New user → Dreams Test (onboarding)
            window.location.href = './index.html?view=dreams';
        }
    } catch (err) {
        // Default: go to dashboard
        window.location.href = './dashboard.html';
    }
}

// ══════════════════════════════════════════════════════════════
// LEGAL MODAL
// ══════════════════════════════════════════════════════════════
function showLegalModal() {
    legalModal.classList.remove('hidden');
    // Hide other UI
    formLogin.classList.add('hidden');
    formRegister.classList.add('hidden');
    authDivider.classList.add('hidden');
    btnGoogle.classList.add('hidden');
    $$('.auth-tab').forEach(t => t.style.display = 'none');
}

function hideLegalModal() {
    legalModal.classList.add('hidden');
}

// Enable/disable continue button based on checkboxes
legalCheckboxes.forEach(cb => {
    cb.addEventListener('change', () => {
        const allChecked = [...legalCheckboxes].every(c => c.checked);
        btnLegalContinue.disabled = !allChecked;
    });
});

btnLegalContinue.addEventListener('click', async () => {
    if (!pendingUser) return;

    setLoading(btnLegalContinue, btnLegalText, legalSpinner, true);

    try {
        const userId = pendingUser.id;
        const now = new Date().toISOString();

        // Insert legal acceptances via RPC to bypass RLS (since user has no session yet)
        const { error } = await supabase.rpc('accept_legal_documents', {
            p_user_id: userId,
            p_documents: ['terms', 'privacy', 'disclaimer', 'cookies'],
            p_version: '1.0'
        });

        if (error) {
            console.error('[Auth] Error saving legal acceptances:', error);
            showGlobalMessage('Error al guardar. Intenta de nuevo.');
            setLoading(btnLegalContinue, btnLegalText, legalSpinner, false);
            return;
        }

        hideLegalModal();

        // Si se acaba de registrar con correo, necesita verificarlo para tener una sesión
        const { data: { session } } = await supabase.auth.getSession();

        if (session) {
            // Ya tiene sesión válida (ej. Google OAuth o email confirmations off)
            await handlePostLogin(pendingUser);
        } else {
            // No tiene sesión = requiere confirmar el correo. Mostramos la pantalla de verificación.
            showVerifyScreen(pendingUser.email);
        }
    } catch (err) {
        showGlobalMessage('Error de conexión. Intenta de nuevo.');
        setLoading(btnLegalContinue, btnLegalText, legalSpinner, false);
    }
});

// ══════════════════════════════════════════════════════════════
// VERIFY EMAIL SCREEN
// ══════════════════════════════════════════════════════════════
function showVerifyScreen(email) {
    // Hide everything else in the card
    formLogin.classList.add('hidden');
    formRegister.classList.add('hidden');
    authDivider.classList.add('hidden');
    btnGoogle.classList.add('hidden');
    $$('.auth-tab').forEach(t => t.style.display = 'none');

    verifyScreen.classList.remove('hidden');
    verifyEmailDisplay.textContent = email;
    startResendCooldown();
}

function hideVerifyScreen() {
    verifyScreen.classList.add('hidden');
    $$('.auth-tab').forEach(t => t.style.display = '');
    switchTab('login');
}

function startResendCooldown() {
    resendCooldown = 60;
    btnResend.disabled = true;
    btnResendText.textContent = `Reenviar en ${resendCooldown}s`;

    clearInterval(resendTimer);
    resendTimer = setInterval(() => {
        resendCooldown--;
        if (resendCooldown <= 0) {
            clearInterval(resendTimer);
            btnResend.disabled = false;
            btnResendText.textContent = 'Reenviar correo';
        } else {
            btnResendText.textContent = `Reenviar en ${resendCooldown}s`;
        }
    }, 1000);
}

btnResend.addEventListener('click', async () => {
    const email = verifyEmailDisplay.textContent;
    if (!email || email === '—') return;

    btnResend.disabled = true;
    btnResendText.textContent = 'Enviando...';

    try {
        const { error } = await supabase.auth.resend({
            type: 'signup',
            email,
        });

        if (error) {
            showGlobalMessage(getErrorMessage(error));
        } else {
            showGlobalMessage('¡Correo reenviado! Revisa tu bandeja.', 'success');
        }
    } catch (err) {
        showGlobalMessage('Error al reenviar. Intenta de nuevo.');
    }

    startResendCooldown();
});

btnBackLogin.addEventListener('click', () => {
    hideVerifyScreen();
});

// ══════════════════════════════════════════════════════════════
// OAUTH CALLBACK HANDLING
// ══════════════════════════════════════════════════════════════
async function handleOAuthCallback() {
    const params = new URLSearchParams(window.location.search);

    if (params.get('oauth_callback') === '1' || window.location.hash.includes('access_token')) {
        // Wait for Supabase to process the token from the URL
        const { data: { session }, error } = await supabase.auth.getSession();

        if (session && session.user) {
            // Clean URL
            history.replaceState(null, '', window.location.pathname);
            await handlePostLogin(session.user);
        } else if (error) {
            showGlobalMessage(getErrorMessage(error));
        }
    }

    // Check for expired session redirect
    if (params.get('expired') === '1') {
        showGlobalMessage('Tu sesión expiró por inactividad. Inicia sesión de nuevo.', 'error');
        history.replaceState(null, '', window.location.pathname);
    }
}

// ══════════════════════════════════════════════════════════════
// AUTH STATE LISTENER
// ══════════════════════════════════════════════════════════════
supabase.auth.onAuthStateChange(async (event, session) => {
    if (event === 'SIGNED_IN' && session) {
        // If we're on the login page with a fresh sign-in, redirect
        const params = new URLSearchParams(window.location.search);
        if (!params.get('oauth_callback')) {
            // Regular sign in handled by form submit
            return;
        }
    }
});

// ══════════════════════════════════════════════════════════════
// INIT — Check if already logged in
// ══════════════════════════════════════════════════════════════
async function init() {
    // Handle OAuth callback first
    await handleOAuthCallback();

    // Check existing session
    const { data: { session } } = await supabase.auth.getSession();
    if (session && session.user) {
        // Already logged in, check if this is a callback
        const params = new URLSearchParams(window.location.search);
        if (!params.get('oauth_callback') && !params.get('expired')) {
            // Already authenticated — redirect
            await handlePostLogin(session.user);
        }
    }
}

init();
