'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import LoginForm from '@/app/components/LoginForm';
import RegisterForm from '@/components/auth/RegisterForm';
import { createBrowserClient } from '@supabase/ssr';

export default function AuthContainer() {
    const [activeTab, setActiveTab] = useState<'login' | 'register'>('login');

    const handleGoogleLogin = async () => {
        const { createClient } = await import('@/utils/supabase/client');
        const supabase = createClient();

        const { error } = await supabase.auth.signInWithOAuth({
            provider: 'google',
            options: {
                redirectTo: `${window.location.origin}/auth/callback?next=/dashboard`
            }
        });

        if (error) {
            console.error('Error logging in with Google:', error.message);
        }
    };

    return (
        <main className="auth-page">
            <div className="auth-card glass-panel" style={{ position: 'relative', zIndex: 10 }}>
                {/* Logo */}
                <Link href="/" className="auth-logo-link">
                    <Image src="/kaudal-logo2.png" alt="Kaudal" width={180} height={42} className="auth-logo" />
                </Link>

                {/* ===== TABS ===== */}
                <div className="auth-tabs" id="auth-tabs">
                    <button
                        className={`auth-tab ${activeTab === 'login' ? 'active' : ''}`}
                        onClick={() => setActiveTab('login')}
                    >
                        Iniciar sesión
                    </button>
                    <button
                        className={`auth-tab ${activeTab === 'register' ? 'active' : ''}`}
                        onClick={() => setActiveTab('register')}
                    >
                        Crear cuenta
                    </button>
                </div>

                {activeTab === 'login' ? <LoginForm /> : <RegisterForm />}

                {/* ===== DIVIDER ===== */}
                <div className="auth-divider" id="auth-divider">
                    <span>o</span>
                </div>

                {/* ===== GOOGLE OAUTH ===== */}
                <button className="auth-google-btn" id="btn-google" onClick={handleGoogleLogin} type="button">
                    <svg width="20" height="20" viewBox="0 0 48 48">
                        <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
                        <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
                        <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
                        <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
                    </svg>
                    <span>Continuar con Google</span>
                </button>
            </div>

            <footer className="auth-footer">
                <p>Al continuar, aceptas la <Link href="/privacidad">Política de Privacidad</Link> y los <Link href="/terminos">Términos del Servicio</Link>.</p>
                <p><Link href="/cookies">Póliza de Cookies</Link></p>
            </footer>
        </main>
    );
}
