'use client';

import { useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import { z } from 'zod';
import Link from 'next/link';

const createClient = () =>
    createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

const loginSchema = z.object({
    email: z.string().min(1, 'El correo es requerido').email('Ingresa un correo válido'),
    password: z.string().min(1, 'Ingresa tu contraseña'),
});

export default function LoginForm() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const reason = searchParams.get('reason');
    const supabase = createClient();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);

    const [loading, setLoading] = useState(false);
    const [errorMSG, setErrorMSG] = useState<string | null>(null);
    const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});

    const [needsConfirmation, setNeedsConfirmation] = useState(false);
    const [resendSuccess, setResendSuccess] = useState(false);

    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();

        setErrorMSG(null);
        setFieldErrors({});
        setNeedsConfirmation(false);
        setResendSuccess(false);

        const validation = loginSchema.safeParse({ email, password });
        if (!validation.success) {
            const formattedErrors = validation.error.format();
            setFieldErrors({
                email: formattedErrors.email?._errors[0],
                password: formattedErrors.password?._errors[0],
            });
            return;
        }

        setLoading(true);

        try {
            const { data, error } = await supabase.auth.signInWithPassword({
                email,
                password,
            });

            if (error) {
                const errMsg = error.message;

                if (errMsg.includes('Invalid login credentials')) {
                    setErrorMSG('Correo o contraseña incorrectos.');
                } else if (errMsg.includes('Email not confirmed')) {
                    setErrorMSG('Debes confirmar tu correo antes de entrar.');
                    setNeedsConfirmation(true);
                } else {
                    setErrorMSG('Error de conexión. Intenta de nuevo.');
                }
            } else if (data.session) {
                router.push('/dashboard');
                router.refresh();
            }
        } catch (err) {
            setErrorMSG('Error de conexión. Intenta de nuevo.');
        } finally {
            setLoading(false);
        }
    };

    const handleResendConfirmation = async () => {
        setLoading(true);
        setErrorMSG(null);
        setResendSuccess(false);

        try {
            const { error } = await supabase.auth.resend({
                type: 'signup',
                email,
            });

            if (error) {
                setErrorMSG('Error al reenviar el correo. Intenta de nuevo.');
            } else {
                setResendSuccess(true);
                setNeedsConfirmation(false);
            }
        } catch (err) {
            setErrorMSG('Error de conexión. Intenta de nuevo.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <form className="auth-form" onSubmit={handleLogin} noValidate style={{ display: 'flex', flexDirection: 'column' }}>
            {reason === 'inactividad' && (
                <div style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', border: '1px solid var(--accent-primary)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p style={{ color: 'var(--accent-primary)', fontSize: '0.85rem' }}>Tu sesión cerró automáticamente por seguridad.</p>
                </div>
            )}

            <div className="auth-field">
                <label htmlFor="login-email">Correo electrónico</label>
                <input
                    type="email"
                    id="login-email"
                    placeholder="tu@email.com"
                    required
                    autoComplete="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    style={fieldErrors.email ? { borderColor: 'var(--danger)' } : {}}
                />
                {fieldErrors.email && <span className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.email}</span>}
            </div>

            <div className="auth-field">
                <label htmlFor="login-password">Contraseña</label>
                <div className="auth-password-wrapper">
                    <input
                        type={showPassword ? "text" : "password"}
                        id="login-password"
                        placeholder="Tu contraseña"
                        required
                        minLength={8}
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        style={fieldErrors.password ? { borderColor: 'var(--danger)' } : {}}
                    />
                    <button
                        type="button"
                        className="auth-password-toggle"
                        onClick={() => setShowPassword(!showPassword)}
                        aria-label="Mostrar contraseña"
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            {showPassword ? (
                                <>
                                    <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24" />
                                    <line x1="1" y1="1" x2="23" y2="23" />
                                </>
                            ) : (
                                <>
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                                    <circle cx="12" cy="12" r="3" />
                                </>
                            )}
                        </svg>
                    </button>
                </div>
                {fieldErrors.password && <span className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.password}</span>}

                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '8px' }}>
                    <Link href="/recuperar-password" style={{ fontSize: '0.85rem', color: 'var(--accent-primary)', textDecoration: 'none' }}>
                        ¿Olvidaste tu contraseña?
                    </Link>
                </div>
            </div>

            {errorMSG && (
                <div style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p style={{ color: '#fca5a5', fontSize: '0.85rem' }}>{errorMSG}</p>
                    {needsConfirmation && (
                        <button
                            type="button"
                            onClick={handleResendConfirmation}
                            style={{ background: 'none', border: 'none', color: '#fca5a5', textDecoration: 'underline', marginTop: '8px', cursor: 'pointer', fontWeight: 'bold' }}
                        >
                            ¿Reenviar correo?
                        </button>
                    )}
                </div>
            )}

            {resendSuccess && (
                <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid var(--success)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p style={{ color: '#6ee7b7', fontSize: '0.85rem' }}>✅ Correo reenviado exitosamente. Revisa tu bandeja de entrada o spam.</p>
                </div>
            )}

            <button type="submit" className="btn btn-cta auth-submit" disabled={loading} style={{ position: 'relative' }}>
                <span style={{ opacity: loading ? 0 : 1 }}>Iniciar sesión</span>
                {loading && (
                    <div className="spinner" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', margin: 0, display: 'block' }}></div>
                )}
            </button>
        </form>
    );
}
