'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { z } from 'zod';

import { createClient } from '@/utils/supabase/client';

const schema = z.object({
    email: z.string().min(1, 'El correo es requerido').email('Ingresa un correo electrónico válido'),
});

export default function RecuperarPasswordPage() {
    const supabase = createClient();
    const [email, setEmail] = useState('');
    const [submitted, setSubmitted] = useState(false);
    const [loading, setLoading] = useState(false);
    const [errorMsg, setErrorMsg] = useState<string | null>(null);

    const onSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setErrorMsg(null);

        const validation = schema.safeParse({ email });
        if (!validation.success) {
            setErrorMsg(validation.error.format().email?._errors[0] || 'Error de validación');
            return;
        }

        setLoading(true);
        try {
            const redirectUrl = typeof window !== 'undefined'
                ? `${window.location.origin}/auth/callback?next=/nueva-password`
                : `${process.env.NEXT_PUBLIC_SITE_URL}/auth/callback?next=/nueva-password`;

            const { error } = await supabase.auth.resetPasswordForEmail(email, {
                redirectTo: redirectUrl,
            });

            if (error) {
                setErrorMsg(error.message);
                return;
            }

            setSubmitted(true);
        } catch (err: unknown) {
            setErrorMsg(err instanceof Error ? err.message : 'Ocurrió un error inesperado al enviar el correo.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-body">
            <div className="auth-bg-orbs">
                <div className="hero-orb hero-orb--blue"></div>
                <div className="hero-orb hero-orb--green"></div>
                <div className="hero-orb hero-orb--purple"></div>
            </div>

            <main className="auth-page">
                <div className="auth-card glass-panel" style={{ position: 'relative', zIndex: 10 }}>
                    <Link href="/" className="auth-logo-link">
                        <Image src="/kaudal-logo2.png" alt="Kaudal" width={180} height={42} className="auth-logo" />
                    </Link>

                    <h1 style={{ textAlign: 'center', fontSize: '1.5rem', marginBottom: '8px', color: 'var(--text-main)', fontFamily: 'var(--font-display)' }}>Recuperar contraseña</h1>

                    {!submitted ? (
                        <>
                            <p style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '32px', lineHeight: 1.5 }}>
                                Ingresa tu correo y te enviaremos un enlace para restablecer tu contraseña.
                            </p>

                            <form className="auth-form" onSubmit={onSubmit} noValidate style={{ display: 'flex', flexDirection: 'column' }}>
                                <div className="auth-field">
                                    <label htmlFor="email">Correo electrónico</label>
                                    <input
                                        type="email"
                                        id="email"
                                        placeholder="tu@email.com"
                                        required
                                        autoComplete="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        style={errorMsg ? { borderColor: 'var(--danger)' } : {}}
                                    />
                                    {errorMsg && <span className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{errorMsg}</span>}
                                </div>

                                <button type="submit" className="btn btn-cta auth-submit" disabled={loading} style={{ position: 'relative', marginTop: '16px' }}>
                                    <span style={{ opacity: loading ? 0 : 1 }}>Enviar enlace</span>
                                    {loading && (
                                        <div className="spinner" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', margin: 0, display: 'block' }}></div>
                                    )}
                                </button>
                            </form>
                        </>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '24px 0' }}>
                            <div style={{ fontSize: '3rem', marginBottom: '16px' }}>📬</div>
                            <h2 style={{ fontSize: '1.25rem', color: 'var(--text-main)', marginBottom: '8px' }}>Revisa tu correo</h2>
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: 1.5, marginBottom: '8px' }}>
                                Si ese correo está registrado en Kaudal, recibirás un enlace para restablecer tu contraseña en los próximos minutos.
                            </p>
                            <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', opacity: 0.7 }}>
                                Recuerda revisar también tu carpeta de spam.
                            </p>
                        </div>
                    )}

                    <div style={{ textAlign: 'center', marginTop: '32px' }}>
                        <Link href="/login" style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textDecoration: 'none' }}>
                            ← Volver al inicio de sesión
                        </Link>
                    </div>
                </div>
            </main>
        </div>
    );
}
