'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { createBrowserClient } from '@supabase/ssr';
import toast, { Toaster } from 'react-hot-toast';
import Link from 'next/link';

// ── Supabase client ──────────────────────────────────────────
const createClient = () =>
    createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

// ── Zod schema ───────────────────────────────────────────────
const schema = z
    .object({
        password: z
            .string()
            .min(8, 'Mínimo 8 caracteres')
            .regex(/[A-Z]/, 'Debe incluir al menos 1 letra mayúscula')
            .regex(/[0-9]/, 'Debe incluir al menos 1 número'),
        confirm: z.string().min(1, 'Confirma tu contraseña'),
    })
    .refine((data) => data.password === data.confirm, {
        path: ['confirm'],
        message: 'Las contraseñas no coinciden',
    });

type FormValues = z.infer<typeof schema>;

// ── Page ─────────────────────────────────────────────────────
export default function NuevaPasswordPage() {
    const router = useRouter();
    const supabase = createClient();

    const [sessionValid, setSessionValid] = useState<boolean | null>(null); // null = loading
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirm, setShowConfirm] = useState(false);

    const {
        register,
        handleSubmit,
        watch,
        formState: { errors },
    } = useForm<FormValues>({ resolver: zodResolver(schema) });

    const passwordValue = watch('password', '');

    // ── Verify token on mount ─────────────────────────────────
    useEffect(() => {
        const verifySession = async () => {
            // Supabase processes the hash from the URL automatically when the client initializes.
            // We just need to check if a valid session now exists.
            const { data: { session } } = await supabase.auth.getSession();
            setSessionValid(!!session);
        };
        verifySession();
    }, []);

    // ── Submit handler ────────────────────────────────────────
    const onSubmit = async ({ password }: FormValues) => {
        setLoading(true);
        try {
            const { error } = await supabase.auth.updateUser({ password });

            if (error) {
                toast.error(error.message || 'Ocurrió un error. Intenta de nuevo.');
                setLoading(false);
                return;
            }

            // Sign out to clean up the recovery session
            await supabase.auth.signOut();

            toast.success('¡Contraseña actualizada exitosamente!');
            setTimeout(() => router.push('/login'), 1500);
        } catch (_) {
            toast.error('Error de conexión. Intenta de nuevo.');
            setLoading(false);
        }
    };

    // ── Strength indicator helper ─────────────────────────────
    const getStrength = (val: string) => {
        let score = 0;
        if (val.length >= 8) score++;
        if (val.length >= 12) score++;
        if (/[A-Z]/.test(val)) score++;
        if (/[0-9]/.test(val)) score++;
        if (/[^A-Za-z0-9]/.test(val)) score++;
        return score;
    };

    const strength = getStrength(passwordValue);
    const strengthColor =
        strength <= 2 ? 'bg-red-500' : strength <= 3 ? 'bg-yellow-500' : 'bg-emerald-500';
    const strengthLabel =
        strength <= 2 ? 'Débil' : strength <= 3 ? 'Media' : '¡Segura!';

    // ── Loading state ─────────────────────────────────────────
    if (sessionValid === null) {
        return (
            <main className="min-h-screen flex items-center justify-center bg-[#0B1120]">
                <svg className="animate-spin h-8 w-8 text-blue-400" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
            </main>
        );
    }

    return (
        <main className="min-h-screen relative flex items-center justify-center bg-[#0B1120] px-4 overflow-hidden">
            {/* React-hot-toast container */}
            <Toaster
                position="top-center"
                toastOptions={{
                    style: {
                        background: '#1e293b',
                        color: '#f1f5f9',
                        border: '1px solid rgba(255,255,255,0.08)',
                    },
                }}
            />

            {/* Background orbs */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] left-[-5%] w-[400px] h-[400px] rounded-full bg-blue-600/30 blur-[80px]" />
                <div className="absolute bottom-[-5%] right-[10%] w-[300px] h-[300px] rounded-full bg-emerald-500/20 blur-[80px]" />
            </div>

            <div className="relative z-10 w-full max-w-[440px] bg-slate-900/75 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl p-8 pb-10 flex flex-col items-center">

                {/* ── INVALID TOKEN STATE ────────────────────────── */}
                {!sessionValid ? (
                    <div className="text-center flex flex-col items-center gap-4">
                        <div className="text-5xl">⚠️</div>
                        <h1 className="text-xl font-bold text-white font-[family-name:var(--font-outfit)]">
                            Enlace inválido o expirado
                        </h1>
                        <p className="text-slate-400 text-sm leading-relaxed max-w-xs">
                            Este enlace de recuperación ya no es válido. Los enlaces expiran después de 24 horas.
                        </p>
                        <Link
                            href="/recuperar-password"
                            className="mt-4 w-full text-center bg-[#2563EB] text-white p-3.5 rounded-xl font-semibold hover:bg-[#1D4ED8] transition-all"
                        >
                            Solicitar nuevo enlace
                        </Link>
                        <Link
                            href="/login"
                            className="text-sm text-slate-500 hover:text-slate-300 transition-colors"
                        >
                            ← Volver al inicio de sesión
                        </Link>
                    </div>
                ) : (
                    /* ── VALID TOKEN — SHOW FORM ─────────────────── */
                    <>
                        <div className="text-5xl mb-5">🔒</div>
                        <h1 className="text-2xl font-bold text-white font-[family-name:var(--font-outfit)] mb-2 text-center">
                            Nueva contraseña
                        </h1>
                        <p className="text-slate-400 text-sm text-center mb-8">
                            Elige una contraseña segura para tu cuenta.
                        </p>

                        <form onSubmit={handleSubmit(onSubmit)} className="w-full flex flex-col gap-5">

                            {/* Nueva contraseña */}
                            <div className="flex flex-col gap-1.5">
                                <label htmlFor="password" className="text-sm font-medium text-slate-300">
                                    Nueva contraseña
                                </label>
                                <div className="relative">
                                    <input
                                        id="password"
                                        type={showPassword ? 'text' : 'password'}
                                        placeholder="Mínimo 8 caracteres"
                                        autoComplete="new-password"
                                        {...register('password')}
                                        className={`w-full bg-slate-900/50 border rounded-lg p-3 pr-11 text-sm text-white outline-none transition-colors placeholder-slate-500
                      ${errors.password
                                                ? 'border-red-500 focus:ring-1 focus:ring-red-500'
                                                : 'border-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
                                            }
                    `}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                                        aria-label="Mostrar contraseña"
                                    >
                                        {showPassword ? (
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" /><line x1="1" y1="1" x2="23" y2="23" />
                                            </svg>
                                        ) : (
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
                                            </svg>
                                        )}
                                    </button>
                                </div>

                                {/* Strength bar */}
                                {passwordValue.length > 0 && (
                                    <div className="flex flex-col gap-1 mt-1">
                                        <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full rounded-full transition-all duration-300 ${strengthColor}`}
                                                style={{ width: `${(strength / 5) * 100}%` }}
                                            />
                                        </div>
                                        <span className={`text-xs font-medium ${strength <= 2 ? 'text-red-400' : strength <= 3 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                                            {strengthLabel}
                                        </span>
                                    </div>
                                )}
                                {errors.password && (
                                    <span className="text-xs text-red-400">{errors.password.message}</span>
                                )}
                            </div>

                            {/* Confirmar contraseña */}
                            <div className="flex flex-col gap-1.5">
                                <label htmlFor="confirm" className="text-sm font-medium text-slate-300">
                                    Confirmar contraseña
                                </label>
                                <div className="relative">
                                    <input
                                        id="confirm"
                                        type={showConfirm ? 'text' : 'password'}
                                        placeholder="Repite tu nueva contraseña"
                                        autoComplete="new-password"
                                        {...register('confirm')}
                                        className={`w-full bg-slate-900/50 border rounded-lg p-3 pr-11 text-sm text-white outline-none transition-colors placeholder-slate-500
                      ${errors.confirm
                                                ? 'border-red-500 focus:ring-1 focus:ring-red-500'
                                                : 'border-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
                                            }
                    `}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowConfirm(!showConfirm)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                                        aria-label="Mostrar contraseña"
                                    >
                                        {showConfirm ? (
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" /><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" /><line x1="1" y1="1" x2="23" y2="23" />
                                            </svg>
                                        ) : (
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
                                            </svg>
                                        )}
                                    </button>
                                </div>
                                {errors.confirm && (
                                    <span className="text-xs text-red-400">{errors.confirm.message}</span>
                                )}
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="mt-2 w-full bg-[#2563EB] text-white p-3.5 rounded-xl font-semibold hover:bg-[#1D4ED8] transition-all shadow-[0_4px_12px_rgba(37,99,235,0.3)] hover:-translate-y-px flex justify-center items-center disabled:opacity-75 disabled:cursor-not-allowed disabled:transform-none"
                            >
                                {loading ? (
                                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                ) : (
                                    'Actualizar contraseña'
                                )}
                            </button>
                        </form>
                    </>
                )}
            </div>
        </main>
    );
}
