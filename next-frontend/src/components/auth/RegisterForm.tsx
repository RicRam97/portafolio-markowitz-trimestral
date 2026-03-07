'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import { z } from 'zod';
import { useNotification } from '@/hooks/useNotification';
import { getErrorMessage, formatErrorToast } from '@/utils/errorMessages';

const createClient = () =>
    createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

const registerSchema = z.object({
    nombre: z.string().min(2, 'Ingresa tu nombre'),
    apellido: z.string().min(2, 'Ingresa tu apellido'),
    fecha_nacimiento: z.string().min(1, 'La fecha es requerida'),
    email: z.string().min(1, 'El correo es requerido').email('Ingresa un correo válido'),
    password: z.string().min(8, 'Mínimo 8 caracteres'),
    passwordConfirm: z.string()
}).refine((data) => data.password === data.passwordConfirm, {
    message: "Las contraseñas no coinciden",
    path: ["passwordConfirm"],
});

export default function RegisterForm() {
    const router = useRouter();
    const supabase = createClient();

    const [formData, setFormData] = useState({
        nombre: '',
        apellido: '',
        fecha_nacimiento: '',
        email: '',
        password: '',
        passwordConfirm: ''
    });

    const [showPassword, setShowPassword] = useState(false);
    const [showPasswordConfirm, setShowPasswordConfirm] = useState(false);

    const [loading, setLoading] = useState(false);
    const [errorMSG, setErrorMSG] = useState<string | null>(null);
    const [successMSG, setSuccessMSG] = useState<string | null>(null);
    const [fieldErrors, setFieldErrors] = useState<Partial<Record<keyof typeof formData, string>>>({});
    const notify = useNotification();

    const handleRegister = async (e: React.FormEvent) => {
        e.preventDefault();

        setErrorMSG(null);
        setSuccessMSG(null);
        setFieldErrors({});

        const validation = registerSchema.safeParse(formData);
        if (!validation.success) {
            const formattedErrors = validation.error.format();
            setFieldErrors({
                nombre: formattedErrors.nombre?._errors[0],
                apellido: formattedErrors.apellido?._errors[0],
                fecha_nacimiento: formattedErrors.fecha_nacimiento?._errors[0],
                email: formattedErrors.email?._errors[0],
                password: formattedErrors.password?._errors[0],
                passwordConfirm: formattedErrors.passwordConfirm?._errors[0],
            });
            return;
        }

        setLoading(true);

        try {
            const { data, error } = await supabase.auth.signUp({
                email: formData.email,
                password: formData.password,
                options: {
                    data: {
                        first_name: formData.nombre,
                        last_name: formData.apellido,
                        fechanacimiento: formData.fecha_nacimiento
                    }
                }
            });

            if (error) {
                if (error.message.includes('already registered')) {
                    const em = getErrorMessage('AUTH_USER_EXISTS');
                    setErrorMSG(em.message);
                    notify.error(formatErrorToast(em));
                } else {
                    const em = getErrorMessage('UNKNOWN');
                    setErrorMSG(em.message);
                    notify.error(formatErrorToast(em));
                }
            } else if (data.user) {
                setSuccessMSG('Cuenta creada! Revisa tu correo electronico para confirmar tu registro.');
                notify.success('Cuenta creada. Revisa tu correo para confirmar.');
                setFormData({ nombre: '', apellido: '', fecha_nacimiento: '', email: '', password: '', passwordConfirm: '' });
            }
        } catch (err) {
            const em = getErrorMessage('NETWORK_ERROR');
            setErrorMSG(em.message);
            notify.error(formatErrorToast(em));
        } finally {
            setLoading(false);
        }
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    return (
        <form className="auth-form" onSubmit={handleRegister} noValidate style={{ display: 'flex', flexDirection: 'column' }}>
            <div className="auth-field">
                <label htmlFor="register-nombre">Nombre</label>
                <input
                    type="text"
                    id="register-nombre"
                    name="nombre"
                    placeholder="Tu nombre"
                    required
                    autoComplete="given-name"
                    value={formData.nombre}
                    onChange={handleChange}
                    aria-invalid={!!fieldErrors.nombre}
                    aria-describedby={fieldErrors.nombre ? 'register-nombre-error' : undefined}
                    style={fieldErrors.nombre ? { borderColor: 'var(--danger)' } : {}}
                />
                {fieldErrors.nombre && <span id="register-nombre-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.nombre}</span>}
            </div>

            <div className="auth-field">
                <label htmlFor="register-apellido">Apellido</label>
                <input
                    type="text"
                    id="register-apellido"
                    name="apellido"
                    placeholder="Tu apellido"
                    required
                    autoComplete="family-name"
                    value={formData.apellido}
                    onChange={handleChange}
                    aria-invalid={!!fieldErrors.apellido}
                    aria-describedby={fieldErrors.apellido ? 'register-apellido-error' : undefined}
                    style={fieldErrors.apellido ? { borderColor: 'var(--danger)' } : {}}
                />
                {fieldErrors.apellido && <span id="register-apellido-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.apellido}</span>}
            </div>

            <div className="auth-field">
                <label htmlFor="register-fecha">Fecha de nacimiento</label>
                <input
                    type="date"
                    id="register-fecha"
                    name="fecha_nacimiento"
                    required
                    value={formData.fecha_nacimiento}
                    onChange={handleChange}
                    aria-invalid={!!fieldErrors.fecha_nacimiento}
                    aria-describedby={fieldErrors.fecha_nacimiento ? 'register-fecha-error' : undefined}
                    style={fieldErrors.fecha_nacimiento ? { borderColor: 'var(--danger)' } : {}}
                />
                {fieldErrors.fecha_nacimiento && <span id="register-fecha-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.fecha_nacimiento}</span>}
            </div>

            <div className="auth-field">
                <label htmlFor="register-email">Correo electrónico</label>
                <input
                    type="email"
                    id="register-email"
                    name="email"
                    placeholder="tu@email.com"
                    required
                    autoComplete="email"
                    value={formData.email}
                    onChange={handleChange}
                    aria-invalid={!!fieldErrors.email}
                    aria-describedby={fieldErrors.email ? 'register-email-error' : undefined}
                    style={fieldErrors.email ? { borderColor: 'var(--danger)' } : {}}
                />
                {fieldErrors.email && <span id="register-email-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.email}</span>}
            </div>

            <div className="auth-field">
                <label htmlFor="register-password">Contraseña</label>
                <div className="auth-password-wrapper">
                    <input
                        type={showPassword ? "text" : "password"}
                        id="register-password"
                        name="password"
                        placeholder="Mínimo 8 caracteres"
                        required
                        minLength={8}
                        autoComplete="new-password"
                        value={formData.password}
                        onChange={handleChange}
                        aria-invalid={!!fieldErrors.password}
                        aria-describedby={fieldErrors.password ? 'register-password-error' : 'register-password-hint'}
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
                {fieldErrors.password ? (
                    <span id="register-password-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.password}</span>
                ) : (
                    <span id="register-password-hint" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>Mínimo 8 caracteres, 1 mayúscula, 1 número</span>
                )}
            </div>

            <div className="auth-field">
                <label htmlFor="register-password-confirm">Confirmar contraseña</label>
                <div className="auth-password-wrapper">
                    <input
                        type={showPasswordConfirm ? "text" : "password"}
                        id="register-password-confirm"
                        name="passwordConfirm"
                        placeholder="Confirma tu contraseña"
                        required
                        autoComplete="new-password"
                        value={formData.passwordConfirm}
                        onChange={handleChange}
                        aria-invalid={!!fieldErrors.passwordConfirm}
                        aria-describedby={fieldErrors.passwordConfirm ? 'register-confirm-error' : undefined}
                        style={fieldErrors.passwordConfirm ? { borderColor: 'var(--danger)' } : {}}
                    />
                    <button
                        type="button"
                        className="auth-password-toggle"
                        onClick={() => setShowPasswordConfirm(!showPasswordConfirm)}
                        aria-label="Mostrar contraseña"
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            {showPasswordConfirm ? (
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
                {fieldErrors.passwordConfirm && <span id="register-confirm-error" role="alert" className="auth-field-error" style={{ display: 'block', color: 'var(--danger)', fontSize: '0.8rem', marginTop: '4px' }}>{fieldErrors.passwordConfirm}</span>}
            </div>

            {errorMSG && (
                <div role="alert" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid var(--danger)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p style={{ color: '#fca5a5', fontSize: '0.85rem' }}>{errorMSG}</p>
                </div>
            )}

            {successMSG && (
                <div role="status" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid var(--success)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                    <p style={{ color: '#6ee7b7', fontSize: '0.85rem' }}>{successMSG}</p>
                </div>
            )}

            <button type="submit" className="btn btn-cta auth-submit" disabled={loading} aria-disabled={loading} style={{ position: 'relative' }}>
                <span style={{ opacity: loading ? 0 : 1 }}>Crear cuenta</span>
                {loading && (
                    <div className="spinner" style={{ position: 'absolute', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', margin: 0, display: 'block' }}></div>
                )}
            </button>
        </form>
    );
}
