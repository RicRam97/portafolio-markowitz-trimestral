'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Send, Lock, ArrowUpRight, ShieldCheck } from 'lucide-react';
import { createClient } from '@/utils/supabase/client';
import { toast } from 'sonner';
import type { PlanTier } from '@/lib/types';
import { PLAN_LABELS } from '@/lib/constants';
import { getErrorMessage, formatErrorToast } from '@/utils/errorMessages';

const ASUNTO_OPTIONS = [
    'Problema técnico',
    'Pregunta sobre mi cuenta',
    'Error en optimización',
    'Solicitud de funcionalidad',
    'Otro',
];

const RESPONSE_TIME: Record<string, string> = {
    pro: '48 horas',
    ultra: '24 horas',
};

interface Props {
    userPlan: PlanTier | null;
    userId?: string;
}

export default function ContactSection({ userPlan, userId }: Props) {
    const [asunto, setAsunto] = useState('');
    const [mensaje, setMensaje] = useState('');
    const [sending, setSending] = useState(false);
    const [sent, setSent] = useState(false);
    const [errors, setErrors] = useState<{ asunto?: string; mensaje?: string }>({});

    const supabase = createClient();

    // Not logged in
    if (!userPlan) {
        return (
            <section>
                <h2 className="text-lg font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                    Contacto
                </h2>
                <div className="glass-panel p-6 text-center">
                    <Lock className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--text-muted)' }} />
                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-main)' }}>
                        Inicia sesión para contactar soporte
                    </p>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                        Necesitas una cuenta para enviar un ticket de soporte.
                    </p>
                    <Link
                        href="/login"
                        className="inline-flex items-center gap-1.5 text-sm font-semibold px-4 py-2 rounded-lg transition-colors"
                        style={{ background: 'var(--accent-primary)', color: 'white' }}
                    >
                        Iniciar sesión <ArrowUpRight className="w-4 h-4" />
                    </Link>
                </div>
            </section>
        );
    }

    // Basic plan — upgrade CTA
    if (userPlan === 'basico') {
        return (
            <section>
                <h2 className="text-lg font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                    Contacto
                </h2>
                <div className="glass-panel p-6 text-center">
                    <ShieldCheck className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--accent-secondary)' }} />
                    <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-main)' }}>
                        Soporte directo disponible en planes Pro y Ultra
                    </p>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                        Tu plan {PLAN_LABELS[userPlan]} incluye acceso al FAQ. Actualiza para contactar directamente a nuestro equipo.
                    </p>
                    <Link
                        href="/planes"
                        className="inline-flex items-center gap-1.5 text-sm font-semibold px-4 py-2 rounded-lg transition-colors"
                        style={{ background: 'var(--accent-primary)', color: 'white' }}
                    >
                        Ver planes <ArrowUpRight className="w-4 h-4" />
                    </Link>
                </div>
            </section>
        );
    }

    // Pro or Ultra — contact form
    const validate = () => {
        const newErrors: { asunto?: string; mensaje?: string } = {};
        if (!asunto) newErrors.asunto = 'Selecciona un asunto';
        if (!mensaje.trim()) newErrors.mensaje = 'Escribe tu mensaje';
        else if (mensaje.trim().length < 10) newErrors.mensaje = 'El mensaje debe tener al menos 10 caracteres';
        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!validate() || !userId) return;

        setSending(true);
        const { error } = await supabase.from('support_tickets').insert({
            user_id: userId,
            asunto,
            mensaje: mensaje.trim(),
            tier: userPlan,
        });

        setSending(false);

        if (error) {
            const em = getErrorMessage('SUPPORT_TICKET_FAILED');
            toast.error(formatErrorToast(em));
            return;
        }

        setSent(true);
        toast.success(`Ticket enviado. Responderemos en ${RESPONSE_TIME[userPlan]}.`);
    };

    if (sent) {
        return (
            <section>
                <h2 className="text-lg font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                    Contacto
                </h2>
                <div className="glass-panel p-6 text-center">
                    <ShieldCheck className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--success)' }} />
                    <p className="text-sm font-semibold mb-1" style={{ color: 'var(--text-main)' }}>
                        Ticket enviado correctamente
                    </p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        Nuestro equipo te responderá en un máximo de {RESPONSE_TIME[userPlan]}.
                    </p>
                    <button
                        onClick={() => { setSent(false); setAsunto(''); setMensaje(''); }}
                        className="mt-4 text-xs font-medium px-3 py-1.5 rounded-lg transition-colors"
                        style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}
                    >
                        Enviar otro ticket
                    </button>
                </div>
            </section>
        );
    }

    return (
        <section>
            <div className="flex items-center gap-3 mb-1">
                <h2 className="text-lg font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                    Contacto
                </h2>
                {userPlan === 'ultra' && (
                    <span className="tier-badge tier-badge--ultra">Prioritario</span>
                )}
            </div>
            <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                {userPlan === 'ultra'
                    ? 'Soporte prioritario — respuesta en 24 horas.'
                    : 'Envía tu consulta — respuesta en 48 horas.'}
            </p>

            <form onSubmit={handleSubmit} className="glass-panel p-6 flex flex-col gap-4">
                {/* Asunto */}
                <div>
                    <label htmlFor="contact-asunto" className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>
                        Asunto
                    </label>
                    <select
                        id="contact-asunto"
                        value={asunto}
                        onChange={(e) => { setAsunto(e.target.value); setErrors((p) => ({ ...p, asunto: undefined })); }}
                        className="contact-form-input"
                        aria-invalid={!!errors.asunto}
                        aria-describedby={errors.asunto ? 'contact-asunto-error' : undefined}
                        style={errors.asunto ? { borderColor: 'var(--danger)' } : undefined}
                    >
                        <option value="">Selecciona un asunto...</option>
                        {ASUNTO_OPTIONS.map((opt) => (
                            <option key={opt} value={opt}>{opt}</option>
                        ))}
                    </select>
                    {errors.asunto && <p id="contact-asunto-error" role="alert" className="text-xs mt-1" style={{ color: 'var(--danger)' }}>{errors.asunto}</p>}
                </div>

                {/* Mensaje */}
                <div>
                    <label htmlFor="contact-mensaje" className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-muted)' }}>
                        Mensaje
                    </label>
                    <textarea
                        id="contact-mensaje"
                        value={mensaje}
                        onChange={(e) => { setMensaje(e.target.value); setErrors((p) => ({ ...p, mensaje: undefined })); }}
                        rows={5}
                        placeholder="Describe tu consulta o problema..."
                        className="contact-form-input resize-none"
                        aria-invalid={!!errors.mensaje}
                        aria-describedby={errors.mensaje ? 'contact-mensaje-error' : undefined}
                        style={errors.mensaje ? { borderColor: 'var(--danger)' } : undefined}
                    />
                    {errors.mensaje && <p id="contact-mensaje-error" role="alert" className="text-xs mt-1" style={{ color: 'var(--danger)' }}>{errors.mensaje}</p>}
                </div>

                {/* Submit */}
                <button
                    type="submit"
                    disabled={sending}
                    aria-disabled={sending}
                    className="flex items-center justify-center gap-2 text-sm font-semibold py-2.5 px-4 rounded-lg transition-colors self-end"
                    style={{
                        background: sending ? 'rgba(255,255,255,0.1)' : 'var(--accent-primary)',
                        color: 'white',
                        opacity: sending ? 0.7 : 1,
                    }}
                >
                    <Send className="w-4 h-4" />
                    {sending ? 'Enviando...' : 'Enviar ticket'}
                </button>
            </form>
        </section>
    );
}
