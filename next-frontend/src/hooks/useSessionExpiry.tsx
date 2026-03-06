"use client";

import { useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import { toast } from 'react-hot-toast';

export const useSessionExpiry = () => {
    const router = useRouter();
    const warningTimeoutRef = useRef<number | null>(null);
    const expiryTimeoutRef = useRef<number | null>(null);
    const warningToastIdRef = useRef<string | null>(null);
    const activityListenerAdded = useRef(false);

    const supabase = createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    useEffect(() => {
        let isAuth = false;

        const checkAuth = async () => {
            const { data: { session } } = await supabase.auth.getSession();
            isAuth = !!session;
            if (isAuth) resetTimer();
        };

        checkAuth();

        const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
            isAuth = !!session;
            if (event === 'SIGNED_OUT') {
                clearTimer();
            } else if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
                resetTimer();
                if (warningToastIdRef.current) {
                    toast.dismiss(warningToastIdRef.current);
                    warningToastIdRef.current = null;
                }
            }
        });

        let lastActivity = Date.now();
        const handleActivity = () => {
            if (isAuth) {
                const now = Date.now();
                if (now - lastActivity > 1000) { // Throttle to 1s
                    lastActivity = now;
                    resetTimer();
                }
            }
        };

        const clearTimer = () => {
            if (warningTimeoutRef.current !== null) {
                window.clearTimeout(warningTimeoutRef.current);
                warningTimeoutRef.current = null;
            }
            if (expiryTimeoutRef.current !== null) {
                window.clearTimeout(expiryTimeoutRef.current);
                expiryTimeoutRef.current = null;
            }
        };

        const resetTimer = () => {
            clearTimer();
            if (warningToastIdRef.current) {
                toast.dismiss(warningToastIdRef.current);
                warningToastIdRef.current = null;
            }

            // 55 minutes warning
            warningTimeoutRef.current = window.setTimeout(() => {
                warningToastIdRef.current = toast(
                    (t) => (
                        <div className="flex flex-col gap-2 p-1" style={{ color: 'var(--text-main)', background: 'transparent' }}>
                            <span className="text-sm font-medium">Tu sesión expirará en 5 minutos por inactividad</span>
                            <button
                                onClick={async () => {
                                    toast.dismiss(t.id);
                                    await supabase.auth.refreshSession();
                                    resetTimer();
                                }}
                                className="btn btn-cta text-sm py-2 px-3 transition"
                                style={{ width: '100%' }}
                            >
                                Mantener sesión activa
                            </button>
                        </div>
                    ),
                    { duration: 5 * 60 * 1000, position: 'bottom-center', style: { background: 'rgba(11,17,32,0.95)', border: '1px solid var(--border-light)', backdropFilter: 'blur(12px)' } }
                );

                // 60 minutes full expiry
                expiryTimeoutRef.current = window.setTimeout(async () => {
                    if (warningToastIdRef.current) toast.dismiss(warningToastIdRef.current);
                    await supabase.auth.signOut();
                    router.push('/login?reason=inactividad');
                }, 5 * 60 * 1000);
            }, 55 * 60 * 1000);
        };

        const events = ['mousemove', 'keydown', 'click', 'scroll'];
        if (!activityListenerAdded.current) {
            events.forEach(event => window.addEventListener(event, handleActivity));
            activityListenerAdded.current = true;
        }

        return () => {
            events.forEach(event => window.removeEventListener(event, handleActivity));
            activityListenerAdded.current = false;
            clearTimer();
            subscription.unsubscribe();
            if (warningToastIdRef.current) {
                toast.dismiss(warningToastIdRef.current);
            }
        };
    }, [router, supabase]);
};
