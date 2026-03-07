'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import {
    LayoutDashboard,
    TrendingUp,
    Crosshair,
    UserCircle,
    CreditCard,
    Settings,
    FileText,
    LogOut,
    ArrowRight
} from 'lucide-react';
import ThemeToggle from '@/components/ThemeToggle';

const navItems = [
    { icon: LayoutDashboard, label: 'Inicio', href: '/dashboard' },
    { icon: Crosshair, label: 'Optimizar', href: '/dashboard/optimizar' },
    { icon: TrendingUp, label: 'Estrategias', href: '/dashboard/estrategias' },
    { icon: UserCircle, label: 'Mi Perfil', href: '/dashboard/perfil' },
    { icon: CreditCard, label: 'Mi Cuenta', href: '/dashboard/cuenta' },
    { icon: Settings, label: 'Configuracion', href: '/dashboard/config' },
    { icon: FileText, label: 'Tests', href: '/dashboard/tests' },
];

interface Props {
    nombre: string;
    email: string;
    testCompletado: boolean;
    children: React.ReactNode;
}

export default function DashboardUI({ nombre, email, testCompletado, children }: Props) {
    const pathname = usePathname();
    const router = useRouter();
    const [signingOut, setSigningOut] = useState(false);

    const supabase = createBrowserClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );

    const handleLogout = async () => {
        setSigningOut(true);
        await supabase.auth.signOut();
        router.push('/login');
    };

    return (
        <div className="flex h-screen overflow-hidden bg-[var(--bg-dark)]">
            {/* DESKTOP SIDEBAR */}
            <aside className="max-lg:hidden flex flex-col w-[240px] flex-shrink-0 bg-[var(--bg-sidebar)] border-r border-[var(--border-medium)]">
                {/* Logo */}
                <div className="p-5 border-b border-[var(--border-medium)]">
                    <div className="flex items-center gap-2">
                        <Link href="/">
                            <Image src="/kaudal-logo2.png" alt="Kaudal" width={120} height={32} className="h-8 w-auto object-contain rounded" />
                        </Link>
                        <span className="text-[10px] font-bold tracking-wide px-1.5 py-0.5 rounded-md bg-gradient-to-br from-blue-500 to-emerald-500 text-white leading-tight shrink-0">BETA</span>
                    </div>
                </div>

                {/* Navigation Links */}
                <nav className="flex-1 overflow-y-auto py-4 px-3 flex flex-col gap-1" aria-label="Menú principal">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = item.href === '/dashboard' ? pathname === '/dashboard' : pathname === item.href || pathname.startsWith(`${item.href}/`);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                aria-current={isActive ? 'page' : undefined}
                                className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors ${isActive
                                        ? 'bg-[var(--accent-primary)] text-white'
                                        : 'text-[var(--text-muted)] hover:bg-[var(--overlay-hover)] hover:text-[var(--text-main)]'
                                    }`}
                            >
                                <Icon className="w-5 h-5" />
                                <span>{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                {/* Sidebar Footer */}
                <div className="p-4 border-t border-[var(--border-medium)]">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-600 to-emerald-500 flex items-center justify-center text-sm font-bold text-white shadow-lg">
                            {nombre.charAt(0).toUpperCase()}
                        </div>
                        <div className="overflow-hidden flex-1">
                            <p className="text-sm font-semibold truncate text-[var(--text-main)]">{nombre}</p>
                            <p className="text-xs truncate text-[var(--text-muted)]">{email}</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleLogout}
                            disabled={signingOut}
                            className="flex-1 flex items-center justify-center gap-2 text-sm py-2 px-3 rounded-lg transition-colors bg-[var(--overlay-soft)] text-[var(--text-muted)] hover:bg-[var(--overlay-hover)] hover:text-[var(--text-main)]"
                        >
                            <LogOut className="w-4 h-4" />
                            {signingOut ? 'Cerrando...' : 'Cerrar sesion'}
                        </button>
                        <ThemeToggle />
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT AREA */}
            <div className="flex-1 flex flex-col min-w-0 h-screen">
                {/* Mobile Header (Hidden on LG) */}
                <header className="lg:hidden flex items-center justify-between p-4 bg-[var(--bg-sidebar)] border-b border-[var(--border-medium)]">
                    <Link href="/" className="flex items-center gap-2">
                        <Image src="/kaudal-logo2.png" alt="Kaudal" width={100} height={28} className="h-7 w-auto object-contain rounded" />
                        <span className="text-[10px] font-bold tracking-wide px-1.5 py-0.5 rounded bg-gradient-to-br from-blue-500 to-emerald-500 text-white leading-tight">BETA</span>
                    </Link>
                    <div className="flex items-center gap-2">
                        <ThemeToggle />
                        <button onClick={handleLogout} className="text-[var(--text-muted)] hover:text-[var(--text-main)]" disabled={signingOut}>
                            <LogOut className="w-5 h-5" />
                        </button>
                    </div>
                </header>

                {/* Persistent Banner (if tests not completed) */}
                {!testCompletado && (
                    <div className="bg-[var(--accent-primary)] text-white px-4 py-3 flex items-center justify-between shrink-0">
                        <span className="text-sm font-medium">Completa tu perfil de inversionista para empezar</span>
                        <Link href="/dashboard/tests" className="flex items-center gap-1 text-sm font-semibold hover:opacity-80 transition-opacity">
                            Ir ahora <ArrowRight className="w-4 h-4" />
                        </Link>
                    </div>
                )}

                {/* Scrollable Content */}
                <main className="flex-1 overflow-y-auto">
                    <div className="min-h-full flex flex-col">
                        {/* Page Content */}
                        <div className="p-4 md:p-6 lg:p-8 flex-1">
                            {children}
                        </div>

                        {/* Persistent Footer */}
                        <footer className="mt-8 py-6 px-4 border-t border-[var(--border-medium)] text-center shrink-0 w-full max-lg:mb-16 bg-[var(--bg-sidebar)]">
                            <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs md:text-sm text-[var(--text-muted)]">
                                <a href="/faq" target="_blank" rel="noopener noreferrer" className="hover:text-[var(--text-main)] transition-colors">Soporte</a>
                                <span className="max-sm:hidden inline">•</span>
                                <Link href="/privacidad" className="hover:text-[var(--text-main)] transition-colors">Aviso de Privacidad</Link>
                                <span className="max-sm:hidden inline">•</span>
                                <Link href="/terminos" className="hover:text-[var(--text-main)] transition-colors">Terminos y Condiciones</Link>
                            </div>
                            <p className="mt-4 text-xs text-[var(--text-muted)]">&copy; {new Date().getFullYear()} Kaudal. Todos los derechos reservados.</p>
                        </footer>
                    </div>
                </main>

                {/* MOBILE BOTTOM NAVIGATION (Hidden on LG) */}
                <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-50 flex justify-between px-2 py-2 bg-[var(--bg-sidebar)] border-t border-[var(--border-medium)] backdrop-blur-md pb-safe" aria-label="Menú principal">
                    {navItems.slice(0, 5).map((item) => {
                        const Icon = item.icon;
                        const isActive = item.href === '/dashboard' ? pathname === '/dashboard' : pathname === item.href || pathname.startsWith(`${item.href}/`);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                aria-current={isActive ? 'page' : undefined}
                                className={`flex flex-col items-center justify-center w-full py-1 text-[10px] font-medium transition-colors ${isActive ? 'text-[var(--accent-primary)]' : 'text-[var(--text-muted)]'
                                    }`}
                            >
                                <Icon className={`w-5 h-5 mb-1 ${isActive ? 'fill-[var(--accent-primary)]/20' : ''}`} />
                                <span className="truncate w-full text-center px-1">{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>
            </div>
        </div>
    );
}
