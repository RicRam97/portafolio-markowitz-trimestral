'use client';

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { createBrowserClient } from '@supabase/ssr';
import { useSessionExpiry } from '@/hooks/useSessionExpiry';

const navItems = [
    { icon: '📊', label: 'Dashboard', href: '/dashboard' },
    { icon: '🎯', label: 'Estrategias', href: '/dashboard/estrategias' },
    { icon: '👤', label: 'Perfil', href: '/dashboard/perfil' },
    { icon: '⚙️', label: 'Ajustes', href: '/dashboard/ajustes' },
    { icon: '📚', label: 'Aprende', href: '/dashboard/aprende', desktopOnly: true },
];

interface Props {
    nombre: string;
    email: string;
    children: React.ReactNode;
}

export default function DashboardShell({ nombre, email, children }: Props) {
    const pathname = usePathname();
    const router = useRouter();
    useSessionExpiry();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
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
        <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-dark)' }}>
            {/* ===== SIDEBAR (desktop) ===== */}
            <aside
                className="hidden md:flex flex-col w-[220px] flex-shrink-0"
                style={{ background: 'rgba(11,17,32,0.98)', borderRight: '1px solid var(--border-light)' }}
            >
                {/* Logo */}
                <div className="p-5 border-b" style={{ borderColor: 'var(--border-light)' }}>
                    <Link href="/">
                        <Image src="/kaudal-logo2.png" alt="Kaudal" width={120} height={32} className="h-8 w-auto object-contain rounded" />
                    </Link>
                </div>

                {/* Nav items */}
                <nav className="flex-1 p-3 flex flex-col gap-1">
                    {navItems.map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${pathname === item.href
                                ? 'text-[var(--accent-primary)] bg-blue-600/10 border border-blue-600/20'
                                : 'text-[var(--text-muted)] hover:text-[var(--text-main)] hover:bg-white/5'
                                }`}
                        >
                            <span className="text-base">{item.icon}</span>
                            <span>{item.label}</span>
                        </Link>
                    ))}
                </nav>

                {/* User footer */}
                <div className="p-4 border-t" style={{ borderColor: 'var(--border-light)' }}>
                    <div className="flex items-center gap-3 mb-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-600 to-emerald-500 flex items-center justify-center text-sm font-bold">
                            {nombre.charAt(0).toUpperCase()}
                        </div>
                        <div className="overflow-hidden">
                            <p className="text-sm font-semibold truncate">{nombre}</p>
                            <p className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>{email}</p>
                        </div>
                    </div>
                    <button
                        onClick={handleLogout}
                        disabled={signingOut}
                        className="w-full text-sm py-2 px-3 rounded-lg transition-all btn btn-secondary"
                    >
                        {signingOut ? 'Cerrando...' : 'Cerrar sesión'}
                    </button>
                </div>
            </aside>

            {/* ===== MAIN AREA ===== */}
            <div className="flex-1 flex flex-col overflow-hidden">
                {/* Header */}
                <header
                    className="flex items-center justify-between px-5 py-3 flex-shrink-0"
                    style={{ background: 'rgba(11,17,32,0.90)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border-light)' }}
                >
                    <div className="flex items-center gap-3">
                        {/* Mobile logo (hidden on desktop) */}
                        <Link href="/" className="md:hidden">
                            <Image src="/kaudal-logo2.png" alt="Kaudal" width={90} height={24} className="h-6 w-auto object-contain rounded" />
                        </Link>
                        <h1 className="text-sm font-semibold hidden md:block">Hola, <span style={{ color: 'var(--accent-primary)' }}>{nombre}</span> 👋</h1>
                    </div>
                    <div className="flex items-center gap-3">
                        <span className="text-xs px-2.5 py-1 rounded-full font-semibold" style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.25)' }}>
                            Básico
                        </span>
                        <button
                            onClick={handleLogout}
                            disabled={signingOut}
                            className="hidden md:block text-xs btn btn-secondary py-1.5 px-3"
                        >
                            {signingOut ? '...' : 'Cerrar sesión'}
                        </button>
                        {/* Mobile hamburger */}
                        <button
                            className="md:hidden p-1.5 rounded-lg text-[var(--text-muted)] hover:bg-white/5 bg-transparent border-none cursor-pointer"
                            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        >
                            ☰
                        </button>
                    </div>
                </header>

                {/* Mobile bottom nav */}
                <nav
                    className="md:hidden fixed bottom-0 left-0 right-0 z-50 flex justify-around py-2 px-4"
                    style={{ background: 'rgba(11,17,32,0.98)', borderTop: '1px solid var(--border-light)' }}
                >
                    {navItems.filter(i => !i.desktopOnly).slice(0, 4).map((item) => (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex flex-col items-center gap-0.5 text-xs font-medium transition-all ${pathname === item.href ? 'text-[var(--accent-primary)]' : 'text-[var(--text-muted)]'
                                }`}
                        >
                            <span className="text-xl leading-none">{item.icon}</span>
                            <span>{item.label}</span>
                        </Link>
                    ))}
                </nav>

                {/* Scrollable content */}
                <main className="flex-1 overflow-y-auto p-5 pb-24 md:pb-5">
                    {children}
                </main>
            </div>

            {/* Mobile menu overlay */}
            {mobileMenuOpen && (
                <div
                    role="button"
                    tabIndex={0}
                    aria-label="Cerrar menú"
                    className="fixed inset-0 bg-black/50 z-40 md:hidden"
                    onClick={() => setMobileMenuOpen(false)}
                    onKeyDown={(e) => { if (e.key === 'Escape' || e.key === 'Enter') setMobileMenuOpen(false); }}
                >
                    <div
                        role="presentation"
                        className="absolute top-0 right-0 w-[240px] h-full p-4 flex flex-col gap-2"
                        style={{ background: 'rgba(11,17,32,0.99)', borderLeft: '1px solid var(--border-light)' }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <h3 className="text-sm font-bold mb-2 px-2">Menú</h3>
                        {navItems.map((item) => (
                            <Link
                                key={item.href}
                                href={item.href}
                                onClick={() => setMobileMenuOpen(false)}
                                className="flex items-center gap-3 px-3 py-3 rounded-xl text-sm font-medium text-[var(--text-muted)] hover:text-white hover:bg-white/5 transition-all"
                            >
                                <span>{item.icon}</span> {item.label}
                            </Link>
                        ))}
                        <button
                            onClick={handleLogout}
                            className="mt-auto btn btn-secondary w-full text-sm"
                            disabled={signingOut}
                        >
                            {signingOut ? 'Cerrando...' : 'Cerrar sesión'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
