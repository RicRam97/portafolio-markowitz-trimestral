'use client';

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Star, X, ChevronUp, AlertTriangle } from 'lucide-react';
import { createBrowserClient } from '@supabase/ssr';
import type { PlanTier } from '@/lib/types';
import { TICKER_LIMITS, PLAN_LABELS } from '@/lib/constants';

// Popular ticker universe for the selector list
const TICKER_UNIVERSE: { ticker: string; name: string }[] = [
    { ticker: 'AAPL', name: 'Apple Inc.' },
    { ticker: 'MSFT', name: 'Microsoft Corp.' },
    { ticker: 'GOOGL', name: 'Alphabet Inc.' },
    { ticker: 'AMZN', name: 'Amazon.com Inc.' },
    { ticker: 'NVDA', name: 'NVIDIA Corp.' },
    { ticker: 'META', name: 'Meta Platforms Inc.' },
    { ticker: 'TSLA', name: 'Tesla Inc.' },
    { ticker: 'BRK-B', name: 'Berkshire Hathaway' },
    { ticker: 'JPM', name: 'JPMorgan Chase & Co.' },
    { ticker: 'V', name: 'Visa Inc.' },
    { ticker: 'JNJ', name: 'Johnson & Johnson' },
    { ticker: 'WMT', name: 'Walmart Inc.' },
    { ticker: 'UNH', name: 'UnitedHealth Group' },
    { ticker: 'MA', name: 'Mastercard Inc.' },
    { ticker: 'PG', name: 'Procter & Gamble' },
    { ticker: 'HD', name: 'The Home Depot' },
    { ticker: 'DIS', name: 'The Walt Disney Co.' },
    { ticker: 'BAC', name: 'Bank of America' },
    { ticker: 'XOM', name: 'Exxon Mobil Corp.' },
    { ticker: 'KO', name: 'Coca-Cola Co.' },
    { ticker: 'PFE', name: 'Pfizer Inc.' },
    { ticker: 'CSCO', name: 'Cisco Systems' },
    { ticker: 'PEP', name: 'PepsiCo Inc.' },
    { ticker: 'NFLX', name: 'Netflix Inc.' },
    { ticker: 'ADBE', name: 'Adobe Inc.' },
    { ticker: 'CRM', name: 'Salesforce Inc.' },
    { ticker: 'AMD', name: 'Advanced Micro Devices' },
    { ticker: 'INTC', name: 'Intel Corp.' },
    { ticker: 'COST', name: 'Costco Wholesale' },
    { ticker: 'NKE', name: 'Nike Inc.' },
    { ticker: 'ABBV', name: 'AbbVie Inc.' },
    { ticker: 'MRK', name: 'Merck & Co.' },
    { ticker: 'T', name: 'AT&T Inc.' },
    { ticker: 'VZ', name: 'Verizon Communications' },
    { ticker: 'QCOM', name: 'Qualcomm Inc.' },
    { ticker: 'ORCL', name: 'Oracle Corp.' },
    { ticker: 'LLY', name: 'Eli Lilly & Co.' },
    { ticker: 'AVGO', name: 'Broadcom Inc.' },
    { ticker: 'TXN', name: 'Texas Instruments' },
    { ticker: 'NOW', name: 'ServiceNow Inc.' },
    { ticker: 'SHOP', name: 'Shopify Inc.' },
    { ticker: 'SQ', name: 'Block Inc.' },
    { ticker: 'PLTR', name: 'Palantir Technologies' },
    { ticker: 'COIN', name: 'Coinbase Global' },
    { ticker: 'SOFI', name: 'SoFi Technologies' },
    { ticker: 'SPY', name: 'SPDR S&P 500 ETF' },
    { ticker: 'QQQ', name: 'Invesco QQQ Trust' },
    { ticker: 'IWM', name: 'iShares Russell 2000' },
    { ticker: 'GLD', name: 'SPDR Gold Shares' },
    { ticker: 'TLT', name: 'iShares 20+ Year Treasury' },
];

interface TickerSelectorModalProps {
    selected: string[];
    onChange: (tickers: string[]) => void;
    userPlan: PlanTier;
}

export default function TickerSelectorModal({ selected, onChange, userPlan }: TickerSelectorModalProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [search, setSearch] = useState('');
    const [onlyFavorites, setOnlyFavorites] = useState(false);
    const [favoriteIds, setFavoriteIds] = useState<Set<string>>(new Set());
    const [loadingFavs, setLoadingFavs] = useState(false);
    const searchRef = useRef<HTMLInputElement>(null);

    // Focus search input when modal opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => searchRef.current?.focus(), 100);
        }
    }, [isOpen]);

    const limit = TICKER_LIMITS[userPlan];
    const atLimit = selected.length >= limit;

    // Fetch user favorites from Supabase
    useEffect(() => {
        const fetchFavorites = async () => {
            setLoadingFavs(true);
            try {
                const supabase = createBrowserClient(
                    process.env.NEXT_PUBLIC_SUPABASE_URL!,
                    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
                );
                const { data: { user } } = await supabase.auth.getUser();
                if (!user) return;

                const { data } = await supabase
                    .from('tickers_favoritos')
                    .select('ticker')
                    .eq('user_id', user.id);

                if (data) {
                    setFavoriteIds(new Set(data.map((f) => f.ticker)));
                }
            } catch {
                // silent fail — favorites just won't show
            } finally {
                setLoadingFavs(false);
            }
        };
        fetchFavorites();
    }, []);

    const toggleFavorite = useCallback(async (ticker: string) => {
        try {
            const supabase = createBrowserClient(
                process.env.NEXT_PUBLIC_SUPABASE_URL!,
                process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
            );
            const { data: { user } } = await supabase.auth.getUser();
            if (!user) return;

            const isFav = favoriteIds.has(ticker);

            if (isFav) {
                await supabase
                    .from('tickers_favoritos')
                    .delete()
                    .eq('user_id', user.id)
                    .eq('ticker', ticker);

                setFavoriteIds((prev) => {
                    const next = new Set(prev);
                    next.delete(ticker);
                    return next;
                });
            } else {
                const tickerData = TICKER_UNIVERSE.find((t) => t.ticker === ticker);
                await supabase
                    .from('tickers_favoritos')
                    .insert({
                        user_id: user.id,
                        ticker,
                        nombre: tickerData?.name ?? null,
                    });

                setFavoriteIds((prev) => new Set(prev).add(ticker));
            }
        } catch {
            // silent fail
        }
    }, [favoriteIds]);

    const toggleTicker = useCallback((ticker: string) => {
        if (selected.includes(ticker)) {
            onChange(selected.filter((t) => t !== ticker));
        } else if (!atLimit) {
            onChange([...selected, ticker]);
        }
    }, [selected, onChange, atLimit]);

    const filteredList = useMemo(() => {
        let list = TICKER_UNIVERSE;

        if (onlyFavorites) {
            list = list.filter((t) => favoriteIds.has(t.ticker));
        }

        if (search.trim()) {
            const q = search.toLowerCase();
            list = list.filter(
                (t) => t.ticker.toLowerCase().includes(q) || t.name.toLowerCase().includes(q),
            );
        }

        return list;
    }, [search, onlyFavorites, favoriteIds]);

    return (
        <>
            {/* Trigger button */}
            <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                        Universo (Tickers)
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded-full font-semibold"
                        style={{ background: 'rgba(37,99,235,0.1)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.2)' }}>
                        {selected.length} / {limit === Infinity ? '\u221E' : limit} seleccionados
                    </span>
                </div>

                <button
                    onClick={() => setIsOpen(true)}
                    className="w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm transition-colors hover:brightness-110"
                    style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                >
                    <span className={selected.length === 0 ? 'opacity-50' : ''}>
                        {selected.length === 0
                            ? 'Seleccionar tickers...'
                            : `${selected.slice(0, 5).join(', ')}${selected.length > 5 ? ` +${selected.length - 5} mas` : ''}`}
                    </span>
                    <ChevronUp className="w-4 h-4 rotate-180" style={{ color: 'var(--text-muted)' }} />
                </button>

                {/* Inline selected chips */}
                {selected.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-2">
                        {selected.map((t) => (
                            <span key={t} className="flex items-center gap-1 text-xs px-2 py-1 rounded-md"
                                style={{ background: 'rgba(37,99,235,0.15)', color: 'var(--accent-primary)', border: '1px solid rgba(37,99,235,0.25)' }}>
                                {t}
                                <button onClick={() => toggleTicker(t)} className="hover:text-white ml-0.5">&times;</button>
                            </span>
                        ))}
                        <button onClick={() => onChange([])} className="text-xs underline ml-1" style={{ color: 'var(--text-muted)' }}>
                            Limpiar
                        </button>
                    </div>
                )}
            </div>

            {/* Modal */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center p-4"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        {/* Backdrop */}
                        {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
                        <div
                            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                            onClick={() => setIsOpen(false)}
                        />

                        {/* Modal content */}
                        <motion.div
                            className="relative w-full max-w-lg rounded-xl flex flex-col"
                            style={{
                                background: 'var(--bg-card, rgba(15,23,42,0.97))',
                                border: '1px solid var(--border-light)',
                                boxShadow: '0 25px 50px rgba(0,0,0,0.5)',
                                maxHeight: '80vh',
                            }}
                            initial={{ scale: 0.9, opacity: 0, y: 20 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.9, opacity: 0, y: 20 }}
                            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between p-5 pb-0">
                                <h3 className="text-lg font-bold" style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}>
                                    Seleccionar Tickers
                                </h3>
                                <button
                                    onClick={() => setIsOpen(false)}
                                    className="p-1 rounded-md hover:bg-white/10 transition-colors"
                                    style={{ color: 'var(--text-muted)' }}
                                    aria-label="Cerrar"
                                >
                                    <X size={18} />
                                </button>
                            </div>

                            {/* Counter + limit warning */}
                            <div className="px-5 pt-3 flex items-center justify-between">
                                <span className="text-xs font-semibold" style={{ color: 'var(--text-muted)' }}>
                                    {selected.length} / {limit === Infinity ? '\u221E' : limit} seleccionados
                                </span>
                                {atLimit && limit !== Infinity && (
                                    <span className="flex items-center gap-1 text-xs px-2 py-1 rounded-md"
                                        style={{ background: 'rgba(245,158,11,0.1)', color: 'var(--warning)', border: '1px solid rgba(245,158,11,0.2)' }}>
                                        <AlertTriangle size={12} />
                                        Limite del plan {PLAN_LABELS[userPlan]} alcanzado
                                    </span>
                                )}
                            </div>

                            {/* Search bar + favorites toggle */}
                            <div className="p-5 pb-3 flex flex-col gap-3">
                                <div className="relative">
                                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                                    <input
                                        ref={searchRef}
                                        type="text"
                                        placeholder="Buscar por empresa o ticker..."
                                        value={search}
                                        onChange={(e) => setSearch(e.target.value)}
                                        className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm outline-none"
                                        style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid var(--border-light)', color: 'var(--text-main)' }}
                                    />
                                </div>

                                <button
                                    onClick={() => setOnlyFavorites((v) => !v)}
                                    className="flex items-center gap-2 text-xs font-semibold px-3 py-2 rounded-lg self-start transition-colors"
                                    style={{
                                        background: onlyFavorites ? 'rgba(245,158,11,0.15)' : 'rgba(255,255,255,0.05)',
                                        color: onlyFavorites ? 'var(--warning)' : 'var(--text-muted)',
                                        border: `1px solid ${onlyFavorites ? 'rgba(245,158,11,0.3)' : 'var(--border-light)'}`,
                                    }}
                                >
                                    <Star size={14} fill={onlyFavorites ? 'currentColor' : 'none'} />
                                    Solo Favoritos
                                </button>
                            </div>

                            {/* Ticker list */}
                            <div className="flex-1 overflow-y-auto px-5 pb-5" style={{ minHeight: 0 }}>
                                {loadingFavs ? (
                                    <div className="text-center py-8 text-xs" style={{ color: 'var(--text-muted)' }}>
                                        Cargando...
                                    </div>
                                ) : filteredList.length === 0 ? (
                                    <div className="text-center py-8 text-xs" style={{ color: 'var(--text-muted)' }}>
                                        {onlyFavorites ? 'No tienes favoritos aun.' : 'Sin resultados para tu busqueda.'}
                                    </div>
                                ) : (
                                    <div className="flex flex-col gap-1">
                                        {filteredList.map((item) => {
                                            const isSelected = selected.includes(item.ticker);
                                            const isFav = favoriteIds.has(item.ticker);
                                            const disabled = !isSelected && atLimit;

                                            return (
                                                <div
                                                    key={item.ticker}
                                                    className="flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors group"
                                                    style={{
                                                        background: isSelected ? 'rgba(37,99,235,0.1)' : 'transparent',
                                                        opacity: disabled ? 0.45 : 1,
                                                    }}
                                                >
                                                    {/* Favorite star */}
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            toggleFavorite(item.ticker);
                                                        }}
                                                        className="flex-shrink-0 p-0.5 transition-colors"
                                                        style={{ color: isFav ? '#f59e0b' : 'var(--text-muted)' }}
                                                        aria-label={isFav ? 'Quitar de favoritos' : 'Agregar a favoritos'}
                                                    >
                                                        <Star size={16} fill={isFav ? 'currentColor' : 'none'} />
                                                    </button>

                                                    {/* Ticker info — clickable area */}
                                                    <button
                                                        onClick={() => !disabled && toggleTicker(item.ticker)}
                                                        disabled={disabled}
                                                        className="flex-1 flex items-center gap-3 text-left min-w-0 disabled:cursor-not-allowed"
                                                    >
                                                        <span className="text-sm font-bold w-16 flex-shrink-0"
                                                            style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)' }}>
                                                            {item.ticker}
                                                        </span>
                                                        <span className="text-xs truncate" style={{ color: 'var(--text-muted)' }}>
                                                            {item.name}
                                                        </span>
                                                    </button>

                                                    {/* Checkbox */}
                                                    <button
                                                        type="button"
                                                        className="w-5 h-5 rounded flex-shrink-0 flex items-center justify-center transition-colors"
                                                        style={{
                                                            background: isSelected ? 'var(--accent-primary)' : 'rgba(255,255,255,0.06)',
                                                            border: `1.5px solid ${isSelected ? 'var(--accent-primary)' : 'var(--border-light)'}`,
                                                        }}
                                                        disabled={disabled}
                                                        onClick={() => toggleTicker(item.ticker)}
                                                        aria-label={isSelected ? `Quitar ${item.ticker}` : `Agregar ${item.ticker}`}
                                                    >
                                                        {isSelected && (
                                                            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                                                                <path d="M2.5 6L5 8.5L9.5 3.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                                                            </svg>
                                                        )}
                                                    </button>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="p-5 pt-3" style={{ borderTop: '1px solid var(--border-light)' }}>
                                <motion.button
                                    onClick={() => setIsOpen(false)}
                                    className="btn btn-cta glow-effect w-full py-3 text-sm font-semibold"
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                >
                                    Confirmar ({selected.length} tickers)
                                </motion.button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
