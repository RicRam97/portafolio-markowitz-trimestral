'use client';

import { useState, useMemo, useEffect, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import * as Accordion from '@radix-ui/react-accordion';
import { HelpCircle, X, Search, ChevronDown, MessageCircle } from 'lucide-react';
import Link from 'next/link';
import articles from '@/data/help-articles.json';

interface Article {
  id: string;
  categoria: string;
  pregunta: string;
  respuesta: string;
}

export default function HelpWidget() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [pulse, setPulse] = useState(false);

  // Pulse animation every 10 seconds
  useEffect(() => {
    if (open) return;
    const interval = setInterval(() => {
      setPulse(true);
      setTimeout(() => setPulse(false), 1500);
    }, 10000);
    return () => clearInterval(interval);
  }, [open]);

  const handleToggle = useCallback(() => {
    setOpen((prev) => !prev);
    setQuery('');
  }, []);

  // Filter and group articles by category
  const grouped = useMemo(() => {
    const q = query.toLowerCase().trim();
    const filtered = q
      ? (articles as Article[]).filter(
          (a) =>
            a.pregunta.toLowerCase().includes(q) ||
            a.respuesta.toLowerCase().includes(q) ||
            a.categoria.toLowerCase().includes(q)
        )
      : (articles as Article[]);

    const map = new Map<string, Article[]>();
    for (const a of filtered) {
      const list = map.get(a.categoria) ?? [];
      list.push(a);
      map.set(a.categoria, list);
    }
    return map;
  }, [query]);

  return (
    <>
      {/* Floating button */}
      <button
        onClick={handleToggle}
        aria-label={open ? 'Cerrar ayuda' : 'Abrir ayuda'}
        className="fixed bottom-6 right-6 z-50 flex items-center justify-center w-14 h-14 rounded-full shadow-lg transition-transform hover:scale-105 active:scale-95"
        style={{
          background: 'var(--accent-primary)',
          color: 'white',
          animation: pulse ? 'help-pulse 1.5s ease-in-out' : undefined,
        }}
      >
        {open ? <X className="w-6 h-6" /> : <HelpCircle className="w-6 h-6" />}
      </button>

      {/* Drawer */}
      <AnimatePresence>
        {open && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={handleToggle}
              className="fixed inset-0 z-40 bg-black/40 backdrop-blur-sm"
              aria-hidden
            />

            {/* Panel */}
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 28, stiffness: 300 }}
              className="fixed top-0 right-0 z-50 h-full w-full max-w-md flex flex-col shadow-2xl"
              style={{ background: 'var(--bg-sidebar)', borderLeft: '1px solid var(--border-light)' }}
              role="dialog"
              aria-label="Centro de ayuda"
            >
              {/* Header */}
              <div
                className="flex items-center justify-between px-6 py-4 shrink-0"
                style={{ borderBottom: '1px solid var(--border-light)' }}
              >
                <h2
                  className="text-lg font-bold"
                  style={{ fontFamily: 'var(--font-display)', color: 'var(--text-main)' }}
                >
                  Centro de ayuda
                </h2>
                <button
                  onClick={handleToggle}
                  aria-label="Cerrar ayuda"
                  className="p-1.5 rounded-lg transition-colors"
                  style={{ color: 'var(--text-muted)' }}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Search */}
              <div className="px-6 py-3 shrink-0">
                <div
                  className="flex items-center gap-2 px-3 py-2 rounded-lg"
                  style={{
                    background: 'var(--overlay-soft)',
                    border: '1px solid var(--border-light)',
                  }}
                >
                  <Search className="w-4 h-4 shrink-0" style={{ color: 'var(--text-muted)' }} />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Buscar en preguntas frecuentes..."
                    className="flex-1 bg-transparent text-sm outline-none placeholder:opacity-60"
                    style={{ color: 'var(--text-main)' }}
                    autoFocus
                  />
                  {query && (
                    <button
                      onClick={() => setQuery('')}
                      className="p-0.5 rounded"
                      style={{ color: 'var(--text-muted)' }}
                      aria-label="Limpiar búsqueda"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>

              {/* FAQ content */}
              <div className="flex-1 overflow-y-auto px-6 pb-4">
                {grouped.size === 0 ? (
                  <div className="text-center py-10">
                    <Search className="w-10 h-10 mx-auto mb-3 opacity-30" style={{ color: 'var(--text-muted)' }} />
                    <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                      No se encontraron resultados para &ldquo;{query}&rdquo;
                    </p>
                  </div>
                ) : (
                  <Accordion.Root type="multiple" className="flex flex-col gap-3">
                    {Array.from(grouped.entries()).map(([categoria, items]) => (
                      <div key={categoria}>
                        <p
                          className="text-xs font-semibold uppercase tracking-wider mb-2 mt-3"
                          style={{ color: 'var(--accent-secondary)' }}
                        >
                          {categoria}
                        </p>
                        {items.map((article) => (
                          <Accordion.Item key={article.id} value={article.id} className="mb-1">
                            <Accordion.Header>
                              <Accordion.Trigger
                                className="group flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-colors"
                                style={{ color: 'var(--text-main)', background: 'var(--overlay-soft)' }}
                              >
                                {article.pregunta}
                                <ChevronDown
                                  className="w-4 h-4 shrink-0 transition-transform duration-200 group-data-[state=open]:rotate-180"
                                  style={{ color: 'var(--text-muted)' }}
                                />
                              </Accordion.Trigger>
                            </Accordion.Header>
                            <Accordion.Content className="overflow-hidden data-[state=open]:animate-accordion-down data-[state=closed]:animate-accordion-up">
                              <p
                                className="px-3 py-2.5 text-sm leading-relaxed"
                                style={{ color: 'var(--text-muted)' }}
                              >
                                {article.respuesta}
                              </p>
                            </Accordion.Content>
                          </Accordion.Item>
                        ))}
                      </div>
                    ))}
                  </Accordion.Root>
                )}
              </div>

              {/* Footer CTA */}
              <div
                className="px-6 py-4 shrink-0"
                style={{ borderTop: '1px solid var(--border-light)' }}
              >
                <Link
                  href="/dashboard/config?tab=soporte"
                  onClick={handleToggle}
                  className="flex items-center justify-center gap-2 w-full text-sm font-semibold py-2.5 rounded-lg transition-colors"
                  style={{ background: 'var(--overlay-soft)', color: 'var(--text-main)' }}
                >
                  <MessageCircle className="w-4 h-4" />
                  No encuentras lo que buscas? Contacta soporte
                </Link>
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Pulse keyframes */}
      <style jsx global>{`
        @keyframes help-pulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.5); }
          50% { box-shadow: 0 0 0 12px rgba(37, 99, 235, 0); }
        }
        @keyframes accordion-down {
          from { height: 0; }
          to { height: var(--radix-accordion-content-height); }
        }
        @keyframes accordion-up {
          from { height: var(--radix-accordion-content-height); }
          to { height: 0; }
        }
        .animate-accordion-down {
          animation: accordion-down 200ms ease-out;
        }
        .animate-accordion-up {
          animation: accordion-up 200ms ease-out;
        }
      `}</style>
    </>
  );
}
