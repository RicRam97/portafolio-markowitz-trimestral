'use client';

import { useState } from 'react';
import faqData from '@/data/faq-soporte.json';

const categories = faqData.categories;

export default function FaqAccordion() {
    const [activeCategory, setActiveCategory] = useState<string | null>(null);

    const filtered = activeCategory
        ? categories.filter((c) => c.id === activeCategory)
        : categories;

    return (
        <section>
            <h2 className="text-lg font-bold mb-1" style={{ fontFamily: 'var(--font-display)' }}>
                Preguntas Frecuentes
            </h2>
            <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                Encuentra respuestas rápidas a las dudas más comunes.
            </p>

            {/* Category Tabs */}
            <div className="faq-category-tabs">
                <button
                    className={`faq-tab ${activeCategory === null ? 'faq-tab--active' : ''}`}
                    onClick={() => setActiveCategory(null)}
                >
                    Todas
                </button>
                {categories.map((cat) => (
                    <button
                        key={cat.id}
                        className={`faq-tab ${activeCategory === cat.id ? 'faq-tab--active' : ''}`}
                        onClick={() => setActiveCategory(cat.id)}
                    >
                        {cat.label}
                    </button>
                ))}
            </div>

            {/* FAQ Items */}
            {filtered.map((cat) => (
                <div key={cat.id} className="mb-6">
                    {activeCategory === null && (
                        <h3 className="text-sm font-semibold mb-3 uppercase tracking-wider" style={{ color: 'var(--accent-primary)' }}>
                            {cat.label}
                        </h3>
                    )}
                    <div className="faq-container">
                        {cat.questions.map((item, idx) => (
                            <details key={idx} className="faq-item glass-panel">
                                <summary className="faq-question">{item.q}</summary>
                                <div className="faq-answer">
                                    <p>{item.a}</p>
                                </div>
                            </details>
                        ))}
                    </div>
                </div>
            ))}
        </section>
    );
}
