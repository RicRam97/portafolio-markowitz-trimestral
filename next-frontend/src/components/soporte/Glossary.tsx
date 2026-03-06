import { BookOpen } from 'lucide-react';
import { glossaryTerms } from '@/data/glossary';

export default function Glossary() {
    return (
        <section>
            <div className="flex items-center gap-2 mb-1">
                <BookOpen className="w-5 h-5" style={{ color: 'var(--accent-secondary)' }} />
                <h2 className="text-lg font-bold" style={{ fontFamily: 'var(--font-display)' }}>
                    Glosario Financiero
                </h2>
            </div>
            <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>
                Términos clave explicados en lenguaje simple.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {glossaryTerms.map((item) => (
                    <div key={item.term} className="glass-panel glossary-card">
                        <h3 className="text-sm font-bold mb-2" style={{ color: 'var(--text-main)' }}>
                            {item.term}
                        </h3>
                        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                            {item.definition}
                        </p>
                    </div>
                ))}
            </div>
        </section>
    );
}
