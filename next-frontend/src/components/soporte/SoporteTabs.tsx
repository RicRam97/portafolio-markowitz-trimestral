'use client';

import { useState } from 'react';
import { HelpCircle, BookOpen } from 'lucide-react';
import FaqAccordion from './FaqAccordion';
import Glossary from './Glossary';

type Tab = 'faq' | 'glossary';

export default function SoporteTabs() {
    const [activeTab, setActiveTab] = useState<Tab>('faq');

    return (
        <>
            {/* Tab Buttons */}
            <div className="flex gap-3 mb-8">
                <button
                    onClick={() => setActiveTab('faq')}
                    className="soporte-tab"
                    data-active={activeTab === 'faq'}
                >
                    <HelpCircle className="w-4 h-4" />
                    Preguntas frecuentes
                </button>
                <button
                    onClick={() => setActiveTab('glossary')}
                    className="soporte-tab"
                    data-active={activeTab === 'glossary'}
                >
                    <BookOpen className="w-4 h-4" />
                    Glosario
                </button>
            </div>

            {/* Tab Content */}
            <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
                {activeTab === 'faq' ? <FaqAccordion /> : <Glossary />}
            </div>
        </>
    );
}
