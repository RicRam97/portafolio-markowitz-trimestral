export interface GlossaryTerm {
    term: string;
    definition: string;
}

export const glossaryTerms: GlossaryTerm[] = [
    {
        term: 'Ratio de Sharpe',
        definition: 'Mide el rendimiento extra que obtienes por cada unidad de riesgo asumido. Un valor alto indica que el portafolio es eficiente.',
    },
    {
        term: 'Volatilidad',
        definition: 'Indica qué tanto fluctúa el precio de un activo. Mayor volatilidad significa mayor incertidumbre sobre los rendimientos futuros.',
    },
    {
        term: 'Diversificación',
        definition: 'Estrategia de invertir en diferentes activos para reducir el riesgo total del portafolio. "No poner todos los huevos en la misma canasta."',
    },
    {
        term: 'Frontera Eficiente',
        definition: 'Conjunto de portafolios que ofrecen el máximo rendimiento esperado para cada nivel de riesgo. Cualquier portafolio debajo de ella es subóptimo.',
    },
    {
        term: 'Rebalanceo',
        definition: 'Ajustar periódicamente los pesos de los activos en tu portafolio para mantener la distribución objetivo original.',
    },
    {
        term: 'Rendimiento Esperado',
        definition: 'Ganancia promedio que se espera obtener de una inversión, calculada a partir de datos históricos. No garantiza resultados futuros.',
    },
    {
        term: 'Covarianza',
        definition: 'Mide cómo se mueven dos activos juntos. Si es positiva, tienden a subir y bajar al mismo tiempo; si es negativa, se mueven en direcciones opuestas.',
    },
    {
        term: 'ETF',
        definition: 'Fondo cotizado en bolsa que agrupa múltiples activos y se compra/vende como una acción. Permite diversificar con una sola operación.',
    },
    {
        term: 'Benchmark',
        definition: 'Índice de referencia (como el S&P 500) contra el cual se compara el desempeño de un portafolio para evaluar si generó valor adicional.',
    },
    {
        term: 'Drawdown',
        definition: 'Caída máxima desde un punto alto hasta el punto más bajo antes de recuperarse. Mide el peor escenario histórico de pérdida de un portafolio.',
    },
];
