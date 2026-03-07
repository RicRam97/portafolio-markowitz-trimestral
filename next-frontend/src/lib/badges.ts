export interface BadgeDefinition {
    name: string;
    icon: string;
    description: string;
}

export const BADGES: Record<string, BadgeDefinition> = {
    first_strategy: {
        name: 'Primera Estrategia',
        icon: '\uD83C\uDFAF',
        description: 'Creaste tu primera estrategia de inversi\u00f3n',
    },
    sharpe_master: {
        name: 'Maestro de Eficiencia',
        icon: '\uD83C\uDFC6',
        description: 'Lograste un Sharpe Ratio mayor a 1.5',
    },
    diversifier: {
        name: 'Diversificador',
        icon: '\uD83C\uDF10',
        description: 'Optimizaste un portafolio con 5 o m\u00e1s activos',
    },
    profile_complete: {
        name: 'Perfil Completo',
        icon: '\u2705',
        description: 'Completaste ambos tests de perfil de inversionista',
    },
    five_strategies: {
        name: 'Estratega Veterano',
        icon: '\u2B50',
        description: 'Guardaste 5 estrategias de inversi\u00f3n',
    },
};

export type BadgeId = keyof typeof BADGES;
