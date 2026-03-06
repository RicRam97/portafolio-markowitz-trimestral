export interface AllocationItem {
    ticker: string;
    weight_pct: number;
    shares: number;
}

export interface Portafolio {
    id: string;
    user_id: string;
    estrategia_id: string | null;
    nombre: string;
    presupuesto: number;
    rendimiento_pct: number | null;
    volatilidad_pct: number | null;
    sharpe_ratio: number | null;
    allocation: AllocationItem[];
    metricas: Record<string, unknown>;
    created_at: string;
}

export interface Estrategia {
    id: string;
    user_id: string;
    nombre: string;
    tipo: 'markowitz' | 'hrp';
    parametros: Record<string, unknown>;
    created_at: string;
    updated_at: string;
}

export interface TickerFavorito {
    id: string;
    user_id: string;
    ticker: string;
    nombre: string | null;
    added_at: string;
}

export interface TickerQuote {
    price: number;
    change_pct: number;
    name: string;
    error?: boolean;
}

export interface RebalanceAlert {
    portfolio_id: string;
    portfolio_nombre: string;
    ticker: string;
    target_weight: number;
    current_weight: number;
    deviation: number;
}

export interface SupportTicket {
    id: string;
    user_id: string;
    asunto: string;
    mensaje: string;
    tier: 'pro' | 'ultra';
    status: 'abierto' | 'en_progreso' | 'resuelto';
    created_at: string;
}

export type PlanTier = 'basico' | 'pro' | 'ultra';
