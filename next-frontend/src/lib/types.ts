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

export type OptimizerModel = 'markowitz' | 'hrp' | 'montecarlo';

export interface PortfolioOptimization {
    weights: Record<string, number>;
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
}

export interface FrontierPoint {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
}

export interface MonteCarloCloudPoint {
    ret: number;
    vol: number;
    sharpe: number;
}

export interface MonteCarloSimulation {
    num_portfolios: number;
    cloud: MonteCarloCloudPoint[];
}

export interface AsignacionTickerReal {
    peso_teorico: number;
    peso_real: number;
    acciones: number;
    inversion: number;
    comision: number;
    precio_compra: number;
}

export interface AsignacionReal {
    asignacion: Record<string, AsignacionTickerReal>;
    efectivo_restante: number;
    inversion_total: number;
    comisiones_totales: number;
    porcentaje_invertido: number;
    desviacion_maxima_peso: number;
    warning?: string;
}

export interface OptimizationResult {
    portafolio_optimo: PortfolioOptimization;
    portafolio_min_vol?: PortfolioOptimization;
    frontera_eficiente?: FrontierPoint[];
    simulacion?: MonteCarloSimulation;
    clustering_data?: number[][] | null;
    tickers_incluidos: string[];
    fecha_calculo: string;
    asignacion_real?: AsignacionReal | null;
    advertencias?: string[];
    compatible_con_perfil?: boolean;
}
