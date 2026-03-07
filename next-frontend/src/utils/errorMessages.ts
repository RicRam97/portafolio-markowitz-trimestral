export interface ErrorMessage {
    title: string;
    message: string;
    action?: string;
}

export const ERROR_MESSAGES: Record<string, ErrorMessage> = {
    // ── Auth ──────────────────────────────────────────────
    AUTH_INVALID_CREDENTIALS: {
        title: 'Credenciales incorrectas',
        message: 'Correo o contrasena incorrectos.',
        action: 'Reintentar',
    },
    AUTH_EMAIL_NOT_CONFIRMED: {
        title: 'Correo no confirmado',
        message: 'Debes confirmar tu correo antes de entrar.',
        action: 'Reenviar correo',
    },
    AUTH_USER_EXISTS: {
        title: 'Correo ya registrado',
        message: 'Ya existe una cuenta con ese correo.',
        action: 'Ir a Iniciar Sesion',
    },
    AUTH_SESSION_EXPIRED: {
        title: 'Sesion expirada',
        message: 'Tu sesion expiro. Por favor, inicia sesion nuevamente.',
        action: 'Iniciar sesion',
    },
    AUTH_REQUIRED: {
        title: 'Autenticacion requerida',
        message: 'Necesitas iniciar sesion para continuar.',
        action: 'Iniciar sesion',
    },
    AUTH_RESEND_FAILED: {
        title: 'Error al reenviar',
        message: 'No se pudo reenviar el correo de confirmacion. Intenta de nuevo.',
    },

    // ── Optimizacion ─────────────────────────────────────
    OPTIMIZATION_INFEASIBLE: {
        title: 'No pudimos optimizar',
        message: 'No encontramos una combinacion optima con esas acciones. Intenta agregar mas opciones.',
        action: 'Agregar mas tickers',
    },
    OPTIMIZATION_MIN_TICKERS: {
        title: 'Tickers insuficientes',
        message: 'Necesitas al menos 2 tickers para optimizar.',
        action: 'Agregar mas tickers',
    },
    OPTIMIZATION_INVALID_BUDGET: {
        title: 'Presupuesto invalido',
        message: 'El presupuesto debe ser mayor a 0.',
    },
    OPTIMIZATION_INSUFFICIENT_DATA: {
        title: 'Datos insuficientes',
        message: 'No hay suficiente historial para estos tickers. Se requieren al menos 6 meses de datos.',
        action: 'Probar otros tickers',
    },
    OPTIMIZATION_INVALID_TICKERS: {
        title: 'Tickers invalidos',
        message: 'Algunos tickers no fueron encontrados o no tienen datos disponibles.',
        action: 'Verificar tickers',
    },

    // ── Plan / Permisos ──────────────────────────────────
    PLAN_UPGRADE_REQUIRED: {
        title: 'Plan insuficiente',
        message: 'Esta funcionalidad requiere un plan superior.',
        action: 'Ver planes',
    },

    // ── Datos / Precios ──────────────────────────────────
    DATA_TICKER_NOT_FOUND: {
        title: 'Ticker no encontrado',
        message: 'No se encontro informacion para el ticker solicitado.',
    },
    DATA_FETCH_ERROR: {
        title: 'Error de datos',
        message: 'No se pudieron obtener los datos del mercado. Intenta de nuevo.',
    },
    DATA_PRICE_NOT_FOUND: {
        title: 'Sin cotizacion',
        message: 'No se encontro cotizacion para el ticker solicitado.',
    },

    // ── Soporte ──────────────────────────────────────────
    SUPPORT_TICKET_FAILED: {
        title: 'Error al enviar',
        message: 'No se pudo enviar el ticket de soporte. Intenta de nuevo.',
    },

    // ── Perfil ───────────────────────────────────────────
    PROFILE_UPDATE_FAILED: {
        title: 'Error al actualizar',
        message: 'No se pudo actualizar la configuracion. Intenta de nuevo.',
    },
    PROFILE_RECALC_FAILED: {
        title: 'Error al recalcular',
        message: 'No se pudo recalcular el perfil. Intenta de nuevo.',
    },

    // ── Estrategias ──────────────────────────────────────
    STRATEGY_NOT_FOUND: {
        title: 'Estrategia no encontrada',
        message: 'No se encontro la estrategia solicitada.',
    },
    STRATEGY_SAVE_FAILED: {
        title: 'Error al guardar',
        message: 'No se pudo guardar el portafolio. Intenta de nuevo.',
    },

    // ── Red / Generico ───────────────────────────────────
    NETWORK_ERROR: {
        title: 'Error de conexion',
        message: 'Error de conexion. Verifica tu internet e intenta de nuevo.',
    },
    RATE_LIMITED: {
        title: 'Demasiadas solicitudes',
        message: 'Has excedido el limite de solicitudes. Espera un momento antes de intentar de nuevo.',
    },
    SERVER_ERROR: {
        title: 'Error del servidor',
        message: 'Ocurrio un error interno. Intenta de nuevo mas tarde.',
    },
    UNKNOWN: {
        title: 'Error inesperado',
        message: 'Ocurrio un error inesperado. Intenta de nuevo.',
    },
};

/**
 * Retrieves the appropriate error message for a given error code.
 * Falls back to UNKNOWN if the code is not found.
 */
export function getErrorMessage(errorCode: string): ErrorMessage {
    return ERROR_MESSAGES[errorCode] ?? ERROR_MESSAGES.UNKNOWN;
}

/**
 * Parses an API error response and returns the error code.
 * Expects the backend to return { code: string, message: string } in the detail field.
 * Falls back to mapping known string patterns for legacy compatibility.
 */
export function parseApiError(error: unknown): ErrorMessage {
    if (error && typeof error === 'object' && 'code' in error) {
        return getErrorMessage((error as { code: string }).code);
    }

    // Legacy: try to match known string patterns from backend detail
    const msg = error instanceof Error ? error.message : String(error);

    if (msg.includes('Invalid login credentials')) return ERROR_MESSAGES.AUTH_INVALID_CREDENTIALS;
    if (msg.includes('Email not confirmed')) return ERROR_MESSAGES.AUTH_EMAIL_NOT_CONFIRMED;
    if (msg.includes('already registered')) return ERROR_MESSAGES.AUTH_USER_EXISTS;
    if (msg.includes('sesion expir') || msg.includes('sesión expir')) return ERROR_MESSAGES.AUTH_SESSION_EXPIRED;
    if (msg.includes('al menos 2 tickers') || msg.includes('at least 2 tickers')) return ERROR_MESSAGES.OPTIMIZATION_MIN_TICKERS;
    if (msg.includes('presupuesto') || msg.includes('budget')) return ERROR_MESSAGES.OPTIMIZATION_INVALID_BUDGET;
    if (msg.includes('no convergi') || msg.includes('infeasible')) return ERROR_MESSAGES.OPTIMIZATION_INFEASIBLE;
    if (msg.includes('Datos insuficientes') || msg.includes('insufficient data')) return ERROR_MESSAGES.OPTIMIZATION_INSUFFICIENT_DATA;
    if (msg.includes('invalidos o sin datos') || msg.includes('invalid tickers')) return ERROR_MESSAGES.OPTIMIZATION_INVALID_TICKERS;
    if (msg.includes('plan') && (msg.includes('Pro') || msg.includes('Ultra') || msg.includes('exclusivo'))) return ERROR_MESSAGES.PLAN_UPGRADE_REQUIRED;
    if (msg.includes('429') || msg.includes('rate limit') || msg.includes('limite de solicitudes')) return ERROR_MESSAGES.RATE_LIMITED;

    return ERROR_MESSAGES.UNKNOWN;
}

/**
 * Formats an ErrorMessage for display in a toast notification.
 */
export function formatErrorToast(err: ErrorMessage): string {
    return `${err.title}: ${err.message}`;
}
