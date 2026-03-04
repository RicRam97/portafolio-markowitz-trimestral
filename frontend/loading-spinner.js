/**
 * loading-spinner.js
 * 
 * Componente interactivo para mostrar el estado de carga y conectarse al endpoint SSE de Kaudal.
 * 
 * Uso:
 * const spinner = new KaudalLoadingSpinner();
 * spinner.start("Calculando Portafolio...");
 * 
 * const result = await spinner.fetchWithSSE("/api/optimizar", {
 *     method: "POST",
 *     headers: { "Content-Type": "application/json" },
 *     body: JSON.stringify({ tickers: ["AAPL", "MSFT"], budget: 1000 })
 * });
 * 
 * // El spinner se cierra automáticamente si result.status === 'success' o si lo llamas explícitamente:
 * // spinner.stop();
 */

const KAUDAL_MESSAGES = [
    // 5 Mensajes de proceso
    "Descargando precios históricos...",
    "Calculando la matriz de covarianza...",
    "Buscando el portafolio con mejor Ratio de Sharpe...",
    "Analizando correlaciones entre tus activos...",
    "Ajustando pesos según tu perfil de riesgo...",

    // 5 Datos históricos
    "💡 El S&P 500 ha promediado un 10% de retorno anual desde su creación.",
    "💡 En 2008, el mercado cayó un 38%, pero se recuperó completamente en 4 años.",
    "💡 El interés compuesto es la octava maravilla del mundo.",
    "💡 A largo plazo, las acciones han superado a la inflación históricamente.",
    "💡 El mercado alcista más largo duró de 2009 a 2020 (11 años).",

    // 5 Frases célebres
    '"El riesgo viene de no saber lo que estás haciendo." — Warren Buffett',
    '"En este negocio, si eres bueno, tienes razón seis veces de cada diez." — Peter Lynch',
    '"No busques la aguja en el pajar. ¡Compra el pajar!" — John Bogle',
    '"La diversificación es el único almuerzo gratis en las finanzas." — Harry Markowitz',
    '"El inversor inteligente es un realista que vende a optimistas y compra a pesimistas." — Benjamin Graham',

    // 5 Datos curiosos
    "🇲🇽 La Bolsa Mexicana de Valores (BMV) fue fundada en 1894.",
    "🌎 La Bolsa de Nueva York (NYSE) maneja más de $20 billones en capitalización.",
    "🇲🇽 CETES Directo es el instrumento sin riesgo por excelencia en México.",
    "📈 El 'Efecto Enero' sugiere que las acciones rinden más en el primer mes del año.",
    "💰 La primera acción emitida en la historia fue de la Dutch East India Company en 1602.",

    // 5 Tips educativos
    "🎓 Diversificar reduce tu riesgo sin sacrificar necesariamente el retorno esperado.",
    "🎓 Rebalancear una vez al año ayuda a mantener tu perfil de riesgo objetivo.",
    "🎓 Invierte dinero que no necesites en los próximos 3 a 5 años.",
    "🎓 Las caídas del mercado son normales; no vendas en pánico.",
    "🎓 Las comisiones altas devoran el interés compuesto. Fíjate en los costos."
];

// Mapeo exhaustivo de etapas SSE a progreso (%)
const STAGE_PROGRESS = {
    "downloading_data": 20,
    "cleaning_data": 40,
    "calculating_covariance": 60,
    "optimizing": 80,
    "calculating_positions": 95,
    "done": 100
};

class KaudalLoadingSpinner {
    constructor() {
        this._createDOM();
        this.messageInterval = null;
        this.timeoutTimer = null;
        this.currentProgress = 0;
        this.controller = new AbortController();
    }

    _createDOM() {
        if (document.getElementById('kq-spinner-overlay')) return;

        this.overlay = document.createElement('div');
        this.overlay.id = 'kq-spinner-overlay';
        this.overlay.className = 'kq-spinner-overlay';

        this.overlay.innerHTML = `
            <div class="kq-spinner-container" id="kq-spinner-box">
                <div class="kq-error-icon">😕</div>
                
                <div class="kq-spinner-ring" id="kq-spinner-ring"></div>
                
                <h3 class="kq-spinner-title" id="kq-spinner-title">Optimizando Portafolio</h3>
                <p class="kq-spinner-message" id="kq-spinner-msg">Iniciando...</p>
                <p class="kq-error-text" id="kq-error-msg"></p>

                <div class="kq-progress-wrapper" id="kq-progress-wrap">
                    <div class="kq-progress-bar" id="kq-progress-bar"></div>
                </div>
                <div class="kq-progress-text" id="kq-progress-text">0%</div>

                <button class="kq-retry-btn" id="kq-retry-btn">Intentar de nuevo</button>
                <button class="kq-cancel-btn" id="kq-cancel-btn">Cancelar Operación</button>
            </div>
        `;

        document.body.appendChild(this.overlay);

        // Bind elements
        this.elTitle = document.getElementById('kq-spinner-title');
        this.elMsg = document.getElementById('kq-spinner-msg');
        this.elBox = document.getElementById('kq-spinner-box');
        this.elErrorMsg = document.getElementById('kq-error-msg');
        this.elProgressBar = document.getElementById('kq-progress-bar');
        this.elProgressText = document.getElementById('kq-progress-text');
        this.btnRetry = document.getElementById('kq-retry-btn');
        this.btnCancel = document.getElementById('kq-cancel-btn');

        // Listeners
        this.btnRetry.addEventListener('click', () => {
            if (this.onRetry) this.onRetry();
            else this.stop();
        });

        this.btnCancel.addEventListener('click', () => {
            this.controller.abort();
            this.stop();
        });
    }

    _shuffleArray(array) {
        const arr = [...array];
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    start(title = "Calculando...") {
        this.elTitle.textContent = title;
        this.elBox.className = 'kq-spinner-container'; // reset classes
        this.setProgress(0);
        this.elErrorMsg.textContent = "";

        // Setup messages
        this.messageQueue = this._shuffleArray(KAUDAL_MESSAGES);
        this.elMsg.textContent = this.messageQueue.pop();

        // Start Message Rotation every 3s
        this.messageInterval = setInterval(() => {
            if (this.messageQueue.length === 0) {
                this.messageQueue = this._shuffleArray(KAUDAL_MESSAGES);
            }
            this._animateTextChange(this.messageQueue.pop());
        }, 3000);

        // Start timeout monitor (15s)
        this.timeoutTimer = setTimeout(() => {
            this.elBox.classList.add('is-timeout');
            this.elMsg.textContent = "El cálculo está tardando más de lo esperado...";
        }, 15000);

        // Show overlay but don't strictly block body scrolling if not full SPA, just visual overlay
        this.overlay.classList.add('active');
        this.controller = new AbortController(); // Reset abort
    }

    _animateTextChange(newText) {
        this.elMsg.classList.add('fade-out');
        setTimeout(() => {
            this.elMsg.textContent = newText;
            this.elMsg.classList.remove('fade-out');
        }, 300); // matches CSS transition
    }

    setProgress(percent) {
        this.currentProgress = percent;
        this.elProgressBar.style.width = \`\${percent}%\`;
        this.elProgressText.textContent = \`\${percent}%\`;
    }

    showError(errorText, failedTickers = []) {
        this._clearTimers();
        this.elBox.className = 'kq-spinner-container is-error';
        
        let msg = errorText;
        if (failedTickers && failedTickers.length > 0) {
            msg += \`<br><br><strong>Problema con Tickers:</strong> \${failedTickers.join(", ")}.<br>Verifica y vuelve a intentarlo.\`;
        }
        
        this.elErrorMsg.innerHTML = msg;
    }

    stop() {
        this._clearTimers();
        this.overlay.classList.remove('active');
    }

    _clearTimers() {
        if (this.messageInterval) clearInterval(this.messageInterval);
        if (this.timeoutTimer) clearTimeout(this.timeoutTimer);
    }

    /**
     * Consume un endpoint que devuelve Server-Sent Events o un JSON crudo simulando la carga.
     * Retorna el Payload Final de éxito o lanza error.
     */
    async fetchWithSSE(url, fetchOptions = {}) {
        try {
            this.start();
            fetchOptions.signal = this.controller.signal;

            const response = await fetch(url, fetchOptions);

            if (!response.ok) {
                let errData;
                try { errData = await response.json(); } catch(e){}
                const detail = errData?.detail || response.statusText;
                throw new Error(typeof detail === 'object' ? detail.message || JSON.stringify(detail) : detail);
            }

            // Read SSE Stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\\n\\n');
                buffer = lines.pop(); // last partial chunk

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        try {
                            const dataObj = JSON.parse(dataStr);
                            
                            if (dataObj.stage === "error") {
                                this.showError(dataObj.detail, dataObj.failed_tickers);
                                return null;
                            }

                            if (dataObj.stage && STAGE_PROGRESS[dataObj.stage]) {
                                this.setProgress(STAGE_PROGRESS[dataObj.stage]);
                            }

                            if (dataObj.stage === "done" || dataObj.status === "success" || dataObj.result) {
                                this.setProgress(100);
                                setTimeout(() => this.stop(), 500); // short visual delay
                                return dataObj.result || dataObj; 
                            }
                        } catch (e) {
                            console.warn("Could not parse SSE line:", dataStr, e);
                        }
                    }
                }
            }

            this.stop();
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log("Fetch aborted by user.");
            } else {
                this.showError(error.message);
            }
            return null;
        }
    }
}

// Export for Modules if needed, otherwise available on browser global
if (typeof window !== "undefined") {
    window.KaudalLoadingSpinner = KaudalLoadingSpinner;
}
