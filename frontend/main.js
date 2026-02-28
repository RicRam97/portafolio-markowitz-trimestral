// API Configuration (Dynamic for Production vs Localhost)
const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
// IMPORTANT: Render will assign a public URL to your backend (e.g., https://portafolio-markowitz-api.onrender.com). Update this string once you deploy!
const PROD_API_BASE = "https://portafolio-markowitz-api.onrender.com/api";
const API_BASE = isLocalhost ? "http://localhost:8000/api" : PROD_API_BASE;

// State
let availableTickers = [];
let selectedTickers = new Set();
let chartInstances = {};

// DOM Elements
const tickerListEl = document.getElementById("ticker-list");
const tickerCountEl = document.getElementById("ticker-count");
const newTickerInput = document.getElementById("new-ticker");
const btnAddTicker = document.getElementById("btn-add-ticker");
const btnSelectAll = document.getElementById("btn-select-all");
const btnClearAll = document.getElementById("btn-clear-all");
const budgetInput = document.getElementById("budget");
const btnOptimize = document.getElementById("btn-optimize");
const btnOptimizeText = document.getElementById("btn-optimize-text");
const btnExportPdf = document.getElementById("btn-export-pdf");
const spinner = document.getElementById("loading-spinner");
const errorToast = document.getElementById("error-message");

const allocBody = document.getElementById("allocation-body");
const metReturn = document.getElementById("metric-return");
const metVol = document.getElementById("metric-volatility");
const metSharpe = document.getElementById("metric-sharpe");
const metRemCash = document.getElementById("metric-rem-cash");

// Default Chart.js Config for dark mode
Chart.defaults.color = "#94a3b8";
Chart.defaults.font.family = "Inter";

// Initialize
async function init() {
  await fetchTickers();
  renderTickerList();

  // Bind Events
  btnAddTicker.addEventListener("click", handleAddTicker);
  btnSelectAll.addEventListener("click", () => {
    availableTickers.forEach(t => selectedTickers.add(t));
    renderTickerList();
  });
  btnClearAll.addEventListener("click", () => {
    selectedTickers.clear();
    renderTickerList();
  });
  newTickerInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleAddTicker();
  });
  btnOptimize.addEventListener("click", handleOptimization);
  btnExportPdf.addEventListener("click", handleExportPdf);
}

// Fetch base tickers from YAML via API
async function fetchTickers() {
  try {
    const res = await fetch(`${API_BASE}/tickers`);
    const data = await res.json();
    availableTickers = data.tickers || [];
    // Select the first 20 by default so we don't bombard the API on first run
    availableTickers.slice(0, 20).forEach(t => selectedTickers.add(t));
  } catch (err) {
    showError("Could not load tickers from server.");
  }
}

function renderTickerList() {
  tickerListEl.innerHTML = "";
  availableTickers.sort().forEach(ticker => {
    const div = document.createElement("div");
    div.className = "ticker-item";
    const checked = selectedTickers.has(ticker) ? "checked" : "";

    div.innerHTML = `
      <input type="checkbox" id="chk-${ticker}" value="${ticker}" ${checked}>
      <label for="chk-${ticker}">${ticker}</label>
    `;

    div.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) selectedTickers.add(ticker);
      else selectedTickers.delete(ticker);
      updateTickerCount();
    });

    tickerListEl.appendChild(div);
  });
  updateTickerCount();
}

function updateTickerCount() {
  tickerCountEl.textContent = `${selectedTickers.size} selected`;
}

function handleAddTicker() {
  const val = newTickerInput.value.trim().toUpperCase();
  if (val && !availableTickers.includes(val)) {
    availableTickers.push(val);
    selectedTickers.add(val);
    newTickerInput.value = "";
    renderTickerList();
  }
}

function showError(msg) {
  errorToast.textContent = msg;
  errorToast.classList.remove("hidden");
  setTimeout(() => errorToast.classList.add("hidden"), 5000);
}

function setLoading(isLoading) {
  if (isLoading) {
    btnOptimize.disabled = true;
    btnOptimizeText.classList.add("hidden");
    spinner.classList.remove("hidden");
  } else {
    btnOptimize.disabled = false;
    btnOptimizeText.classList.remove("hidden");
    spinner.classList.add("hidden");
  }
}

// Optimization Flow
async function handleOptimization() {
  if (selectedTickers.size < 2) {
    showError("Please select at least 2 tickers to optimize.");
    return;
  }

  const budget = parseFloat(budgetInput.value);
  if (isNaN(budget) || budget < 100) {
    showError("Please enter a valid budget (minimum $100).");
    return;
  }

  setLoading(true);
  errorToast.classList.add("hidden");

  // Fixed 3-year lookback for now
  const end = new Date();
  const start = new Date();
  start.setFullYear(end.getFullYear() - 3);

  const payload = {
    tickers: Array.from(selectedTickers),
    budget: budget,
    start_date: start.toISOString().split('T')[0],
    end_date: end.toISOString().split('T')[0]
  };

  try {
    // 1. Optimize
    const resOpt = await fetch(`${API_BASE}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resOpt.ok) throw new Error((await resOpt.json()).detail || "Optimization failed");
    const dataOpt = await resOpt.json();

    renderResults(dataOpt);
    renderDoughnutChart(dataOpt.allocation);

    // 2. Track History in parallel after optimize returns
    const resTrack = await fetch(`${API_BASE}/track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (resTrack.ok) {
      const dataTrack = await resTrack.json();
      renderHistoryChart(dataTrack);
    }

    // Flag to allow export
    window.HAS_OPTIMIZED = true;

  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  // Top Metrics
  metReturn.textContent = `${data.metrics.expected_return_annual_pct}%`;
  metVol.textContent = `${data.metrics.volatility_annual_pct}%`;
  metSharpe.textContent = data.metrics.sharpe_ratio;
  metRemCash.textContent = `$${data.budget_analysis.remaining_cash}`;

  // Table
  allocBody.innerHTML = "";
  if (data.allocation.length === 0) {
    allocBody.innerHTML = `<tr><td colspan="5" class="empty-state">No shares allocated (budget too low)</td></tr>`;
    return;
  }

  data.allocation.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${row.ticker}</strong></td>
      <td>${row.weight_pct}%</td>
      <td>$${row.last_price}</td>
      <td style="color:#3b82f6; font-weight:bold">${row.shares_to_buy}</td>
      <td>$${row.dollar_allocation}</td>
    `;
    allocBody.appendChild(tr);
  });
}

// --- Chart.js Renderers ---

function generateColors(count) {
  const hues = [217, 160, 45, 340, 280, 10, 190, 80]; // Tailwind-ish hues
  return Array.from({ length: count }, (_, i) => `hsl(${hues[i % hues.length]}, 80%, 60%)`);
}

function renderDoughnutChart(allocation) {
  const ctx = document.getElementById("allocation-chart").getContext("2d");
  if (chartInstances.doughnut) chartInstances.doughnut.destroy();

  const labels = allocation.map(a => a.ticker);
  const data = allocation.map(a => a.dollar_allocation);

  chartInstances.doughnut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: generateColors(labels.length),
        borderWidth: 0,
        hoverOffset: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '75%',
      plugins: {
        legend: { position: 'right', labels: { boxWidth: 12, padding: 15 } }
      }
    }
  });
}

function renderHistoryChart(trackData) {
  const ctx = document.getElementById("history-chart").getContext("2d");
  if (chartInstances.line) chartInstances.line.destroy();

  chartInstances.line = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trackData.dates,
      datasets: [
        {
          label: 'Optimized Portfolio',
          data: trackData.portfolio,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.1
        },
        {
          label: 'S&P 500 (^GSPC)',
          data: trackData.benchmark,
          borderColor: '#94a3b8',
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { maxTicksLimit: 8 } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

async function handleExportPdf() {
  console.log("Export triggered!");

  if (!window.HAS_OPTIMIZED) {
    showError("Please run 'Optimize Portfolio' to generate data before exporting.");
    return;
  }

  if (typeof window.html2pdf === 'undefined') {
    showError("PDF renderer failed to load. Please check your internet connection.");
    console.error("html2pdf is not defined in window object.");
    return;
  }

  btnExportPdf.disabled = true;
  const originalText = btnExportPdf.innerHTML;
  btnExportPdf.innerText = "Exporting...";

  try {
    // 1. Aplicar clases de exportación
    document.body.classList.add("pdf-exporting-body");
    const element = document.getElementById("printable-dashboard");
    element.classList.add("pdf-exporting");

    // 2. CRUCIAL: Esperar a que el navegador termine de aplicar los estilos
    await new Promise(resolve => setTimeout(resolve, 500));

    // 3. Calcular la altura real del contenido después del ajuste
    const contentHeight = Math.max(element.scrollHeight, 2000); // Mínimo 2000px
    console.log("Content height detected:", contentHeight);

    // 4. Configurar opciones de PDF
    const opt = {
      margin: [10, 10, 10, 10], // Márgenes reducidos para aprovechar más espacio
      filename: 'Portfolio_Report.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        logging: true, // Cambiado a true para debugging
        windowWidth: 1200,
        windowHeight: contentHeight, // Usar altura calculada
        scrollY: -window.scrollY, // Compensar scroll
        scrollX: -window.scrollX,
        x: 0,
        y: 0,
        width: 1200, // Forzar ancho
        height: contentHeight // Forzar altura completa
      },
      jsPDF: {
        unit: 'mm',
        format: 'a3',
        orientation: 'portrait'
      },
      pagebreak: {
        mode: ['avoid-all', 'css', 'legacy'],
        before: '.page-break-before',
        after: '.page-break-after',
        avoid: ['.metric-card', '.chart-card', '.table-section']
      }
    };

    console.log("Starting html2pdf render engine...");

    // 5. Generar PDF
    await window.html2pdf().set(opt).from(element).save();

    console.log("PDF successfully saved!");

  } catch (e) {
    console.error("PDF Export Exception:", e);
    showError("An error occurred generating the PDF. Check console.");
  } finally {
    // 6. Restaurar estado normal
    document.body.classList.remove("pdf-exporting-body");
    const element = document.getElementById("printable-dashboard");
    element.classList.remove("pdf-exporting");
    btnExportPdf.innerHTML = originalText;
    btnExportPdf.disabled = false;
  }
}

// Boot
init();
