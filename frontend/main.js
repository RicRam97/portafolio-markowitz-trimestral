// API Configuration (Dynamic for Production vs Localhost via Vite)
const isLocalhost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
const PROD_API_BASE = import.meta.env.VITE_API_BASE || "https://portafolio-markowitz-api-production.up.railway.app/api";
const API_BASE = isLocalhost ? "http://localhost:8000/api" : PROD_API_BASE;

// --- Auth / Session Manager ---
import { initSessionManager, getCurrentUser, logout, getSupabase } from './session-manager.js';

// --- State & Persistence ---
let allTickers = JSON.parse(localStorage.getItem('allTickers')) || [];
let tickerDetails = JSON.parse(localStorage.getItem('tickerDetails')) || {};
let favoriteTickers = new Set(JSON.parse(localStorage.getItem('favoriteTickers')) || []);
let selectedTickers = new Set(JSON.parse(localStorage.getItem('lastSelectedTickers')) || []);
let chartInstances = {};
let latestOptimizationData = null;

// Helper: XSS Protection
function escapeHTML(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Helper: Save state to LocalStorage
function saveState() {
  localStorage.setItem('allTickers', JSON.stringify(allTickers));
  localStorage.setItem('tickerDetails', JSON.stringify(tickerDetails));
  localStorage.setItem('favoriteTickers', JSON.stringify(Array.from(favoriteTickers)));
  localStorage.setItem('lastSelectedTickers', JSON.stringify(Array.from(selectedTickers)));
  localStorage.setItem('userBudget', budgetInput.value);
}

// DOM Elements — Views
const landingView = document.getElementById("landing-view");
const dashboardView = document.getElementById("dashboard-view");
const directoryView = document.getElementById("directory-view");
const aboutView = document.getElementById("about-view");
const faqView = document.getElementById("faq-view");
const conoceView = document.getElementById("conoce-view");
const planesView = document.getElementById("planes-view");
const demoView = document.getElementById("demo-view");

// DOM Elements — Nav
const navLinkLanding = document.getElementById("nav-link-landing");
const navLinkConoce = document.getElementById("nav-link-conoce");
const navLinkPlanes = document.getElementById("nav-link-planes");
const navLinkDemo = document.getElementById("nav-link-demo");
const navLinkFaq = document.getElementById("nav-link-faq");
const navLinkAbout = document.getElementById("nav-link-about");
const navBrand = document.getElementById("nav-brand");
const navHamburger = document.getElementById("nav-hamburger");
const navLinks = document.querySelector(".nav-links");

// Landing CTAs
const heroCtaDashboard = document.getElementById("hero-cta-dashboard");
const heroCtaBeginners = document.getElementById("hero-cta-beginners");
const footerCtaDashboard = document.getElementById("footer-cta-dashboard");
const footerLinkAbout = document.getElementById("footer-link-about");
const socialProofText = document.getElementById("social-proof-text");
const dataFreshnessText = document.getElementById("data-freshness-text");

const tickerListEl = document.getElementById("ticker-list");
const tickerCountEl = document.getElementById("ticker-count");
const newTickerInput = document.getElementById("new-ticker");
const btnAddTicker = document.getElementById("btn-add-ticker");
const btnSelectAll = document.getElementById("btn-select-all");
const btnClearAll = document.getElementById("btn-clear-all");
const budgetInput = document.getElementById("budget");
const strategySelect = document.getElementById("strategy");
const btnOptimize = document.getElementById("btn-optimize");
const btnOptimizeText = document.getElementById("btn-optimize-text");
const btnExportPdf = document.getElementById("btn-export-pdf");
const spinner = document.getElementById("loading-spinner");
const errorToast = document.getElementById("error-message");

const btnBackDashboard = document.getElementById("btn-back-dashboard");
const dirSearch = document.getElementById("dir-search");
const dirTickerList = document.getElementById("dir-ticker-list");
const btnDirFilterFav = document.getElementById("btn-dir-filter-fav");
const btnDirAddTicker = document.getElementById("btn-dir-add-ticker");
const dirNewTicker = document.getElementById("dir-new-ticker");
const sidebarSearch = document.getElementById("search-ticker");
const btnDirSelectFavs = document.getElementById("btn-dir-select-favs");
const btnDirSelectAll = document.getElementById("btn-dir-select-all");
const dirFilterMarket = document.getElementById("dir-filter-market");
const dirFilterSector = document.getElementById("dir-filter-sector");

let showOnlyFavorites = false;

const allocBody = document.getElementById("allocation-body");
const metReturn = document.getElementById("metric-return");
const metVol = document.getElementById("metric-volatility");
const metSharpe = document.getElementById("metric-sharpe");
const metRemCash = document.getElementById("metric-rem-cash");

// --- Beginner UX Phase 9 ---
const thematicBaskets = document.getElementById("thematic-baskets");
const advTableSection = document.getElementById("advanced-table-section");
const dreamsRecipeSection = document.getElementById("dreams-recipe");
const recipeList = document.getElementById("recipe-list");
const recipeRemCash = document.getElementById("recipe-rem-cash");

const BRAND_OVERRIDES = {
  "AAPL": { name: "Apple", logo: "https://logo.clearbit.com/apple.com", color: "#A3AAAE" },
  "GOOGL": { name: "Google", logo: "https://logo.clearbit.com/google.com", color: "#4285F4" },
  "MSFT": { name: "Microsoft", logo: "https://logo.clearbit.com/microsoft.com" },
  "NVDA": { name: "Nvidia", logo: "https://logo.clearbit.com/nvidia.com" },
  "FEMSAUBD.MX": { name: "OXXO / Femsa", logo: "https://logo.clearbit.com/femsa.com", color: "#E31837" },
  "WALMEX.MX": { name: "Walmart de México", logo: "https://logo.clearbit.com/walmart.com.mx", color: "#0071CE" },
  "BIMBOA.MX": { name: "Grupo Bimbo", logo: "https://logo.clearbit.com/grupobimbo.com", color: "#1E3B8B" },
  "ALSEA.MX": { name: "Alsea", logo: "https://logo.clearbit.com/alsea.net" },
  "KO": { name: "Coca-Cola", logo: "https://logo.clearbit.com/coca-cola.com", color: "#F40009" },
  "JNJ": { name: "Johnson & Johnson", logo: "https://logo.clearbit.com/jnj.com" },
  "PG": { name: "Procter & Gamble", logo: "https://logo.clearbit.com/pg.com" },
  "PEP": { name: "PepsiCo", logo: "https://logo.clearbit.com/pepsico.com" },
  "TSLA": { name: "Tesla", logo: "https://logo.clearbit.com/tesla.com" },
  "ENPH": { name: "Enphase Energy", logo: "https://logo.clearbit.com/enphase.com" },
  "NEE": { name: "NextEra Energy", logo: "https://logo.clearbit.com/nexteraenergy.com" },
  "FSLR": { name: "First Solar", logo: "https://logo.clearbit.com/firstsolar.com" }
};

const BASKETS = {
  tech: ["AAPL", "GOOGL", "MSFT", "NVDA"],
  mexico: ["FEMSAUBD.MX", "BIMBOA.MX", "WALMEX.MX", "ALSEA.MX"],
  dividend: ["KO", "JNJ", "PG", "PEP"],
  green: ["TSLA", "ENPH", "NEE", "FSLR"]
};

// --- Dreams Test Mode ---
const btnModeAdvanced = document.getElementById("btn-mode-advanced");
const btnModeDreams = document.getElementById("btn-mode-dreams");
const advancedConfig = document.getElementById("advanced-config");
const advancedMetrics = document.getElementById("advanced-metrics");
const dreamsMetrics = document.getElementById("dreams-metrics");
const dreamMetricGain = document.getElementById("dream-metric-gain");
const dreamMetricLoss = document.getElementById("dream-metric-loss");
const dreamsProjections = document.getElementById("dreams-projections");
const dreamMetricDivs = document.getElementById("dream-metric-divs");

const dreamsModal = document.getElementById("dreams-modal");
const btnCloseDreams = document.getElementById("btn-close-dreams");
const dreamsForm = document.getElementById("dreams-form");
const dreamCost = document.getElementById("dream-cost");
const dreamYears = document.getElementById("dream-years");
const dreamInitial = document.getElementById("dream-initial");
const dreamMonthly = document.getElementById("dream-monthly");
const dreamError = document.getElementById("dream-error");
const dreamSpinner = document.getElementById("dream-spinner");
const dreamBtnText = document.getElementById("dream-btn-text");
const dreamResults = document.getElementById("dream-results");
const dreamProfileTitle = document.getElementById("dream-profile-title");
const dreamProfileDesc = document.getElementById("dream-profile-desc");
const dreamRate = document.getElementById("dream-rate");
const btnApplyDream = document.getElementById("btn-apply-dream");

// Default Chart.js Config for dark mode
Chart.defaults.color = "#94a3b8";
Chart.defaults.font.family = "Inter";

// ===== View Navigation System =====
function navigateTo(viewName) {
  if (viewName === "dashboard") {
    window.location.href = './dashboard.html';
    return;
  }

  // Hide all views
  landingView.classList.add("hidden");
  dashboardView.classList.add("hidden");
  directoryView.classList.add("hidden");
  aboutView.classList.add("hidden");
  faqView.classList.add("hidden");
  if (conoceView) conoceView.classList.add("hidden");
  if (planesView) planesView.classList.add("hidden");
  if (demoView) demoView.classList.add("hidden");

  // Deactivate all nav links
  navLinkLanding.classList.remove("active");
  if (navLinkConoce) navLinkConoce.classList.remove("active");
  if (navLinkPlanes) navLinkPlanes.classList.remove("active");
  if (navLinkDemo) navLinkDemo.classList.remove("active");
  navLinkFaq.classList.remove("active");
  navLinkAbout.classList.remove("active");

  // Close mobile menu
  navLinks.classList.remove("open");

  // Show the target view
  if (viewName === "landing") {
    landingView.classList.remove("hidden");
    landingView.classList.add("fade-in");
    navLinkLanding.classList.add("active");
    document.body.classList.remove("view-app");
  } else if (viewName === "conoce") {
    if (conoceView) {
      conoceView.classList.remove("hidden");
      conoceView.classList.add("fade-in");
    }
    if (navLinkConoce) navLinkConoce.classList.add("active");
    document.body.classList.add("view-app");
  } else if (viewName === "planes") {
    if (planesView) {
      planesView.classList.remove("hidden");
      planesView.classList.add("fade-in");
    }
    if (navLinkPlanes) navLinkPlanes.classList.add("active");
    document.body.classList.add("view-app");
  } else if (viewName === "demo") {
    if (demoView) {
      demoView.classList.remove("hidden");
      demoView.classList.add("fade-in");
      renderDemoView(); // Render data when tab is opened
    }
    if (navLinkDemo) navLinkDemo.classList.add("active");
    document.body.classList.add("view-app");
  } else if (viewName === "directory") {
    directoryView.classList.remove("hidden");
    directoryView.classList.add("fade-in");
    document.body.classList.add("view-app");
    dirSearch.value = "";
    renderDirectoryList("");
  } else if (viewName === "about") {
    aboutView.classList.remove("hidden");
    aboutView.classList.add("fade-in");
    navLinkAbout.classList.add("active");
    document.body.classList.add("view-app");
  } else if (viewName === "faq") {
    faqView.classList.remove("hidden");
    faqView.classList.add("fade-in");
    navLinkFaq.classList.add("active");
    document.body.classList.add("view-app");
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ===== Social Proof & Data Freshness =====
async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) return;
    const data = await res.json();
    const count = data.optimizations_count || 0;
    if (count > 0) {
      socialProofText.textContent = `🔬 Más de ${count.toLocaleString()} portafolios optimizados`;
    } else {
      socialProofText.textContent = `🔬 Herramienta activa — optimiza tu primer portafolio`;
    }
  } catch {
    socialProofText.textContent = `🔬 Herramienta activa — optimiza tu primer portafolio`;
  }
}

// Initialize
async function init() {
  // --- Auth: Initialize session manager & update nav ---
  try {
    initSessionManager();
    const user = await getCurrentUser();
    const navAuthBtn = document.getElementById('nav-link-auth');
    if (navAuthBtn) {
      if (user) {
        navAuthBtn.textContent = 'Mi cuenta';
        navAuthBtn.href = '#';
        navAuthBtn.addEventListener('click', async (e) => {
          e.preventDefault();
          if (confirm('¿Deseas cerrar sesión?')) {
            await logout();
          }
        });
      } else {
        navAuthBtn.textContent = 'Iniciar sesión';
        navAuthBtn.href = './login.html';
      }
    }
  } catch (err) {
    console.log('[Auth] Session manager not available:', err.message);
  }

  // --- Auth: Handle redirect query params from login ---
  const urlParams = new URLSearchParams(window.location.search);
  const authView = urlParams.get('view');
  if (authView) {
    // Clean URL
    history.replaceState(null, '', window.location.pathname);
  }

  // Load saved budget
  const savedBudget = localStorage.getItem('userBudget');
  if (savedBudget) {
    budgetInput.value = savedBudget;
  }

  fetchTickers().then(() => {
    renderTickerList();
  });

  // ===== Navigation =====
  navLinkLanding.addEventListener("click", (e) => { e.preventDefault(); navigateTo("landing"); });
  if (navLinkConoce) navLinkConoce.addEventListener("click", (e) => { e.preventDefault(); navigateTo("conoce"); });
  if (navLinkPlanes) navLinkPlanes.addEventListener("click", (e) => { e.preventDefault(); navigateTo("planes"); });
  if (navLinkDemo) navLinkDemo.addEventListener("click", (e) => { e.preventDefault(); navigateTo("demo"); });
  navLinkFaq.addEventListener("click", (e) => { e.preventDefault(); navigateTo("faq"); });
  navLinkAbout.addEventListener("click", (e) => { e.preventDefault(); navigateTo("about"); });
  navBrand.addEventListener("click", () => navigateTo("landing"));
  footerLinkAbout.addEventListener("click", (e) => { e.preventDefault(); navigateTo("about"); });

  // New footer navigation links
  const footerFaq = document.getElementById("footer-link-faq-bottom");
  if (footerFaq) footerFaq.addEventListener("click", (e) => { e.preventDefault(); navigateTo("faq"); });
  const footerDash = document.getElementById("footer-link-dashboard-bottom");
  if (footerDash) footerDash.addEventListener("click", (e) => { e.preventDefault(); navigateTo("dashboard"); });

  // About page buttons
  document.getElementById("about-btn-dashboard").addEventListener("click", () => navigateTo("dashboard"));
  document.getElementById("about-btn-landing").addEventListener("click", () => navigateTo("landing"));

  // Ticker validation modal buttons
  document.getElementById("tv-confirm").addEventListener("click", () => {
    const modal = document.getElementById("ticker-validate-modal");
    const data = modal._pendingTicker;
    if (data) {
      if (!allTickers.includes(data.ticker)) {
        allTickers.push(data.ticker);
      }
      tickerDetails[data.ticker] = { name: data.name, sector: data.sector, market: data.market };
      if (favoriteTickers.size < 120) {
        favoriteTickers.add(data.ticker);
        selectedTickers.add(data.ticker);
      }
      saveState();
      dirNewTicker.value = "";
      renderDirectoryList(dirSearch.value.toUpperCase());
    }
    modal.classList.add("hidden");
  });
  document.getElementById("tv-cancel").addEventListener("click", () => {
    document.getElementById("ticker-validate-modal").classList.add("hidden");
  });
  document.getElementById("tv-error-close").addEventListener("click", () => {
    document.getElementById("ticker-validate-modal").classList.add("hidden");
  });

  // Hamburger toggle
  navHamburger.addEventListener("click", () => {
    navLinks.classList.toggle("open");
  });

  // Hero CTAs
  heroCtaDashboard.addEventListener("click", () => navigateTo("dashboard"));
  footerCtaDashboard.addEventListener("click", () => navigateTo("dashboard"));
  heroCtaBeginners.addEventListener("click", () => {
    navigateTo("dashboard");
    // Trigger beginner mode after a small delay for view to load
    setTimeout(() => {
      btnModeDreams.click();
    }, 350);
  });

  // Fetch social proof stats on load + every 60s
  fetchStats();
  setInterval(fetchStats, 60000);

  // --- Auth: Navigate to correct view if redirected from login ---
  if (authView === 'dashboard') {
    navigateTo('dashboard');
  } else if (authView === 'dreams') {
    navigateTo('dashboard');
    setTimeout(() => {
      btnModeDreams.click();
    }, 350);
  }

  // Bind Events
  budgetInput.addEventListener("change", () => {
    saveState();
  });
  btnAddTicker.addEventListener("click", handleAddTicker);
  btnSelectAll.addEventListener("click", () => {
    favoriteTickers.forEach(t => selectedTickers.add(t)); // Select all FAVORITES only
    saveState();
    renderTickerList();
  });
  btnClearAll.addEventListener("click", () => {
    selectedTickers.clear();
    saveState();
    renderTickerList();
  });

  // Search Bar logic
  sidebarSearch.addEventListener("input", (e) => {
    const q = e.target.value.toUpperCase();
    document.querySelectorAll(".sidebar-ticker").forEach(el => {
      const txt = el.textContent.toUpperCase();
      el.style.display = txt.includes(q) ? "" : "none";
    });
  });

  // --- Dreams Test Logic ---
  if (btnModeAdvanced) {
    btnModeAdvanced.addEventListener("click", () => {
      btnModeAdvanced.classList.add("active");
      btnModeDreams.classList.remove("active");
      advancedConfig.classList.remove("hidden");
      thematicBaskets.classList.add("hidden");

      // Toggle metric views if data exists
      if (latestOptimizationData) {
        advancedMetrics.classList.remove("hidden");
        dreamsMetrics.classList.add("hidden");
        dreamsProjections.classList.add("hidden");
        advTableSection.classList.remove("hidden");
        dreamsRecipeSection.classList.add("hidden");
      }
    });
  }

  // --- Landing Page Preview Chart & Toggles ---
  initPreviewCharts();
  initDemoCharts();



  btnModeDreams.addEventListener("click", () => {
    // Open Modal
    dreamsModal.classList.remove("hidden");
    // We don't change the active tabs until they finish the test.
  });

  btnCloseDreams.addEventListener("click", () => {
    dreamsModal.classList.add("hidden");
  });

  dreamsForm.addEventListener("submit", handleDreamsTestSubmit);

  btnApplyDream.addEventListener("click", () => {
    // Switch UI modes
    btnModeDreams.classList.add("active");
    btnModeAdvanced.classList.remove("active");
    advancedConfig.classList.add("hidden");
    thematicBaskets.classList.remove("hidden");

    // Toggle metric views
    if (latestOptimizationData) {
      advancedMetrics.classList.add("hidden");
      dreamsMetrics.classList.remove("hidden");
      dreamsProjections.classList.remove("hidden");
      advTableSection.classList.add("hidden");
      dreamsRecipeSection.classList.remove("hidden");
    }

    // Transfer the calculated budget
    const initial = parseFloat(dreamInitial.value);
    budgetInput.value = isNaN(initial) ? 10000 : initial;

    dreamsModal.classList.add("hidden");

    // Trigger optimization automatically
    handleOptimization();
  });

  // Directory Page logic
  btnBackDashboard.addEventListener("click", () => {
    navigateTo("dashboard");
  });

  dirSearch.addEventListener("input", (e) => {
    renderDirectoryList(e.target.value.toUpperCase());
  });

  dirFilterMarket.addEventListener("change", () => {
    renderDirectoryList(dirSearch.value.toUpperCase());
  });

  dirFilterSector.addEventListener("change", () => {
    renderDirectoryList(dirSearch.value.toUpperCase());
  });

  btnDirFilterFav.addEventListener("click", () => {
    showOnlyFavorites = !showOnlyFavorites;
    btnDirFilterFav.classList.toggle("btn-primary", showOnlyFavorites);
    btnDirFilterFav.classList.toggle("btn-secondary", !showOnlyFavorites);
    btnDirFilterFav.textContent = showOnlyFavorites ? "Todos los Tickers" : "Ver Favs";
    renderDirectoryList(dirSearch.value.toUpperCase());
  });

  dirNewTicker.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleDirAddTicker();
  });
  btnDirAddTicker.addEventListener("click", handleDirAddTicker);

  btnDirSelectFavs.addEventListener("click", () => {
    let addedCount = 0;
    const currentFavs = Array.from(favoriteTickers);
    for (const ticker of currentFavs) {
      if (selectedTickers.size >= 120 && !selectedTickers.has(ticker)) {
        showError("Se puede seleccionar un máximo de 120 tickers para el análisis.", false);
        break;
      }
      if (!selectedTickers.has(ticker)) {
        selectedTickers.add(ticker);
        addedCount++;
      }
    }

    if (addedCount > 0) {
      saveState();
      renderDirectoryList(dirSearch.value.toUpperCase());
    }
  });

  btnDirSelectAll.addEventListener("click", () => {
    let addedCount = 0;
    // Get currently visible list (filtered or not)
    const currentList = Array.from(dirTickerList.children).map(div => div.querySelector('span').textContent);

    for (const ticker of currentList) {
      if (selectedTickers.size >= 120 && !selectedTickers.has(ticker)) {
        showError("Se puede seleccionar un máximo de 120 tickers para el análisis.", false);
        break;
      }
      if (!selectedTickers.has(ticker)) {
        selectedTickers.add(ticker);
        // Also add to favorites if we are auto-selecting it
        if (!favoriteTickers.has(ticker)) {
          favoriteTickers.add(ticker);
        }
        addedCount++;
      }
    }

    if (addedCount > 0) {
      saveState();
      renderDirectoryList(dirSearch.value.toUpperCase());
    }
  });

  newTickerInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleAddTicker();
  });
  btnOptimize.addEventListener("click", handleOptimization);
  btnExportPdf.addEventListener("click", handleExportPdf);

  strategySelect.addEventListener("change", () => {
    if (latestOptimizationData) {
      renderDashboard(latestOptimizationData);
    }
  });

  // Baskets logic
  document.querySelectorAll('.basket-card').forEach(card => {
    card.addEventListener('click', (e) => {
      const theme = e.currentTarget.dataset.basket;
      const tickers = BASKETS[theme];

      // Clear selection for beginners
      selectedTickers.clear();

      tickers.forEach(t => {
        selectedTickers.add(t);
        if (!favoriteTickers.has(t)) favoriteTickers.add(t);
      });

      saveState();
      renderTickerList();

    });
  });
} // End init()

// ============================================
// Landing Page Data Visualization 
// ============================================

const DEMO_PORTFOLIO = [
  { ticker: "AAPL", name: "Apple", weight: 0.15, color: "#A3AAAE" },
  { ticker: "MSFT", name: "Microsoft", weight: 0.15, color: "#00A4EF" },
  { ticker: "GOOGL", name: "Google", weight: 0.15, color: "#4285F4" },
  { ticker: "AMZN", name: "Amazon", weight: 0.10, color: "#FF9900" },
  { ticker: "META", name: "Meta", weight: 0.10, color: "#0668E1" },
  { ticker: "NVDA", name: "Nvidia", weight: 0.10, color: "#76B900" },
  { ticker: "JPM", name: "JPMorgan Chase", weight: 0.10, color: "#111" },
  { ticker: "WMT", name: "Walmart", weight: 0.05, color: "#0071CE" },
  { ticker: "TSLA", name: "Tesla", weight: 0.05, color: "#E31937" },
  { ticker: "OXY", name: "Occidental Petroleum", weight: 0.05, color: "#004B87" }
];

let previewChartInstance = null;
let demoChartInstance = null;

function renderPieChart(ctxId, data, labels, colors) {
  const ctx = document.getElementById(ctxId);
  if (!ctx) return null;

  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors,
        borderWidth: 1,
        borderColor: 'rgba(15, 23, 42, 1)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#94a3b8' } }
      }
    }
  });
}

function renderGrowthChart(ctxId) {
  const ctx = document.getElementById(ctxId);
  if (!ctx) return null;

  // Simulate 10 years of compound growth at 24.7%
  const years = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const amounts = years.map(y => 10000 * Math.pow(1.247, y));

  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: years.map(y => `Año ${y}`),
      datasets: [{
        label: 'Crecimiento del Portafolio ($)',
        data: amounts.map(v => Math.round(v)),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { ticks: { color: '#94a3b8', callback: val => '$' + val.toLocaleString() } },
        x: { ticks: { color: '#94a3b8' } }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function initPreviewCharts() {
  const btnAlloc = document.getElementById("btn-preview-allocation");
  const btnGrowth = document.getElementById("btn-preview-growth");

  if (!btnAlloc || !btnGrowth) return;

  const labels = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"];
  const data = [35, 25, 20, 12, 8];
  const colors = ["#2563EB", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"];

  // Default display
  previewChartInstance = renderPieChart("preview-chart", data, labels, colors);

  btnAlloc.addEventListener("click", () => {
    btnAlloc.classList.add("active");
    btnGrowth.classList.remove("active");
    if (previewChartInstance) previewChartInstance.destroy();
    previewChartInstance = renderPieChart("preview-chart", data, labels, colors);
  });

  btnGrowth.addEventListener("click", () => {
    btnGrowth.classList.add("active");
    btnAlloc.classList.remove("active");
    if (previewChartInstance) previewChartInstance.destroy();
    previewChartInstance = renderGrowthChart("preview-chart");
  });
}

function renderDemoView() {
  const tableBody = document.getElementById("demo-table-body");
  if (!tableBody) return;
  tableBody.innerHTML = "";

  let totalAmount = 10000;

  DEMO_PORTFOLIO.forEach(item => {
    const allocAmount = (totalAmount * item.weight).toLocaleString("en-US", { minimumFractionDigits: 2 });
    tableBody.innerHTML += `
      <tr>
        <td><strong>${item.ticker}</strong></td>
        <td>${item.name}</td>
        <td><span style="color:${item.color}; font-weight:700;">${(item.weight * 100).toFixed(0)}%</span></td>
        <td style="font-family: var(--font-mono);">$${allocAmount} USD</td>
      </tr>
    `;
  });
}

function initDemoCharts() {
  const btnAlloc = document.getElementById("btn-demo-allocation");
  const btnGrowth = document.getElementById("btn-demo-growth");

  if (!btnAlloc || !btnGrowth) return;

  const labels = DEMO_PORTFOLIO.map(p => p.ticker);
  const data = DEMO_PORTFOLIO.map(p => p.weight * 100);
  const colors = DEMO_PORTFOLIO.map(p => p.color);

  // Defer initialization to when the canvas becomes visible to avoid measuring errors
  setTimeout(() => {
    demoChartInstance = renderPieChart("demo-chart", data, labels, colors);
  }, 100);

  btnAlloc.addEventListener("click", () => {
    btnAlloc.classList.add("active");
    btnGrowth.classList.remove("active");
    if (demoChartInstance) demoChartInstance.destroy();
    demoChartInstance = renderPieChart("demo-chart", data, labels, colors);
  });

  btnGrowth.addEventListener("click", () => {
    btnGrowth.classList.add("active");
    btnAlloc.classList.remove("active");
    if (demoChartInstance) demoChartInstance.destroy();
    demoChartInstance = renderGrowthChart("demo-chart");
  });
}

// Fetch base tickers from YAML via API (Always fetch details for sectors)
async function fetchTickers() {
  try {
    const res = await fetch(`${API_BASE}/tickers`);
    const data = await res.json();

    // Always update details for sector dropdown
    tickerDetails = data.details || {};

    if (allTickers.length === 0) {
      // First load: use API tickers and set defaults
      allTickers = data.tickers || [];
      allTickers.slice(0, 20).forEach(t => {
        favoriteTickers.add(t);
        selectedTickers.add(t);
      });
      saveState();
    }

    // Populate sector dropdown
    const sectors = new Set();
    Object.values(tickerDetails).forEach(d => {
      if (d.sector && d.sector !== "N/A") sectors.add(d.sector);
    });

    while (dirFilterSector.options.length > 1) {
      dirFilterSector.remove(1);
    }

    Array.from(sectors).sort().forEach(sec => {
      const opt = document.createElement("option");
      opt.value = sec;
      opt.textContent = sec;
      dirFilterSector.appendChild(opt);
    });
  } catch (err) {
    showError("No se pudieron cargar los tickers del servidor.");
  }
}

function renderTickerList() {
  tickerListEl.innerHTML = "";

  // Create the "Manage Favorites" entry button at the top
  const manageDiv = document.createElement("div");
  manageDiv.className = "ticker-item btn-manage";
  manageDiv.style.justifyContent = "center";
  manageDiv.style.color = "var(--accent-primary)";
  manageDiv.style.fontWeight = "600";
  manageDiv.style.gridColumn = "1 / -1";
  manageDiv.innerHTML = `<svg style="width:14px;height:14px;margin-right:4px;" viewBox="0 0 24 24"><path fill="currentColor" d="M12,17.27L18.18,21L16.54,13.97L22,9.24L14.81,8.62L12,2L9.19,8.62L2,9.24L7.45,13.97L5.82,21L12,17.27Z" /></svg> Administrar Directorio`;
  manageDiv.addEventListener("click", showDirectoryView);
  tickerListEl.appendChild(manageDiv);

  // Render ONLY favorites in the main sidebar
  const renderList = Array.from(favoriteTickers).sort();

  if (renderList.length === 0) {
    tickerListEl.insertAdjacentHTML('beforeend', `<div style="grid-column: 1/-1; text-align:center; padding:10px; color:var(--text-muted); font-size:0.8rem;">Aún no se han agregado favoritos.</div>`);
  }

  renderList.forEach(ticker => {
    const div = document.createElement("div");
    div.className = "ticker-item sidebar-ticker";
    const checked = selectedTickers.has(ticker) ? "checked" : "";
    const details = tickerDetails[ticker] || { name: "", sector: "" };

    const brand = BRAND_OVERRIDES[ticker];
    const displayName = brand ? brand.name : (details.name || ticker);

    let avatarHTML = '';
    if (brand) {
      avatarHTML = brand.logo
        ? `<img src="${escapeHTML(brand.logo)}" alt="${escapeHTML(displayName)}" class="ticker-logo" onerror="this.onerror=null;this.outerHTML='<div class=\\'ticker-avatar-text\\'>${escapeHTML(displayName.charAt(0))}</div>';">`
        : `<div class="ticker-avatar-text">${escapeHTML(displayName.charAt(0))}</div>`;
    }

    div.innerHTML = `
      <input type="checkbox" id="chk-${escapeHTML(ticker)}" value="${escapeHTML(ticker)}" ${checked}>
      ${avatarHTML}
      <div style="display: flex; flex-direction: column; flex-grow: 1; margin-left: 8px;">
        <label for="chk-${escapeHTML(ticker)}" style="margin: 0; cursor: pointer; ${brand ? 'font-weight: 500;' : ''}">${escapeHTML(displayName.length > 25 ? displayName.substring(0, 25) + "..." : displayName)}</label>
        ${!brand && details.name ? `<span style="font-size: 0.75rem; color: var(--text-muted); cursor: pointer;" onclick="document.getElementById('chk-${escapeHTML(ticker)}').click()">${escapeHTML(details.name.length > 25 ? details.name.substring(0, 25) + '...' : details.name)}</span>` : ''}
      </div>
    `;

    div.querySelector("input").addEventListener("change", (e) => {
      if (e.target.checked) selectedTickers.add(ticker);
      else selectedTickers.delete(ticker);
      saveState();
      updateTickerCount();
    });

    tickerListEl.appendChild(div);
  });
  updateTickerCount();
}

function updateTickerCount() {
  tickerCountEl.textContent = `${selectedTickers.size} seleccionados`;
}

// --- Dreams Test Handlers ---
async function handleDreamsTestSubmit(e) {
  e.preventDefault();

  const cost = parseFloat(dreamCost.value);
  const years = parseInt(dreamYears.value);
  const initial = parseFloat(dreamInitial.value);
  const monthly = parseFloat(dreamMonthly.value);

  if (isNaN(cost) || isNaN(years) || isNaN(initial) || isNaN(monthly) || cost <= 0 || years <= 0) {
    showDreamError("Por favor completa todos los campos con valores positivos.");
    return;
  }

  setDreamLoading(true);
  dreamError.classList.add("hidden");
  dreamResults.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/dreams_test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        meta_costo: cost,
        años: years,
        capital_inicial: initial,
        aporte_mensual: monthly
      })
    });

    if (!res.ok) throw new Error((await res.json()).detail || "Error calculando el test");

    const data = await res.json();

    if (data.status === "impossible") {
      showDreamError(data.message);
      return;
    }

    // Set Profile Data
    dreamProfileTitle.textContent = `Perfil Sugerido: ${data.perfil_sugerido}`;
    dreamProfileDesc.textContent = data.mensaje_perfil;
    dreamRate.textContent = `${data.tasa_objetivo_anual}%`;

    // Show results 
    dreamResults.classList.remove("hidden");

    // Pre-select strategy based on profile
    if (data.tasa_objetivo_anual < 10) {
      strategySelect.value = "hrp"; // Defensive
    } else {
      strategySelect.value = "markowitz"; // Aggressive
    }

  } catch (err) {
    showDreamError(err.message);
  } finally {
    setDreamLoading(false);
  }
}

function showDreamError(msg) {
  dreamError.textContent = msg;
  dreamError.classList.remove("hidden");
}

function setDreamLoading(isLoading) {
  if (isLoading) {
    dreamBtnText.classList.add("hidden");
    dreamSpinner.classList.remove("hidden");
  } else {
    dreamBtnText.classList.remove("hidden");
    dreamSpinner.classList.add("hidden");
  }
}

function handleAddTicker() {
  const val = newTickerInput.value.trim().toUpperCase();
  if (val && !allTickers.includes(val)) {
    allTickers.push(val);
    if (favoriteTickers.size < 120) {
      favoriteTickers.add(val); // Auto-favorite new manual entries if under limit
      selectedTickers.add(val);
    }
    saveState();
    newTickerInput.value = "";
    renderTickerList();
  } else if (allTickers.includes(val)) {
    if (favoriteTickers.size < 120) {
      favoriteTickers.add(val);
      selectedTickers.add(val);
    } else {
      showError("Se ha alcanzado el límite de 120 favoritos. Elimina algunos para añadir más.");
    }
    saveState();
    newTickerInput.value = "";
    renderTickerList();
  }
}

// --- Directory Controls ---
function showDirectoryView() {
  navigateTo("directory");
}

function renderDirectoryList(query = "") {
  dirTickerList.innerHTML = "";

  let filtered = allTickers.filter(t => t.includes(query)).sort();

  if (showOnlyFavorites) {
    filtered = filtered.filter(t => favoriteTickers.has(t));
  }

  const marketVal = dirFilterMarket.value;
  const sectorVal = dirFilterSector.value;

  if (marketVal !== "ALL") {
    filtered = filtered.filter(t => {
      const d = tickerDetails[t];
      if (d && d.market) return d.market === marketVal;
      // Fallback inference if missing
      if (marketVal === "MX") return t.endsWith(".MX");
      if (marketVal === "US") return !t.endsWith(".MX");
      return true;
    });
  }

  if (sectorVal !== "ALL") {
    filtered = filtered.filter(t => {
      const d = tickerDetails[t];
      return d && d.sector === sectorVal;
    });
  }

  filtered.forEach(ticker => {
    const isFav = favoriteTickers.has(ticker);
    const isSelected = selectedTickers.has(ticker);
    const div = document.createElement("div");
    const details = tickerDetails[ticker] || { name: "", sector: "" };

    // Add visual cue if it's currently selected in the sidebar
    div.className = `global-ticker-item ${isFav ? 'is-favorite' : ''} ${isSelected ? 'is-selected' : ''}`;
    div.style.border = isSelected ? '1px solid var(--accent-primary)' : '';

    div.innerHTML = `
      <div style="flex-grow: 1; display: flex; flex-direction: column;">
        <span style="font-size: 1.1rem; font-weight: 600;">${escapeHTML(ticker)}</span>
        ${details.name ? `<span style="font-size: 0.80rem; color: var(--text-muted); margin-top: 2px;">${escapeHTML(details.name)} &bull; ${escapeHTML(details.sector)}</span>` : ''}
      </div>
      <button class="btn-star ${isFav ? 'active' : ''}">★</button>
    `;

    div.querySelector(".btn-star").addEventListener("click", () => toggleFavorite(ticker));
    dirTickerList.appendChild(div);
  });
}

async function handleDirAddTicker() {
  const val = dirNewTicker.value.trim().toUpperCase();
  if (!val) return;

  // If already in list, just add to favorites
  if (allTickers.includes(val)) {
    if (favoriteTickers.size < 120) {
      favoriteTickers.add(val);
      selectedTickers.add(val);
    } else {
      alert("Se ha alcanzado el límite de 120 favoritos. Elimina algunos para añadir más.");
    }
    saveState();
    dirNewTicker.value = "";
    renderDirectoryList(dirSearch.value.toUpperCase());
    return;
  }

  // Show validation modal
  const modal = document.getElementById("ticker-validate-modal");
  const spinner = document.getElementById("ticker-validate-spinner");
  const result = document.getElementById("ticker-validate-result");
  const errorDiv = document.getElementById("ticker-validate-error");

  modal.classList.remove("hidden");
  spinner.classList.remove("hidden");
  result.classList.add("hidden");
  errorDiv.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/validate_ticker/${encodeURIComponent(val)}`);
    spinner.classList.add("hidden");

    if (!res.ok) {
      const err = await res.json();
      document.getElementById("tv-error-msg").textContent = err.detail || "Ticker no encontrado.";
      errorDiv.classList.remove("hidden");
      return;
    }

    const data = await res.json();
    document.getElementById("tv-ticker").textContent = data.ticker;
    document.getElementById("tv-name").textContent = data.name || "Nombre no disponible";
    document.getElementById("tv-sector").textContent = data.sector || "N/A";
    document.getElementById("tv-market").textContent = data.market === "MX" ? "🇲🇽 México" : "🇺🇸 EEUU";
    result.classList.remove("hidden");

    // Store data for confirm
    modal._pendingTicker = data;

  } catch (e) {
    spinner.classList.add("hidden");
    document.getElementById("tv-error-msg").textContent = "Error de conexión al validar el ticker.";
    errorDiv.classList.remove("hidden");
  }
}

function toggleFavorite(ticker) {
  if (favoriteTickers.has(ticker)) {
    favoriteTickers.delete(ticker);
    selectedTickers.delete(ticker); // Auto un-select if unfavorited 
  } else {
    if (favoriteTickers.size >= 120) {
      alert("Se permite un máximo de 120 favoritos.");
      return;
    }
    favoriteTickers.add(ticker);
  }
  saveState();
  renderDirectoryList(dirSearch.value.toUpperCase());
}

let errorTimeoutId;

function showError(msg, persist = false) {
  clearTimeout(errorTimeoutId);
  errorToast.innerHTML = `
    <span>${escapeHTML(msg)}</span>
    <button onclick="document.getElementById('error-message').classList.add('hidden')" 
            style="background:none;border:none;color:inherit;font-size:1.2rem;cursor:pointer;line-height:1;margin-left:12px;">
      &times;
    </button>
  `;
  errorToast.classList.remove("hidden");

  if (!persist) {
    errorTimeoutId = setTimeout(() => errorToast.classList.add("hidden"), 8000);
  }
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
    showError("Por favor, selecciona al menos 2 tickers para optimizar.");
    return;
  }

  // Phase 8 Reliability Guard
  if (selectedTickers.size > 120) {
    showError("Error: Límite de 120 tickers excedido. Optimizar con demasiados activos causa alta distorsión matemática. Por favor, deselecciona algunos activos.");
    return;
  }

  const budget = parseFloat(budgetInput.value);
  if (isNaN(budget) || budget < 100) {
    showError("Por favor ingresa un presupuesto válido (mínimo $100).");
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

    if (!resOpt.ok) throw new Error((await resOpt.json()).detail || "Fallo en la optimización");
    const dataOpt = await resOpt.json();
    latestOptimizationData = dataOpt; // CACHE IT

    renderDashboard(latestOptimizationData);

    // Flag to allow export
    window.HAS_OPTIMIZED = true;

    // Update data freshness indicator
    const now = new Date();
    dataFreshnessText.textContent = `📡 Datos al: ${now.toLocaleDateString('es-MX')} ${now.toLocaleTimeString('es-MX', { hour: '2-digit', minute: '2-digit' })}`;

    // Refresh social proof counter (it just incremented on backend)
    fetchStats();

  } catch (err) {
    showError(err.message, true); // Persist optimization error so it can be read
  } finally {
    setLoading(false);
  }
}

function renderDashboard(data) {
  const strategy = strategySelect.value;
  let metricsData, allocationData, trackData, budgetData, riskData, projectionData, divData;

  if (strategy === "markowitz") {
    metricsData = data.metrics;
    allocationData = data.allocation;
    trackData = data.track_data;
    budgetData = data.budget_analysis;
    riskData = data.risk_analysis;
    projectionData = data.projections;
    divData = data.dividends;
  } else {
    // HRP
    metricsData = data.hrp_metrics;
    allocationData = data.hrp_metrics.hrp_allocation;
    trackData = data.hrp_metrics.hrp_track_data;
    budgetData = data.hrp_metrics.hrp_budget_analysis;
    riskData = data.hrp_metrics.risk_analysis;
    projectionData = data.hrp_metrics.projections;
    divData = data.hrp_metrics.dividends;
  }

  // Render sub components
  renderResults(metricsData, allocationData, budgetData, riskData, divData);
  renderDoughnutChart(allocationData);
  if (projectionData) {
    renderProjectionChart(projectionData);
  }
}


function formatMoney(amount) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 0 }).format(amount);
}

function renderResults(metrics, allocation, budget_analysis, risk_analysis, div_data) {
  // Top Metrics (Advanced)
  metReturn.textContent = `${metrics.expected_return_annual_pct}%`;
  metVol.textContent = `${metrics.volatility_annual_pct}%`;
  metSharpe.textContent = metrics.sharpe_ratio;
  metRemCash.textContent = `$${budget_analysis.remaining_cash}`;

  // Top Metrics (Dreams/Beginner)
  if (risk_analysis) {
    dreamMetricGain.textContent = `+${formatMoney(risk_analysis.ganancia_esperada)}`;
    dreamMetricLoss.textContent = `${formatMoney(risk_analysis.perdida_peor_escenario)}`;
  }

  // Dividends Metrics
  if (div_data) {
    dreamMetricDivs.textContent = `+${formatMoney(div_data.trimestral_estimado)}`;
  }

  // Toggle correct view based on current mode
  if (btnModeDreams.classList.contains("active")) {
    advancedMetrics.classList.add("hidden");
    dreamsMetrics.classList.remove("hidden");
    dreamsProjections.classList.remove("hidden");
    advTableSection.classList.add("hidden");
    dreamsRecipeSection.classList.remove("hidden");
  } else {
    advancedMetrics.classList.remove("hidden");
    dreamsMetrics.classList.add("hidden");
    dreamsProjections.classList.add("hidden");
    advTableSection.classList.remove("hidden");
    dreamsRecipeSection.classList.add("hidden");
  }

  // Table
  allocBody.innerHTML = "";
  if (allocation.length === 0) {
    allocBody.innerHTML = `<tr><td colspan="5" class="empty-state">No se asignaron acciones (presupuesto muy bajo)</td></tr>`;
    recipeList.innerHTML = `<li class="empty-state">No se asignaron acciones (presupuesto muy bajo)</li>`;
    return;
  }

  let totalAllocated = 0;
  recipeList.innerHTML = "";

  allocation.forEach(row => {
    totalAllocated += row.dollar_allocation || 0;

    // Advanced Table Row
    const tr = document.createElement("tr");
    const shares_to_buy = row.shares_to_buy !== undefined ? row.shares_to_buy : "N/A (Límite de API)";
    const dollar_allocation = row.dollar_allocation !== undefined ? formatMoney(row.dollar_allocation) : "N/A";
    const last_price = row.last_price !== undefined ? formatMoney(row.last_price) : "N/A";

    tr.innerHTML = `
      <td style="font-weight:600;">${escapeHTML(row.ticker)}</td>
      <td>${escapeHTML(row.weight_pct)}%</td>
      <td>${escapeHTML(last_price)}</td>
      <td style="color:#3b82f6; font-weight:bold">${escapeHTML(shares_to_buy)}</td>
      <td>${escapeHTML(dollar_allocation)}</td>
    `;
    allocBody.appendChild(tr);

    // Beginner Recipe Row
    if (row.shares_to_buy > 0) {
      const brand = BRAND_OVERRIDES[row.ticker] || null;
      const details = tickerDetails[row.ticker] || {};
      const displayName = brand ? brand.name : (details.name || row.ticker);

      let avatarHTML = '';
      if (brand && brand.logo) {
        avatarHTML = `<img src="${escapeHTML(brand.logo)}" style="width: 28px; height: 28px; border-radius: 50%; background: white;">`;
      } else {
        avatarHTML = `<div class="ticker-avatar-text" style="width: 28px; height: 28px; font-size: 0.9rem;">${escapeHTML(displayName.charAt(0))}</div>`;
      }

      const li = document.createElement("li");
      li.style.display = "flex";
      li.style.alignItems = "center";
      li.style.gap = "12px";
      li.style.padding = "12px";
      li.style.background = "rgba(15, 23, 42, 0.4)";
      li.style.borderRadius = "8px";
      li.style.border = "1px solid var(--border-light)";

      li.innerHTML = `
        <div style="color: var(--success); font-size: 1.2rem;">✓</div>
        ${avatarHTML}
        <div style="flex-grow: 1;">
          <span style="font-size: 1.05rem;">Compra <strong>${escapeHTML(row.shares_to_buy)} acciones</strong> de <strong>${escapeHTML(displayName)}</strong></span>
        </div>
        <div style="color: var(--text-muted); font-size: 0.85rem; text-align: right;">
          Destinando<br>~${escapeHTML(dollar_allocation)}
        </div>
      `;
      recipeList.appendChild(li);
    }
  });

  if (budget_analysis && budget_analysis.remaining_cash !== undefined) {
    recipeRemCash.textContent = formatMoney(budget_analysis.remaining_cash);
  }
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
        legend: { position: 'bottom', labels: { boxWidth: 12, padding: 15 } }
      }
    }
  });
}


function renderProjectionChart(projData) {
  const ctx = document.getElementById("projection-chart").getContext("2d");
  if (chartInstances.projection) chartInstances.projection.destroy();

  chartInstances.projection = new Chart(ctx, {
    type: 'line',
    data: {
      labels: projData.anios,
      datasets: [
        {
          label: 'Portafolio Invertido',
          data: projData.portafolio,
          borderColor: '#10b981', // Success/Green
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: '#10b981',
          fill: true,
          tension: 0.3
        },
        {
          label: 'CETES (10% aprox.)',
          data: projData.cetes,
          borderColor: '#94a3b8', // Gray
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: 'Cuenta Corriente (Inflación)',
          data: projData.inflacion,
          borderColor: '#ef4444', // Danger/Red
          borderWidth: 2,
          borderDash: [2, 4],
          pointRadius: 0,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        tooltip: {
          callbacks: {
            label: function (context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              if (context.parsed.y !== null) {
                label += formatMoney(context.parsed.y);
              }
              return label;
            }
          }
        }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' } },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: {
            callback: function (value) {
              return '$' + value;
            }
          }
        }
      }
    }
  });
}

async function handleExportPdf() {
  console.log("Export triggered!");

  if (!window.HAS_OPTIMIZED) {
    showError("Por favor ejecuta 'Optimizar Portafolio' para generar datos antes de exportar.");
    return;
  }

  if (typeof window.html2pdf === 'undefined') {
    showError("El renderizador de PDF no cargó. Por favor verifica tu conexión a internet.");
    console.error("html2pdf is not defined in window object.");
    return;
  }

  btnExportPdf.disabled = true;
  const originalText = btnExportPdf.innerHTML;
  btnExportPdf.innerText = "Exportando...";

  try {
    // 1. Aplicar clases de exportación y asegurar el scroll al tope
    window.scrollTo(0, 0);
    document.body.classList.add("pdf-exporting-body");
    const element = document.getElementById("printable-dashboard");
    element.classList.add("pdf-exporting");

    // 2. CRUCIAL: Esperar a que el navegador termine de aplicar los estilos y gráficos
    await new Promise(resolve => setTimeout(resolve, 800));

    // Forzar redibujado de gráficos
    if (chartInstances.doughnut) chartInstances.doughnut.resize();
    if (chartInstances.line) chartInstances.line.resize();

    await new Promise(resolve => setTimeout(resolve, 200));

    // 3. Calcular la altura real del contenido después del ajuste
    const contentHeight = Math.max(element.scrollHeight, 1800);

    // 4. Configurar opciones de PDF
    const opt = {
      margin: [10, 10, 10, 10],
      filename: 'Reporte_Portafolio.pdf',
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: {
        scale: 2,
        useCORS: true,
        logging: false,
        windowWidth: 1200,
        windowHeight: contentHeight,
        scrollY: 0,
        scrollX: 0,
        x: 0,
        y: 0,
        width: 1200,
        height: contentHeight
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
    showError("Ocurrió un error al generar el PDF. Revisa la consola.");
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
