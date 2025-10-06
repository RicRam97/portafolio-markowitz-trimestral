# main.py — Pipeline trimestral end-to-end (ingesta → BL → optimización → reporte)
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from datetime import datetime
import io, base64, os
import matplotlib.pyplot as plt

TICKERS = [
    'AAL','AAPL','ABNB','ADBE','AC.MX','ALSEA.MX','AMD','AMZN','AXP','B','BA','BABA','BAC','BIMBOA.MX','BWXT',
    'CAT','CB','CCL','CEG','CELH','CEMEXCPO.MX','CHWY','CMG','COIN','COPX','CRM','CRWD','CSCO','CVX','DAL','DDOG','DG','DIS',
    'DKNG','DLTR','ELF','FDX','FEMSAUBD.MX','GDX','GLD','GOOG','HD','HIMS','HON','HP','INTC','IWM','JNJ','JPM','KIMBERA.MX',
    'KMI','KO','KR','LABB.MX','LCID','LLY','LMND','LULU','LUV','M','MA','MARA','MAT','MCD','MEGACPO.MX','META','MMM','MRK','MS',
    'MSFT','MSTR','NCLH','NEE','NEMAKA.MX','NFLX','NIO','NKE','NVDA','NVO','OXY','PANW','PENN','PEP','PG','PGR','PINS','PTON',
    'PYPL','QQQ','RBLX','RCL','RIVN','RUN','SBUX','SCCO','SHOP','SLB','SLV','SNOW','SOFI','SOXX','SPOT','SPY','TDOC','TGT',
    'TLT','TMO','TRV','TSLA','TSM','TWLO','U','UAL','UBER','UL','UNH','UNP','UPST','V','VESTA.MX','VOLARA.MX','VOO','VZ',
    'WALMEX.MX','WBA','WFC','WMT','XLE','XOM','ZM'
]

def descargar_lotes(tickers, start, end, interval="1d", lote=35):
    grupos = [tickers[i:i+lote] for i in range(0, len(tickers), lote)]
    frames, fallidos = [], []
    for g in grupos:
        df = yf.download(g, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by='ticker')
        if df.empty: fallidos.extend(g); continue
        frames.append(df)
    if not frames: return pd.DataFrame(), tickers
    return pd.concat(frames, axis=1).sort_index(), fallidos

def extraer_panel(raw, tickers, field="Adj Close"):
    if raw.empty: return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        t = tickers[0]
        if field in raw.columns: return raw[[field]].rename(columns={field: t})
        col0 = raw.columns[0]; return raw[[col0]].rename(columns={col0: t})
    cols=[]
    for t in tickers:
        if (t, field) in raw.columns: cols.append(raw[(t, field)].rename(t))
        elif (t, "Close") in raw.columns: cols.append(raw[(t, "Close")].rename(t))
    if not cols: return pd.DataFrame()
    panel = pd.concat(cols, axis=1).sort_index()
    return panel.tz_localize(None)

def generar_reporte_html(ef, w_best, mu_bl, cov_bl, quality, filename):
    assets = list(mu_bl.index)
    Sigma = cov_bl.reindex(index=assets, columns=assets).values
    mu = mu_bl.reindex(assets).values
    w = w_best.reindex(assets).fillna(0).values
    rf = 0.0
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w, Sigma @ w)))
    sh = (ret - rf) / (vol + 1e-12)
    n = len(assets)
    w_eq = np.repeat(1/n, n)
    ret_eq = float(np.dot(w_eq, mu))
    vol_eq = float(np.sqrt(np.dot(w_eq, Sigma @ w_eq)))
    sh_eq = (ret_eq - rf) / (vol_eq + 1e-12)
    if 'SPY' in assets:
        spy = np.zeros(n); spy[assets.index('SPY')] = 1.0
        ret_spy = float(np.dot(spy, mu))
        vol_spy = float(np.sqrt(np.dot(spy, Sigma @ spy)))
        sh_spy = (ret_spy - rf) / (vol_spy + 1e-12)
    else:
        ret_spy=vol_spy=sh_spy=np.nan

    import matplotlib.pyplot as plt, io, base64
    plt.figure(figsize=(6,4))
    sc = plt.scatter(ef['vol'], ef['ret'], c=ef['sharpe'], cmap='viridis')
    plt.colorbar(sc, label='Sharpe')
    plt.xlabel('Volatilidad anual'); plt.ylabel('Retorno anual'); plt.title('Frontera Eficiente')
    plt.scatter([vol],[ret], color='red', marker='x', s=120, label='Mejor Sharpe'); plt.legend(); plt.grid(alpha=0.3)
    buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png', dpi=120); plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    Sigma_np = cov_bl.reindex(index=assets, columns=assets).values
    marginal = Sigma_np @ w
    rc = (w * marginal) / (vol + 1e-12)
    rc_series = pd.Series(rc, index=assets).sort_values(ascending=False)
    top_w = w_best.head(15).round(4)
    top_rc = rc_series.head(15).round(4)

    today = datetime.utcnow().strftime('%Y-%m-%d')
    html = f"""
    <html><head><meta charset='utf-8'/>
    <style>body{{font-family:Arial;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:6px;text-align:right}}th{{background:#f2f2f2}}.left{{text-align:left}}</style>
    <title>Reporte Trimestral - {today}</title></head><body>
    <h1>Reporte Trimestral</h1><div>Generado: {today}</div>
    <h2>Resumen</h2>
    <p><b>Retorno:</b> {ret:.2%} | <b>Vol:</b> {vol:.2%} | <b>Sharpe:</b> {sh:.3f}</p>
    <p>Equiponderado Sharpe {sh_eq:.3f} | SPY Sharpe {sh_spy:.3f}</p>
    <img src="data:image/png;base64,{img_b64}" style="max-width:640px"/>
    <h2>Top 15 Pesos</h2>
    <table><tr><th class='left'>Ticker</th><th>Peso</th></tr>
    {''.join(f"<tr><td class='left'>{k}</td><td>{v:.4f}</td></tr>" for k,v in top_w.items())}
    </table>
    <h2>Top 15 Contribuciones a Riesgo</h2>
    <table><tr><th class='left'>Ticker</th><th>RC</th></tr>
    {''.join(f"<tr><td class='left'>{k}</td><td>{v:.4f}</td></tr>" for k,v in top_rc.items())}
    </table>
    <h2>Calidad de datos — peores 10 coberturas</h2>
    <table><tr><th class='left'>Ticker</th><th>Coverage %</th><th>Gaps</th><th class='left'>Start</th><th class='left'>End</th></tr>
    {''.join(f"<tr><td class='left'>{idx}</td><td>{row['coverage_%']:.2f}</td><td>{int(row['gaps'])}</td><td class='left'>{row['start']}</td><td class='left'>{row['end']}</td></tr>" for idx,row in quality.sort_values('coverage_%').head(10).iterrows())}
    </table>
    <div style='font-size:12px;color:#666;margin-top:12px'>Metodología: Ledoit–Wolf + Black–Litterman (neutral), SLSQP con límites 0–15% por activo.</div>
    </body></html>
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print("Reporte generado:", filename)

def pipeline_trimestral():
    start = (pd.Timestamp.today(tz="UTC") - pd.DateOffset(years=3)).date()
    end = pd.Timestamp.today(tz="UTC").date()

    # Ingesta
    raw, _ = descargar_lotes(TICKERS, start, end)
    prices = extraer_panel(raw, TICKERS, "Adj Close")
    idx_master = pd.date_range(prices.index.min(), prices.index.max(), freq="B")
    prices = prices.reindex(idx_master).ffill(limit=3)

    # Limpieza y outliers
    returns_raw = prices.pct_change(fill_method=None)
    sigma6 = 6 * returns_raw.std(skipna=True)
    returns = returns_raw.clip(lower=-sigma6, upper=sigma6, axis=1)

    # Cobertura
    coverage = returns.notna().mean()
    keep = coverage[coverage >= 0.98].index
    rets_clean = returns[keep].dropna(how='any')

    # Estimación robusta
    lw = LedoitWolf().fit(rets_clean.values)
    cov_annual = pd.DataFrame(lw.covariance_, index=rets_clean.columns, columns=rets_clean.columns) * 252
    mu_annual = rets_clean.mean() * 252
    mu_shrunk = (1-0.25)*mu_annual

    # BL neutral
    mu_bl = mu_shrunk.copy()
    Sigma = cov_annual

    # Optimización max Sharpe
    assets = list(mu_bl.index); n = len(assets)
    Sigma_np = Sigma.values; mu_np = mu_bl.values
    bounds = [(0.0, 0.15)] * n
    cons = [{'type':'eq','fun': lambda w: np.sum(w)-1.0}]
    rf = 0.0; lambda_l2 = 1e-5

    def stats(w):
        r = float(np.dot(w, mu_np))
        v = float(np.sqrt(np.dot(w, Sigma_np @ w)))
        return r, v
    def obj(w):
        r, v = stats(w)
        return - (r-rf)/(v+1e-12) + lambda_l2*np.dot(w,w)

    w0 = np.repeat(1/n, n)
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
    w_best = pd.Series(res.x, index=assets).sort_values(ascending=False)

    # Frontera eficiente
    mu_lo, mu_hi = np.percentile(mu_np, 20), np.percentile(mu_np, 80)
    targets = np.linspace(mu_lo, mu_hi, 25)
    rows = []; weights = []
    def obj_minvar(w): return float(np.dot(w, Sigma_np @ w) + lambda_l2*np.dot(w,w))
    for t in targets:
        cons_t = cons + [{'type':'ineq','fun': lambda w, tt=t: np.dot(w, mu_np) - tt}]
        resf = minimize(obj_minvar, w0, method='SLSQP', bounds=bounds, constraints=cons_t, options={'maxiter': 1000})
        if resf.success:
            r,v = stats(resf.x)
            rows.append([t, r, v, (r-rf)/(v+1e-12)])
            weights.append(resf.x)
    ef = pd.DataFrame(rows, columns=['target','ret','vol','sharpe'])

    # Quality report
    def reporte_calidad(prices, rets):
        coverage = prices.notna().sum() / len(prices)
        start_dates = prices.apply(lambda s: s.first_valid_index())
        end_dates = prices.apply(lambda s: s.last_valid_index())
        gaps = prices.isna().sum()
        return pd.DataFrame({
            'coverage_%': (coverage*100).round(2),
            'gaps': gaps,
            'start': start_dates,
            'end': end_dates,
            'mean_ret_daily_%': (rets.mean()*100).round(3),
            'vol_daily_%': (rets.std()*100).round(3)
        }).sort_values('coverage_%')

    quality = reporte_calidad(prices, rets_clean)

    # Guardados útiles (opcional)
    os.makedirs("reports", exist_ok=True)
    ef.to_csv('efficient_frontier.csv', index=False)
    w_best.to_csv('weights_frontier_best.csv', header=['weight'])
    mu_bl.to_csv('mu_black_litterman.csv', header=['mu_bl'])
    Sigma.to_csv('cov_black_litterman.csv')
    quality.to_csv('quality_report.csv')

    # Reporte
    report_name = f"reports/reporte_trimestral_{datetime.utcnow().strftime('%Y_%m_%d')}.html"
    generar_reporte_html(ef, w_best, mu_bl, Sigma, quality, report_name)
    return report_name

if __name__ == "__main__":
    path = pipeline_trimestral()
    print("Reporte listo en:", path)
