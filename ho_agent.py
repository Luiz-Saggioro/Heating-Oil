#!/usr/bin/env python3
"""
Heating Oil Probability Engine -- Deterministic, no LLM.
Fetches live HO/WTI/Brent/RBOB/DXY/VIX, runs a 3-model ensemble,
generates probability tables, custom bands, scenario weights, and charts.
Called by app.py via run().
"""

import os, io, json, uuid, base64, datetime, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import data_fetcher as _df
import requests

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

HORIZONS     = ["1M", "3M", "6M", "9M", "12M"]
HORIZON_DAYS = {"1M": 21, "3M": 63, "6M": 126, "9M": 189, "12M": 252}

HO_PRICE_BINS = [
    "<1.80", "1.80-2.20", "2.20-2.60", "2.60-3.00",
    "3.00-3.30", "3.30-3.69", "3.69-4.10", "4.10-4.60",
    "4.60-5.10", "5.10-5.70", "5.70-6.25", ">6.25",
]
BIN_EDGES = [-np.inf, 1.80, 2.20, 2.60, 3.00,
              3.30, 3.69, 4.10, 4.60, 5.10, 5.70, 6.25, np.inf]

CUSTOM_BANDS = [
    ("P(3.30-3.69)",  3.30,      3.69     ),
    ("P(>5.10)",      5.10,      np.inf   ),
    ("P(>6.25)",      6.25,      np.inf   ),
    ("P(<3.00)",     -np.inf,    3.00     ),
    ("P(<1.80)",     -np.inf,    1.80     ),
]

EIA_API_KEY = "DEMO_KEY"


# -- PROBABILITY MODELS --------------------------------------------------------

def lognormal_probs(current, returns, horizon_days):
    """
    Log-normal probability for heating oil prices.
    mu=0: HO is a commodity. Forward price = spot (no drift assumption).
    Only sigma drives the distribution. Annualised vol capped at 80%.
    """
    from scipy.stats import norm
    r   = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.015, 60)
    sig_daily = float(np.std(r, ddof=1)) or 0.015
    sig_daily = min(sig_daily, 0.80 / np.sqrt(252))
    log_mu  = np.log(current) - 0.5 * sig_daily**2 * horizon_days
    log_sig = sig_daily * np.sqrt(horizon_days)
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        p_lo = norm.cdf(np.log(max(lo,1e-6)), loc=log_mu, scale=log_sig) if lo > -np.inf else 0.0
        p_hi = norm.cdf(np.log(hi),           loc=log_mu, scale=log_sig) if hi <  np.inf else 1.0
        probs.append(max(0.0, p_hi - p_lo))
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def bootstrap_probs(current, returns, horizon_days, n=3000):
    """Demeaned historical bootstrap — keeps vol structure, removes trend bias."""
    r = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.015, 60)
    r = r - np.mean(r)  # remove historical trend
    sims  = np.exp(np.sum(np.random.choice(r, size=(n, horizon_days), replace=True), axis=1))
    final = current * sims
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        mask = ((final >= lo) if lo > -np.inf else np.ones(n, bool)) & \
               ((final <  hi) if hi <  np.inf else np.ones(n, bool))
        probs.append(float(mask.sum())/n)
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def mean_reversion_probs(current, returns, horizon_days, long_run_mean=None):
    """
    Log-OU (Ornstein-Uhlenbeck in log-price space) for heating oil.

    Corrected implementation:
    - kappa = 0.30 per YEAR (half-life = ln(2)/0.30 = 2.3 years).
      Oil markets empirically mean-revert in 1-3 years.
    - T is expressed in YEARS (horizon_days / 252).
    - exp_kT = exp(-kappa * T)  [not * 252 — that was a double-counting bug]
    - Variance uses annualised sigma, consistent with kappa units.
    - long_run_mean is the 90-day average price, anchoring the reversion target.
    """
    from scipy.stats import norm
    r   = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.015, 60)
    sig_daily  = float(np.std(r, ddof=1)) or 0.015
    sig_daily  = min(sig_daily, 0.80 / np.sqrt(252))
    sig_annual = sig_daily * np.sqrt(252)
    if long_run_mean is None:
        long_run_mean = current
    kappa = 0.30           # per year (half-life ~2.3 years)
    T     = horizon_days / 252.0   # in years
    exp_kT    = np.exp(-kappa * T)
    fwd_mean  = np.log(long_run_mean) + (np.log(current) - np.log(long_run_mean)) * exp_kT
    fwd_var   = (sig_annual**2 / (2.0 * kappa)) * (1.0 - exp_kT**2)
    fwd_sig   = np.sqrt(max(fwd_var, 1e-8))
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        p_lo = norm.cdf(np.log(max(lo,1e-6)), loc=fwd_mean, scale=fwd_sig) if lo > -np.inf else 0.0
        p_hi = norm.cdf(np.log(hi),           loc=fwd_mean, scale=fwd_sig) if hi <  np.inf else 1.0
        probs.append(max(0.0, p_hi - p_lo))
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def ensemble3(p1, p2, p3, w=(0.40, 0.35, 0.25)):
    combined = [w[0]*a + w[1]*b + w[2]*c for a,b,c in zip(p1,p2,p3)]
    t = sum(combined) or 1.0
    return [p/t for p in combined]


def band_prob(ens_probs, lo, hi):
    """Compute P(lo <= price < hi) from ensemble bin probabilities."""
    prob = 0.0
    for i in range(len(BIN_EDGES)-1):
        b_lo, b_hi = BIN_EDGES[i], BIN_EDGES[i+1]
        overlap_lo = max(b_lo, lo)
        overlap_hi = min(b_hi, hi)
        if overlap_hi > overlap_lo:
            prob += ens_probs[i]
    return round(prob, 4)


# -- DATA FETCHING -------------------------------------------------------------

def fetch_eia_distillate(send=print):
    send("Fetching EIA distillate stocks ...")
    url = (
        "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        "?api_key={}&frequency=weekly&data[0]=value"
        "&facets[product][]=DFO&facets[duoarea][]=NUS"
        "&sort[0][column]=period&sort[0][direction]=desc&length=8".format(EIA_API_KEY)
    )
    try:
        r = requests.get(url, timeout=15)
        rows = r.json().get("response", {}).get("data", [])
        if rows:
            latest = float(rows[0]["value"])
            prev   = float(rows[1]["value"]) if len(rows) > 1 else latest
            send("  EIA distillate: {:,.0f} Mbbl  WoW: {:+.0f}".format(latest, latest-prev))
            return {"stocks_mbbl": latest, "wow_change": latest-prev,
                    "weeks": [{"period": r["period"], "value": float(r["value"])} for r in rows]}
    except Exception as e:
        send("  [WARN] EIA API: {}".format(e))
    return {"stocks_mbbl": None, "wow_change": None, "weeks": []}


# -- REGIME DETECTION ----------------------------------------------------------

def detect_regime(ho, wti, vix, crack_spread, eia_data):
    """
    Rule-based regime label from market indicators.
    Returns (label, scenario_weights dict).
    """
    # Base weights
    weights = {
        "de-escalation": 0.15,
        "status_quo":    0.40,
        "escalation":    0.20,
        "recession":     0.15,
        "phys_squeeze":  0.10,
    }

    # VIX signal
    if vix and vix > 30:
        weights["escalation"]    += 0.08
        weights["recession"]     += 0.05
        weights["status_quo"]    -= 0.08
        weights["de-escalation"] -= 0.05
    elif vix and vix < 15:
        weights["de-escalation"] += 0.05
        weights["status_quo"]    += 0.05
        weights["escalation"]    -= 0.10

    # Crack spread signal (high crack = tight refinery supply)
    if crack_spread and crack_spread > 30:
        weights["phys_squeeze"]  += 0.08
        weights["escalation"]    += 0.05
        weights["de-escalation"] -= 0.08
        weights["status_quo"]    -= 0.05
    elif crack_spread and crack_spread < 10:
        weights["de-escalation"] += 0.05
        weights["status_quo"]    += 0.05
        weights["phys_squeeze"]  -= 0.07
        weights["escalation"]    -= 0.03

    # EIA inventory signal
    wow = eia_data.get("wow_change")
    if wow and wow < -3000:   # large draw
        weights["phys_squeeze"]  += 0.05
        weights["escalation"]    += 0.03
        weights["de-escalation"] -= 0.05
        weights["status_quo"]    -= 0.03
    elif wow and wow > 3000:  # large build
        weights["de-escalation"] += 0.05
        weights["recession"]     += 0.03
        weights["phys_squeeze"]  -= 0.05
        weights["escalation"]    -= 0.03

    # Normalise to 1.0
    total = sum(weights.values())
    weights = {k: round(max(0.01, v/total), 4) for k, v in weights.items()}
    # Re-normalise after clipping
    total = sum(weights.values())
    weights = {k: round(v/total, 4) for k, v in weights.items()}

    dominant = max(weights, key=weights.get)
    regime_labels = {
        "de-escalation": "EASING",
        "status_quo":    "STABLE",
        "escalation":    "TIGHTENING",
        "recession":     "RISK-OFF",
        "phys_squeeze":  "SUPPLY SQUEEZE",
    }
    return regime_labels.get(dominant, "STABLE"), weights


# -- RULE-BASED MARKET SUMMARY -------------------------------------------------

def market_summary(ho, wti, brent, vix, dxy, crack, eia_data, regime, weights):
    ho_hist = []  # not used in summary text, kept for future
    vix_r   = "HIGH" if vix and vix > 30 else "ELEVATED" if vix and vix > 20 else "NORMAL"
    eia_s   = eia_data.get("stocks_mbbl")
    wow     = eia_data.get("wow_change")
    lines   = [
        "=== HEATING OIL PROBABILITY ENGINE -- {} ===".format(datetime.date.today()), "",
        "MARKET SNAPSHOT",
        "  HO ($/gal)  : ${:.4f}".format(ho),
        "  WTI ($/bbl) : ${:.2f}".format(wti or 0),
        "  Brent($/bbl): ${:.2f}".format(brent or 0),
        "  RBOB ($/gal): see dashboard",
        "  DXY          : {:.2f}".format(dxy or 0),
        "  VIX          : {:.1f}  [{}]".format(vix or 0, vix_r),
        "  HO Crack     : ${:.2f}/bbl".format(crack or 0), "",
        "EIA DISTILLATE STOCKS",
        "  Latest : {:,.0f} Mbbl".format(eia_s) if eia_s else "  Latest : N/A",
        "  WoW    : {:+,.0f} Mbbl".format(wow) if wow else "  WoW    : N/A", "",
        "REGIME DETECTION",
        "  Current regime: {}".format(regime),
        "  Scenario weights:",
    ]
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append("    {:18s}: {:.0%}".format(k, v))
    lines += [
        "",
        "MODEL",
        "  Ensemble: 40% log-normal + 35% bootstrap + 25% mean-reversion (OU)",
        "  Regime weights applied via VIX, crack spread, and EIA inventory signals.",
        "  All probabilities are model outputs. Not financial advice.",
    ]
    return "\n".join(lines)


# -- CHARTS --------------------------------------------------------------------

def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def chart_price(history, ho_price):
    dates, prices = [], []
    for row in history:
        try:
            from dateutil import parser as dp
            dates.append(dp.parse(str(row["date"])).date())
            prices.append(float(row["price"]))
        except Exception:
            pass
    if not dates:
        return None

    today = datetime.date.today()
    if dates[-1] != today:
        dates.append(today); prices.append(ho_price)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    ax.grid(color='#1e2936', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.plot(dates, prices, color='#ff9500', linewidth=2.0, label='HO front-month')
    ax.fill_between(dates, prices, min(prices)*0.97, alpha=0.10, color='#ff9500')
    if len(prices) >= 20:
        ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
        ax.plot(dates[19:], ma20, color='#00d4ff', linewidth=1.1,
                linestyle='--', alpha=0.75, label='20d MA')
    ax.scatter([today], [ho_price], color='#fff', s=60, zorder=10)
    ax.annotate(' ${:.4f}'.format(ho_price), xy=(today, ho_price),
                color='#fff', fontsize=9, fontweight='bold', va='center')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=30, color='#8b9ab5', fontsize=8)
    plt.yticks(color='#8b9ab5', fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:'${:.2f}'.format(x)))
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2936')
    fig.suptitle('HEATING OIL -- 1-YEAR PRICE HISTORY',
                 color='#e8edf5', fontsize=13, fontweight='bold',
                 fontfamily='monospace', y=0.97)
    ax.set_ylabel('Price ($/gal)', color='#8b9ab5', fontsize=10)
    ax.legend(loc='upper left', framealpha=0.2, facecolor='#0d1117',
              edgecolor='#1e2936', labelcolor='#c8d0de', fontsize=8)
    plt.tight_layout()
    return _b64(fig)


def chart_prob_table(prob_table):
    horizons = list(prob_table.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor('#0d1117')
    colors = ['#ff9500','#e08000','#c06800','#a05000','#803800']
    for idx, (h, ax) in enumerate(zip(horizons, axes)):
        bins  = [b for b,_ in prob_table[h]]
        probs = [p*100 for _,p in prob_table[h]]
        bars  = ax.barh(bins, probs, color=colors[idx], alpha=0.85, edgecolor='#0d1117')
        ax.set_facecolor('#0d1117')
        ax.set_title(h, color='#e8edf5', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#8b9ab5', labelsize=7)
        ax.set_xlabel('Prob (%)', color='#8b9ab5', fontsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e2936')
        for bar, p in zip(bars, probs):
            if p > 1.5:
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        '{:.1f}%'.format(p), va='center', color='#c8d0de', fontsize=6.5)
    fig.suptitle('HEATING OIL PRICE PROBABILITY DISTRIBUTION BY HORIZON',
                 color='#e8edf5', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _b64(fig)


def chart_scenario(weights):
    labels  = [k.replace("_"," ").title() for k in weights]
    values  = list(weights.values())
    colors  = ['#00d4ff','#ffd700','#ff6b35','#f85149','#bc8cff']
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.0f%%',
        colors=colors[:len(values)], startangle=140,
        textprops={'color':'#c8d0de','fontsize':9},
        wedgeprops={'edgecolor':'#0d1117','linewidth':2}
    )
    for at in autotexts: at.set_color('#0d1117'); at.set_fontweight('bold')
    ax.set_title("Scenario Weights (rule-based)", color='#e8edf5', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _b64(fig)


def chart_custom_bands(custom_bands_by_horizon):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
    ax.grid(color='#1e2936', linewidth=0.5, linestyle='--', alpha=0.6)

    colors = ['#00d4ff','#ff9500','#ffd700','#ff4444','#bc8cff']
    x = np.arange(len(HORIZONS))
    band_labels = list(custom_bands_by_horizon[HORIZONS[0]].keys())
    width = 0.15

    for i, label in enumerate(band_labels):
        vals = [custom_bands_by_horizon[h][label]*100 for h in HORIZONS]
        ax.bar(x + i*width, vals, width, label=label,
               color=colors[i % len(colors)], alpha=0.85, edgecolor='#0d1117')

    ax.set_xticks(x + width*(len(band_labels)-1)/2)
    ax.set_xticklabels(HORIZONS, color='#8b9ab5', fontsize=9)
    ax.tick_params(colors='#8b9ab5')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:'{:.0f}%'.format(x)))
    for sp in ax.spines.values(): sp.set_edgecolor('#1e2936')
    fig.suptitle('CUSTOM RISK BAND PROBABILITIES BY HORIZON',
                 color='#e8edf5', fontsize=12, fontweight='bold', y=0.97)
    ax.set_ylabel('Probability (%)', color='#8b9ab5', fontsize=10)
    ax.legend(loc='upper right', framealpha=0.2, facecolor='#0d1117',
              edgecolor='#1e2936', labelcolor='#c8d0de', fontsize=8)
    plt.tight_layout()
    return _b64(fig)


# -- MAIN ENTRY POINT ----------------------------------------------------------

def run(send=print):
    send("=== HEATING OIL PROBABILITY ENGINE (deterministic) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "{}_{}".format(ts, uuid.uuid4().hex[:8]))
    os.makedirs(run_dir, exist_ok=True)
    send("  Output: {}".format(run_dir))

    # -- Market data -----------------------------------------------------------
    names = ["HO", "WTI", "Brent", "RBOB", "DXY", "VIX"]
    raw   = _df.fetch_all(names, send=send, history_days=365)

    ho_d    = raw.get("HO",    {})
    wti_d   = raw.get("WTI",   {})
    brent_d = raw.get("Brent", {})
    rbob_d  = raw.get("RBOB",  {})
    dxy_d   = raw.get("DXY",   {})
    vix_d   = raw.get("VIX",   {})

    ho      = ho_d.get("current", 3.50)
    wti     = wti_d.get("current")
    brent   = brent_d.get("current")
    rbob    = rbob_d.get("current")
    dxy     = dxy_d.get("current")
    vix     = vix_d.get("current")
    returns = ho_d.get("returns", [])
    history = ho_d.get("history", [])

    today_str = str(datetime.date.today())
    if history and history[-1]["date"] != today_str:
        history.append({"date": today_str, "price": ho})
    elif not history:
        history = [{"date": today_str, "price": ho}]

    crack = round(ho * 42 - (wti or 0), 2) if wti else None
    send("  HO ${:.4f}  WTI ${:.2f}  crack ${:.2f}  VIX {:.1f}".format(
        ho, wti or 0, crack or 0, vix or 0))

    # -- EIA data --------------------------------------------------------------
    eia_data = fetch_eia_distillate(send)

    # -- Regime & scenario weights ---------------------------------------------
    send("Detecting market regime ...")
    regime, weights = detect_regime(ho, wti, vix, crack, eia_data)
    send("  Regime: {}".format(regime))

    # -- Probability tables ----------------------------------------------------
    send("Running probability models ...")
    # Long-run mean: 90-day average of HO history
    prices_hist = [r["price"] for r in history]
    long_run_mean = float(np.mean(prices_hist)) if prices_hist else ho

    prob_table         = {}
    custom_by_horizon  = {}

    for h in HORIZONS:
        days = HORIZON_DAYS[h]
        try:
            p1 = lognormal_probs(ho, returns, days)
            p2 = bootstrap_probs(ho, returns, days)
            p3 = mean_reversion_probs(ho, returns, days, long_run_mean)
            ens = ensemble3(p1, p2, p3)
        except Exception as e:
            send("  [WARN] Model failed for {}: {}".format(h, e))
            ens = [1/len(HO_PRICE_BINS)] * len(HO_PRICE_BINS)

        prob_table[h] = list(zip(HO_PRICE_BINS, ens))
        custom_by_horizon[h] = {
            label: band_prob(ens, lo, hi)
            for label, lo, hi in CUSTOM_BANDS
        }
        send("  {} done  (top bin: {})".format(
            h, max(zip(HO_PRICE_BINS, ens), key=lambda x: x[1])[0]))

    # -- Summary ---------------------------------------------------------------
    summary = market_summary(ho, wti, brent, vix, dxy, crack, eia_data, regime, weights)

    # -- Charts ----------------------------------------------------------------
    send("Generating charts ...")
    price_chart    = chart_price(history, ho)
    prob_chart     = chart_prob_table(prob_table)
    scenario_chart = chart_scenario(weights)
    bands_chart    = chart_custom_bands(custom_by_horizon)

    # -- Save report -----------------------------------------------------------
    report_path = os.path.join(run_dir, "ho_report_{}.txt".format(ts))
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary + "\n\n")
        f.write("--- PROBABILITY TABLES ---\n")
        for h in HORIZONS:
            f.write("\n{}:\n".format(h))
            for b, p in prob_table[h]:
                f.write("  {:18s} {:.2%}\n".format(b, p))
        f.write("\n--- CUSTOM BANDS ---\n")
        for h in HORIZONS:
            f.write("\n{}:\n".format(h))
            for label, p in custom_by_horizon[h].items():
                f.write("  {:20s} {:.2%}\n".format(label, p))
    send("  Report saved: {}".format(report_path))
    send("=== DONE ===")

    return {
        "agent":      "ho",
        "ho_price":   ho,
        "history":    history,
        "market_data": {
            "HO": ho, "WTI": wti, "Brent": brent,
            "RBOB": rbob, "DXY": dxy, "VIX": vix,
            "crack_spread": crack,
        },
        "eia_data":          eia_data,
        "regime":            regime,
        "prob_table":        {h: list(prob_table[h]) for h in HORIZONS},
        "custom_bands":      custom_by_horizon,
        "scenario_weights":  weights,
        "summary":           summary,
        "charts": {
            "price":    price_chart,
            "prob":     prob_chart,
            "scenario": scenario_chart,
            "bands":    bands_chart,
        },
        "run_dir": run_dir,
    }


if __name__ == "__main__":
    result = run()
    print(result["summary"])
