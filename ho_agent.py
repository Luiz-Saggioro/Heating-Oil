#!/usr/bin/env python3
"""
Heating Oil Probability Engine -- Deterministic, no LLM.
Fetches live HO/WTI/Brent/RBOB/DXY/VIX, runs a 3-model ensemble.
Called by streamlit_app.py via run().

Security: EIA_API_KEY loaded from environment only — never hardcoded.
"""

from __future__ import annotations
import os, io, json, uuid, base64, datetime, warnings
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import data_fetcher as _df

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# Load .env for local dev — must happen before _eia_key() is ever called
try:
    from dotenv import load_dotenv as _load_dotenv
    # find .env relative to this file so it works from any working directory
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    _load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

warnings.filterwarnings('ignore')

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
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
    ("P(3.30-3.69)",  3.30,   3.69    ),
    ("P(>5.10)",      5.10,   np.inf  ),
    ("P(>6.25)",      6.25,   np.inf  ),
    ("P(<3.00)",     -np.inf, 3.00    ),
    ("P(<1.80)",     -np.inf, 1.80    ),
]

# Month abbreviations for contract labeling
MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Default display bins for expiry distribution table (Table 2)
DISPLAY_BIN_EDGES  = [-np.inf, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, np.inf]
DISPLAY_BIN_LABELS = ["<$2.00","$2.00-2.50","$2.50-3.00","$3.00-3.50","$3.50-4.00","$4.00-4.50",">$4.50"]


def _eia_key():
    """
    Read EIA_API_KEY from every possible source, in priority order:
      1. Streamlit Cloud secrets  (st.secrets["EIA_API_KEY"])
      2. os.environ               (works for local .env loaded by dotenv)
      3. .env file directly       (last resort re-read)
    Returns empty string if none found or key is the placeholder "DEMO_KEY".
    """
    # 1. Streamlit secrets (Streamlit Cloud / secrets.toml)
    try:
        import streamlit as _st
        k = _st.secrets.get("EIA_API_KEY", "").strip()
        if k and k != "DEMO_KEY":
            return k
    except Exception:
        pass

    # 2. Environment variable (loaded by dotenv or set directly)
    k = os.environ.get("EIA_API_KEY", "").strip()
    if k and k != "DEMO_KEY":
        return k

    # 3. Re-read .env directly as a last resort
    try:
        from dotenv import load_dotenv as _ld
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        _ld(dotenv_path=_env_path, override=True)
        k = os.environ.get("EIA_API_KEY", "").strip()
        if k and k != "DEMO_KEY":
            return k
    except Exception:
        pass

    return ""


# ── Probability Models ────────────────────────────────────────────────────────

def lognormal_probs(current, returns, horizon_days):
    from scipy.stats import norm
    r = np.array(returns) if len(returns) > 5 else np.random.normal(0, 0.015, 60)
    sig = min(float(np.std(r, ddof=1)) or 0.015, 0.80/np.sqrt(252))
    lm  = np.log(current) - 0.5*sig**2*horizon_days
    ls  = sig*np.sqrt(horizon_days)
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        p_lo = norm.cdf(np.log(max(lo,1e-6)), loc=lm, scale=ls) if lo>-np.inf else 0.0
        p_hi = norm.cdf(np.log(hi),           loc=lm, scale=ls) if hi<np.inf  else 1.0
        probs.append(max(0.0, p_hi-p_lo))
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def bootstrap_probs(current, returns, horizon_days, n=3000):
    r = np.array(returns) if len(returns)>5 else np.random.normal(0,0.015,60)
    r = r - np.mean(r)
    sims  = np.exp(np.sum(np.random.choice(r, size=(n, horizon_days), replace=True), axis=1))
    final = current * sims
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        mask = ((final>=lo) if lo>-np.inf else np.ones(n,bool)) & \
               ((final<hi)  if hi<np.inf  else np.ones(n,bool))
        probs.append(float(mask.sum())/n)
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def mean_reversion_probs(current, returns, horizon_days, long_run_mean=None):
    from scipy.stats import norm
    r = np.array(returns) if len(returns)>5 else np.random.normal(0,0.015,60)
    sig_d = min(float(np.std(r,ddof=1)) or 0.015, 0.80/np.sqrt(252))
    sig_a = sig_d*np.sqrt(252)
    if long_run_mean is None: long_run_mean = current
    kappa  = 0.30
    T      = horizon_days/252.0
    exp_kT = np.exp(-kappa*T)
    fwd_mean = np.log(long_run_mean)+(np.log(current)-np.log(long_run_mean))*exp_kT
    fwd_var  = (sig_a**2/(2.0*kappa))*(1.0-exp_kT**2)
    fwd_sig  = np.sqrt(max(fwd_var,1e-8))
    probs = []
    for i in range(len(BIN_EDGES)-1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i+1]
        p_lo = norm.cdf(np.log(max(lo,1e-6)), loc=fwd_mean, scale=fwd_sig) if lo>-np.inf else 0.0
        p_hi = norm.cdf(np.log(hi),           loc=fwd_mean, scale=fwd_sig) if hi<np.inf  else 1.0
        probs.append(max(0.0, p_hi-p_lo))
    t = sum(probs) or 1.0
    return [p/t for p in probs]


def ensemble3(p1, p2, p3, w=(0.40, 0.35, 0.25)):
    combined = [w[0]*a+w[1]*b+w[2]*c for a,b,c in zip(p1,p2,p3)]
    t = sum(combined) or 1.0
    return [p/t for p in combined]


def band_prob(ens, lo, hi):
    prob = 0.0
    for i in range(len(BIN_EDGES)-1):
        b_lo, b_hi = BIN_EDGES[i], BIN_EDGES[i+1]
        if min(b_hi,hi) > max(b_lo,lo):
            prob += ens[i]
    return round(prob, 4)


def compute_ev_by_horizon(prob_table):
    bin_mids = [1.60,2.00,2.40,2.80,3.15,3.495,3.895,4.35,4.85,5.40,5.975,6.75]
    ev = {}
    for h in HORIZONS:
        rows  = prob_table.get(h, [])
        total = sum(p for _,p in rows) or 1
        ev[h] = round(sum(mid*p for mid,(_,p) in zip(bin_mids,rows))/total, 4)
    return ev


def compute_var_es(current, returns, horizon_days=21, confidence=0.95, n_sims=10000):
    r = np.array(returns) if len(returns)>5 else np.random.normal(0,0.015,60)
    r = r - np.mean(r)
    sims  = np.exp(np.sum(np.random.choice(r, size=(n_sims, horizon_days), replace=True), axis=1))
    pnl   = current*sims - current
    var   = float(np.percentile(pnl, (1-confidence)*100))
    es    = float(np.mean(pnl[pnl<=var]))
    return {
        "confidence": round(confidence*100, 0),
        "horizon_days": horizon_days,
        "var": round(var, 4),
        "es":  round(es, 4),
        "pnl_distribution": [round(float(p),4) for p in np.percentile(pnl,[1,5,10,25,50,75,90,95,99])],
        "percentile_labels": ["1%","5%","10%","25%","50%","75%","90%","95%","99%"],
    }


def _build_lognorm_shape(current, sig_daily, horizon_days):
    from scipy.stats import lognorm as _lognorm
    lm  = np.log(current) - 0.5*sig_daily**2*horizon_days
    ls  = sig_daily*np.sqrt(horizon_days)
    x_lo = max(0.5, float(np.exp(lm - 3*ls)))
    x_hi = float(np.exp(lm + 3*ls))
    xs = np.linspace(x_lo, x_hi, 200)
    ys = _lognorm.pdf(xs, s=ls, scale=np.exp(lm))
    skew = float((np.exp(ls**2)+2)*np.sqrt(np.exp(ls**2)-1))
    kurt = float(np.exp(4*ls**2)+2*np.exp(3*ls**2)+3*np.exp(2*ls**2)-6)
    return {
        "x": [round(float(v),4) for v in xs],
        "y": [round(float(v),6) for v in ys],
        "mean":     round(float(np.exp(lm+0.5*ls**2)),4),
        "median":   round(float(np.exp(lm)),4),
        "skewness": round(skew,3),
        "kurtosis": round(kurt,3),
    }


def build_crack_spread_history(ho_history, wti_history):
    wti_map = {r["date"]: r["price"] for r in wti_history}
    return [
        {"date": r["date"], "crack": round(float(r["price"])*42 - float(wti_map[r["date"]]), 2)}
        for r in ho_history if r["date"] in wti_map
    ]


# ── Forward Curve, KO Barrier & Expiry Distribution ──────────────────────────

def _last_biz_day(year, month):
    """Return the last business day (Mon–Fri) of the given year/month."""
    import calendar
    last = calendar.monthrange(year, month)[1]
    d = datetime.date(year, month, last)
    while d.weekday() >= 5:          # 5=Sat, 6=Sun
        d -= datetime.timedelta(days=1)
    return d


def get_ho_contract_schedule(spot_price, sigma_daily, today=None):
    """
    Build 13 monthly HO contract rows (front month + next 12).
    HO futures expire on the last business day of the month PRIOR to delivery.
    Forward prices use a flat curve with a light seasonal overlay.

    Returns a list of dicts with keys:
      label, expiry_date, t_days (trading-day equivalent), fwd_price, sigma_daily
    """
    if today is None:
        today = datetime.date.today()

    # Locate the front delivery month: first month whose expiry (= last biz day of
    # the prior month) is still on or after today.
    y, m = today.year, today.month
    for _ in range(14):                        # safety cap
        exp_m, exp_y = m - 1, y
        if exp_m == 0:
            exp_m, exp_y = 12, y - 1
        if _last_biz_day(exp_y, exp_m) >= today:
            break
        m += 1
        if m > 12:
            m, y = 1, y + 1

    # Seasonal multipliers — mild winter premium, otherwise flat
    seasonal_mult = {
        "Nov": 1.020, "Dec": 1.030, "Jan": 1.025,
        "Feb": 1.015, "Mar": 1.005,
    }

    contracts = []
    for _ in range(13):
        exp_m, exp_y = m - 1, y
        if exp_m == 0:
            exp_m, exp_y = 12, y - 1
        exp_date    = _last_biz_day(exp_y, exp_m)
        t_cal       = max(1, (exp_date - today).days)
        # Convert calendar days to approximate trading days (252 / 365 ratio)
        t_days      = max(1, round(t_cal * 252 / 365))
        month_label = MONTH_ABBR[m - 1]
        fwd_price   = round(spot_price * seasonal_mult.get(month_label, 1.0), 4)
        contracts.append({
            "label":       "{} {}".format(month_label, y),
            "expiry_date": str(exp_date),
            "t_days":      t_days,           # trading days (used for vol scaling)
            "t_cal":       t_cal,            # calendar days (for reference)
            "fwd_price":   fwd_price,
            "sigma_daily": sigma_daily,
        })
        m += 1
        if m > 12:
            m, y = 1, y + 1

    return contracts


def barrier_touch_prob(S, B, T_days, sigma_daily, drift_daily=0.0):
    """
    Probability that a GBM price touches barrier B at any time in [0, T_days].
    Uses the reflection-principle formula for log-normal diffusion.

    Parameters
    ----------
    S           : current / forward futures price
    B           : knock-out barrier price
    T_days      : trading days to expiry
    sigma_daily : daily log-return volatility
    drift_daily : log-drift per trading day; 0 = risk-neutral martingale (default for futures)

    Formula (unified for upper & lower barriers):
        c = ln(B / S)           — signed log-distance
        d1 = (m·T − |c|) / (σ·√T)
        d2 = (−m·T − |c|) / (σ·√T)
        P = Φ(d1) + exp(2·m·c / σ²) · Φ(d2)

    Zero-drift simplification: P = 2·Φ(−|c| / (σ·√T))
    """
    from scipy.stats import norm as _norm
    if S <= 0 or B <= 0 or sigma_daily <= 0 or T_days <= 0:
        return 0.0
    if abs(S - B) < 1e-8:
        return 1.0          # already at the barrier
    c     = float(np.log(B / S))
    abs_c = abs(c)
    sT    = sigma_daily * float(np.sqrt(T_days))
    m     = drift_daily

    if abs(m) < 1e-10:
        prob = 2.0 * float(_norm.cdf(-abs_c / sT))
    else:
        d1   = (m * T_days - abs_c) / sT
        d2   = (-m * T_days - abs_c) / sT
        prob = float(_norm.cdf(d1)) + float(np.exp(2.0 * m * c / sigma_daily**2)) * float(_norm.cdf(d2))

    return float(np.clip(prob, 0.0, 1.0))


def compute_ko_probabilities(contracts, ko_price):
    """
    Table 1: For each contract compute P(price touches ko_price before expiry).

    Parameters
    ----------
    contracts : list of dicts from get_ho_contract_schedule()
    ko_price  : float — the knock-out barrier ($/gal)

    Returns a list of dicts with keys:
      label, expiry, fwd_price, ko_price, ko_prob (%), t_days
    """
    rows = []
    for c in contracts:
        sig  = c.get("sigma_daily") or 0.015
        prob = barrier_touch_prob(c["fwd_price"], ko_price, c["t_days"], sig)
        rows.append({
            "label":     c["label"],
            "expiry":    c["expiry_date"],
            "fwd_price": c["fwd_price"],
            "ko_price":  round(float(ko_price), 4),
            "ko_prob":   round(prob * 100, 2),
            "t_days":    c["t_days"],
        })
    return rows


def compute_expiry_distributions(contracts, sigma_daily, bin_edges=None, bin_labels=None):
    """
    Table 2: For each contract compute the lognormal settlement distribution.

    Parameters
    ----------
    contracts   : list of dicts from get_ho_contract_schedule()
    sigma_daily : daily volatility (used as fallback if contract doesn't carry its own)
    bin_edges   : list of floats (±inf allowed) defining N+1 edges for N bins.
                  Defaults to DISPLAY_BIN_EDGES.
    bin_labels  : list of N label strings. Defaults to DISPLAY_BIN_LABELS.

    Returns a list of dicts with keys:
      label, expiry, fwd_price, t_days, bin_probs (list of % per bin), bin_labels
    """
    from scipy.stats import norm as _norm
    if bin_edges is None:
        bin_edges  = DISPLAY_BIN_EDGES
    if bin_labels is None:
        bin_labels = DISPLAY_BIN_LABELS

    rows = []
    for c in contracts:
        sig = c.get("sigma_daily") or sigma_daily or 0.015
        S   = c["fwd_price"]
        T   = c["t_days"]
        # Lognormal params — zero drift (risk-neutral martingale for futures)
        lm  = float(np.log(S)) - 0.5 * sig**2 * T
        ls  = sig * float(np.sqrt(T))

        bin_probs = []
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            p_lo = _norm.cdf(np.log(max(lo, 1e-9)), loc=lm, scale=ls) if lo > -np.inf else 0.0
            p_hi = _norm.cdf(np.log(hi),            loc=lm, scale=ls) if hi <  np.inf else 1.0
            bin_probs.append(float(np.clip(p_hi - p_lo, 0.0, 1.0)))

        total = sum(bin_probs) or 1.0
        bin_probs = [round(p / total * 100, 2) for p in bin_probs]

        rows.append({
            "label":      c["label"],
            "expiry":     c["expiry_date"],
            "fwd_price":  S,
            "t_days":     T,
            "bin_probs":  bin_probs,
            "bin_labels": list(bin_labels),
        })
    return rows


def fetch_eia_distillate(send=print):
    """
    Fetch weekly US distillate fuel oil stocks from EIA API.

    Waterfall fetch strategy (6 paths — urllib first, requests fallback per source):
      1.  EIA v2 (urllib)   — raw brackets in URL (EIA rejects %5B%5D)
      1b. EIA v2 (requests) — SSL/cert fallback for Streamlit Cloud
      2.  EIA v1 (urllib)   — legacy series endpoint, broader key compatibility
      2b. EIA v1 (requests) — SSL/cert fallback
      3.  FRED CSV (urllib) — WDISTUS mirrors EIA weekly distillate
      3b. FRED CSV (requests) — SSL/cert fallback

    Product: EPD2F (Distillate Fuel Oil, No. 2), Area: NUS (US Total)
    """
    import urllib.parse
    import urllib.request as _ur
    from collections import defaultdict

    send("Fetching EIA distillate stocks ...")
    key = _eia_key()
    if not key:
        send("  [WARN] EIA_API_KEY not found in Streamlit secrets, environment, or .env file")
        send("  [INFO] On Streamlit Cloud: Manage app > Settings > Secrets > add EIA_API_KEY")
        send("  [INFO] Locally: add EIA_API_KEY=<your_key> to your .env file")
        return {"stocks_mbbl": None, "wow_change": None, "weeks": [], "history": [],
                "seasonal_bands": [], "current_year_data": [], "key_missing": True}

    send("  EIA key found (length {}), calling API ...".format(len(key)))

    # ── URL builder: keeps brackets UNENCODED (EIA v2 rejects %5B%5D) ─────────
    def _eia_v2_url(n_rows=260):
        pairs = [
            ("api_key",            key),
            ("frequency",          "weekly"),
            ("data[]",             "value"),
            ("facets[product][]",  "EPD2F"),
            ("facets[duoarea][]",  "NUS"),
            ("sort[0][column]",    "period"),
            ("sort[0][direction]", "desc"),
            ("length",             str(n_rows)),
        ]
        qs = "&".join("{}={}".format(k, urllib.parse.quote(str(v), safe=""))
                      for k, v in pairs)
        return "https://api.eia.gov/v2/petroleum/stoc/wstk/data/?" + qs

    # ── Common: parse v2 JSON payload → list of {period, value} dicts ─────────
    def _parse_v2(payload, send):
        inner = payload.get("response", {})
        rows  = inner.get("data", [])
        send("  EIA v2 rows in response: {}".format(len(rows)))
        if not rows:
            errs = payload.get("error") or inner.get("error") or str(payload)[:300]
            send("  EIA v2 no-data detail: {}".format(errs))
            return []
        valid = [r for r in rows
                 if r.get("value") not in (None, "", "null", "NA")]
        send("  EIA v2 valid rows: {}".format(len(valid)))
        return [{"period": r["period"], "value": float(r["value"])} for r in valid]

    # ── Source 1: EIA v2 via urllib ───────────────────────────────────────────
    rows_desc = []
    for n_rows in (260, 52):   # try 5-year first, fall back to 1-year
        url = _eia_v2_url(n_rows)
        send("  [EIA v2] GET {} rows — {}".format(n_rows, url.replace(key, "***")))
        try:
            req  = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                             "Accept":     "application/json"})
            with _ur.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode())
            rows_desc = _parse_v2(payload, send)
            if rows_desc:
                send("  [EIA v2] OK — {} data points".format(len(rows_desc)))
                break
        except _ur.HTTPError as e:
            deny = e.headers.get("x-deny-reason", "")
            body = ""
            try: body = e.read().decode()[:200]
            except Exception: pass
            send("  [EIA v2] HTTPError {} {} deny={} body={}".format(
                e.code, e.reason, deny, body))
        except Exception as e:
            send("  [EIA v2] {}: {}".format(type(e).__name__, e))

    # ── Source 1b: EIA v2 via requests (Streamlit Cloud SSL fallback) ─────────
    if not rows_desc and _REQUESTS_OK:
        send("  [EIA v2-req] Retrying EIA v2 with requests library ...")
        for n_rows in (260, 52):
            url = _eia_v2_url(n_rows)
            send("  [EIA v2-req] GET {} rows".format(n_rows))
            try:
                resp = _requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                    timeout=25,
                    verify=True,
                )
                resp.raise_for_status()
                rows_desc = _parse_v2(resp.json(), send)
                if rows_desc:
                    send("  [EIA v2-req] OK — {} data points".format(len(rows_desc)))
                    break
            except Exception as e:
                send("  [EIA v2-req] {}: {}".format(type(e).__name__, e))

    # ── Source 2: EIA v1 series API via urllib ────────────────────────────────
    if not rows_desc:
        send("  [EIA v1] Trying legacy series API ...")
        v1_url = ("https://api.eia.gov/series/"
                  "?api_key={}&series_id=PET.WDISTUS1.W&num=260".format(key))
        send("  [EIA v1] GET {}".format(v1_url.replace(key, "***")))
        try:
            req = _ur.Request(v1_url, headers={"User-Agent": "Mozilla/5.0",
                                               "Accept":     "application/json"})
            with _ur.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode())
            series = payload.get("series", [])
            if series and series[0].get("data"):
                raw_pts = series[0]["data"]
                for pt in raw_pts:
                    try:
                        dt_str  = str(pt[0])
                        val     = float(pt[1])
                        dt      = datetime.datetime.strptime(dt_str, "%Y%m%d")
                        rows_desc.append({"period": dt.strftime("%Y-%m-%d"), "value": val})
                    except Exception:
                        pass
                send("  [EIA v1] OK — {} pts".format(len(rows_desc)))
            else:
                errs = payload.get("data", {}).get("error") or str(payload)[:300]
                send("  [EIA v1] No series data: {}".format(errs))
        except _ur.HTTPError as e:
            deny = e.headers.get("x-deny-reason", "")
            body = ""
            try: body = e.read().decode()[:200]
            except Exception: pass
            send("  [EIA v1] HTTPError {} {} deny={} body={}".format(
                e.code, e.reason, deny, body))
        except Exception as e:
            send("  [EIA v1] {}: {}".format(type(e).__name__, e))

    # ── Source 2b: EIA v1 via requests (Streamlit Cloud SSL fallback) ─────────
    if not rows_desc and _REQUESTS_OK:
        send("  [EIA v1-req] Retrying EIA v1 with requests library ...")
        v1_url = ("https://api.eia.gov/series/"
                  "?api_key={}&series_id=PET.WDISTUS1.W&num=260".format(key))
        try:
            resp = _requests.get(
                v1_url,
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=25,
                verify=True,
            )
            resp.raise_for_status()
            payload = resp.json()
            series = payload.get("series", [])
            if series and series[0].get("data"):
                for pt in series[0]["data"]:
                    try:
                        dt = datetime.datetime.strptime(str(pt[0]), "%Y%m%d")
                        rows_desc.append({"period": dt.strftime("%Y-%m-%d"),
                                          "value":  float(pt[1])})
                    except Exception:
                        pass
                rows_desc.sort(key=lambda r: r["period"], reverse=True)
                send("  [EIA v1-req] OK — {} pts".format(len(rows_desc)))
            else:
                send("  [EIA v1-req] No series data in response")
        except Exception as e:
            send("  [EIA v1-req] {}: {}".format(type(e).__name__, e))

    # ── Source 3: FRED CSV via urllib (WDISTUS mirrors EIA weekly distillate) ──
    if not rows_desc:
        send("  [FRED] Trying FRED WDISTUS series as last resort ...")
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WDISTUS"
        send("  [FRED] GET {}".format(fred_url))
        try:
            req = _ur.Request(fred_url, headers={"User-Agent": "Mozilla/5.0"})
            with _ur.urlopen(req, timeout=25) as resp:
                raw = resp.read().decode()
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) == 2 and parts[1] not in (".", ""):
                    try:
                        rows_desc.append({"period": parts[0], "value": float(parts[1])})
                    except ValueError:
                        pass
            rows_desc.sort(key=lambda r: r["period"], reverse=True)
            send("  [FRED] OK — {} pts".format(len(rows_desc)))
        except _ur.HTTPError as e:
            deny = e.headers.get("x-deny-reason", "")
            send("  [FRED] HTTPError {} deny={}".format(e.code, deny))
        except Exception as e:
            send("  [FRED] {}: {}".format(type(e).__name__, e))

    # ── Source 3b: FRED CSV via requests (Streamlit Cloud SSL fallback) ────────
    if not rows_desc and _REQUESTS_OK:
        send("  [FRED-req] Retrying FRED with requests library ...")
        fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WDISTUS"
        try:
            resp = _requests.get(
                fred_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=25,
                verify=True,
            )
            resp.raise_for_status()
            lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) == 2 and parts[1] not in (".", ""):
                    try:
                        rows_desc.append({"period": parts[0], "value": float(parts[1])})
                    except ValueError:
                        pass
            rows_desc.sort(key=lambda r: r["period"], reverse=True)
            send("  [FRED-req] OK — {} pts".format(len(rows_desc)))
        except Exception as e:
            send("  [FRED-req] {}: {}".format(type(e).__name__, e))

    # ── All sources failed ────────────────────────────────────────────────────
    if not rows_desc:
        send("  [WARN] All EIA/FRED sources failed — inventory section will be empty")
        return {"stocks_mbbl": None, "wow_change": None, "weeks": [], "history": [],
                "seasonal_bands": [], "current_year_data": []}

    # ── Process data (rows_desc is newest-first) ──────────────────────────────
    latest = rows_desc[0]["value"]
    prev   = rows_desc[1]["value"] if len(rows_desc) > 1 else latest
    send("  Distillate stocks: {:,.0f} Mbbl  WoW: {:+.0f}".format(latest, latest - prev))

    history_all_asc = list(reversed(rows_desc))

    from collections import defaultdict
    week_buckets = defaultdict(list)
    for row in history_all_asc:
        try:
            dt       = datetime.datetime.strptime(row["period"], "%Y-%m-%d").date()
            week_num = dt.isocalendar()[1]
            week_buckets[week_num].append(row["value"])
        except Exception:
            pass

    seasonal_bands = []
    for wk in sorted(week_buckets.keys()):
        vals = week_buckets[wk]
        if len(vals) >= 2:
            seasonal_bands.append({
                "week": wk,
                "avg":  round(float(np.mean(vals)), 0),
                "min":  round(float(np.min(vals)), 0),
                "max":  round(float(np.max(vals)), 0),
            })

    current_year      = datetime.date.today().year
    current_year_data = []
    for row in history_all_asc:
        try:
            dt = datetime.datetime.strptime(row["period"], "%Y-%m-%d").date()
            if dt.year == current_year:
                current_year_data.append({
                    "week":   dt.isocalendar()[1],
                    "period": row["period"],
                    "value":  row["value"],
                })
        except Exception:
            pass

    send("  Seasonal bands: {} weeks  current-year overlay: {} pts".format(
        len(seasonal_bands), len(current_year_data)))

    return {
        "stocks_mbbl":       latest,
        "wow_change":        latest - prev,
        "weeks":             rows_desc[:8],
        "history":           history_all_asc,
        "seasonal_bands":    seasonal_bands,
        "current_year_data": current_year_data,
    }


def detect_regime(ho, wti, vix, crack_spread, eia_data):
    weights = {"de-escalation":0.15,"status_quo":0.40,"escalation":0.20,"recession":0.15,"phys_squeeze":0.10}
    if vix and vix>30:
        weights["escalation"]+=0.08; weights["recession"]+=0.05
        weights["status_quo"]-=0.08; weights["de-escalation"]-=0.05
    elif vix and vix<15:
        weights["de-escalation"]+=0.05; weights["status_quo"]+=0.05; weights["escalation"]-=0.10
    if crack_spread and crack_spread>30:
        weights["phys_squeeze"]+=0.08; weights["escalation"]+=0.05
        weights["de-escalation"]-=0.08; weights["status_quo"]-=0.05
    elif crack_spread and crack_spread<10:
        weights["de-escalation"]+=0.05; weights["status_quo"]+=0.05
        weights["phys_squeeze"]-=0.07; weights["escalation"]-=0.03
    wow = eia_data.get("wow_change")
    if wow and wow<-3000:
        weights["phys_squeeze"]+=0.05; weights["escalation"]+=0.03
        weights["de-escalation"]-=0.05; weights["status_quo"]-=0.03
    elif wow and wow>3000:
        weights["de-escalation"]+=0.05; weights["recession"]+=0.03
        weights["phys_squeeze"]-=0.05; weights["escalation"]-=0.03
    total = sum(weights.values())
    weights = {k: round(max(0.01,v/total),4) for k,v in weights.items()}
    total = sum(weights.values())
    weights = {k: round(v/total,4) for k,v in weights.items()}
    dominant = max(weights, key=weights.get)
    labels = {"de-escalation":"EASING","status_quo":"STABLE","escalation":"TIGHTENING",
              "recession":"RISK-OFF","phys_squeeze":"SUPPLY SQUEEZE"}
    return labels.get(dominant,"STABLE"), weights


def market_summary(ho, wti, brent, vix, dxy, crack, eia_data, regime, weights):
    vix_r = "HIGH" if vix and vix>30 else "ELEVATED" if vix and vix>20 else "NORMAL"
    eia_s = eia_data.get("stocks_mbbl"); wow = eia_data.get("wow_change")
    lines = [
        "=== HEATING OIL PROBABILITY ENGINE -- {} ===".format(datetime.date.today()),"",
        "MARKET SNAPSHOT",
        "  HO ($/gal)        : ${:.4f}".format(ho),
        "  WTI ($/bbl)       : ${:.2f}".format(wti or 0),
        "  Brent ($/bbl)     : ${:.2f}".format(brent or 0),
        "  DXY               : {:.2f}".format(dxy or 0),
        "  Volatility (VIX)  : {:.1f}  [{}]".format(vix or 0, vix_r),
        "  HO Crack Spread   : ${:.2f}/bbl".format(crack or 0),"",
        "EIA DISTILLATE STOCKS",
        "  Latest : {:,.0f} Mbbl".format(eia_s) if eia_s else "  Latest : N/A",
        "  WoW    : {:+,.0f} Mbbl".format(wow) if wow else "  WoW    : N/A","",
        "REGIME DETECTION", "  Current regime: {}".format(regime), "  Scenario weights:",
    ]
    for k,v in sorted(weights.items(), key=lambda x:-x[1]):
        lines.append("    {:18s}: {:.0%}".format(k,v))
    lines += ["","MODEL",
        "  Ensemble: 40% log-normal + 35% bootstrap + 25% mean-reversion (OU)",
        "  Probabilities are model outputs. Not financial advice."]
    return "\n".join(lines)


def run(send=print):
    send("=== HEATING OIL PROBABILITY ENGINE (deterministic) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_DIR, "{}_{}".format(ts, uuid.uuid4().hex[:8]))
    os.makedirs(run_dir, exist_ok=True)

    names = ["HO","WTI","Brent","RBOB","DXY","VIX"]
    raw   = _df.fetch_all(names, send=send, history_days=365)

    ho_d=raw.get("HO",{}); wti_d=raw.get("WTI",{}); brent_d=raw.get("Brent",{})
    rbob_d=raw.get("RBOB",{}); dxy_d=raw.get("DXY",{}); vix_d=raw.get("VIX",{})

    ho=ho_d.get("current",3.50); wti=wti_d.get("current"); brent=brent_d.get("current")
    rbob=rbob_d.get("current"); dxy=dxy_d.get("current"); vix=vix_d.get("current")
    returns=ho_d.get("returns",[]); history=ho_d.get("history",[])
    wti_history=wti_d.get("history",[])
    is_synthetic_history = ho_d.get("is_synthetic", any(r.get("synthetic") for r in history))

    today_str = str(datetime.date.today())
    if history and history[-1]["date"]!=today_str:
        history.append({"date":today_str,"price":ho})
    elif not history:
        history=[{"date":today_str,"price":ho}]

    crack = round(ho*42-(wti or 0),2) if wti else None
    send("  HO ${:.4f}  WTI ${:.2f}  RBOB ${:.4f}  crack ${:.2f}  Volatility(VIX) {:.1f}".format(
        ho, wti or 0, rbob or 0, crack or 0, vix or 0))

    eia_data = fetch_eia_distillate(send)
    send("Detecting market regime ...")
    regime, weights = detect_regime(ho, wti, vix, crack, eia_data)
    send("  Regime: {}".format(regime))

    send("Running probability models ...")
    prices_hist   = [r["price"] for r in history]
    long_run_mean = float(np.mean(prices_hist)) if prices_hist else ho

    prob_table={}; custom_by_horizon={}
    for h in HORIZONS:
        days = HORIZON_DAYS[h]
        try:
            p1=lognormal_probs(ho,returns,days)
            p2=bootstrap_probs(ho,returns,days)
            p3=mean_reversion_probs(ho,returns,days,long_run_mean)
            ens=ensemble3(p1,p2,p3)
        except Exception as e:
            send("  [WARN] Model failed for {}: {}".format(h,e))
            ens=[1/len(HO_PRICE_BINS)]*len(HO_PRICE_BINS)
        prob_table[h]=list(zip(HO_PRICE_BINS,ens))
        custom_by_horizon[h]={label:band_prob(ens,lo,hi) for label,lo,hi in CUSTOM_BANDS}
        send("  {} done  (top bin: {})".format(h,max(zip(HO_PRICE_BINS,ens),key=lambda x:x[1])[0]))

    summary = market_summary(ho,wti,brent,vix,dxy,crack,eia_data,regime,weights)

    send("Computing extended analytics ...")
    r_arr     = np.array(returns) if len(returns)>5 else np.random.normal(0,0.015,60)
    sig_daily = min(float(np.std(r_arr,ddof=1)) or 0.015, 0.80/np.sqrt(252))

    # CI Bands
    ci_bands={}
    for h in HORIZONS:
        days=HORIZON_DAYS[h]; lm=np.log(ho)-0.5*sig_daily**2*days; ls=sig_daily*np.sqrt(days)
        ci_bands[h]={"mid":round(float(np.exp(lm)),4),
            "ci80":[round(float(np.exp(lm-1.28*ls)),4),round(float(np.exp(lm+1.28*ls)),4)],
            "ci90":[round(float(np.exp(lm-1.645*ls)),4),round(float(np.exp(lm+1.645*ls)),4)],
            "ci95":[round(float(np.exp(lm-1.96*ls)),4),round(float(np.exp(lm+1.96*ls)),4)]}

    # Rolling 10-day vol (line chart)
    vol_heatmap=[]
    if len(history)>=11:
        for i in range(10,len(history)):
            chunk=r_arr[max(0,i-10):i]
            if len(chunk)>=2:
                vol_heatmap.append({"date":history[i]["date"],
                    "vol":round(float(np.std(chunk,ddof=1))*np.sqrt(252)*100,2)})

    # Log-normal shape
    lognorm_shape = _build_lognorm_shape(ho, sig_daily, HORIZON_DAYS["1M"])

    # Drivers (no Retail/Margin)
    vix_v=vix or 20.0; crack_v=crack or 15.0
    seasonal_m=datetime.date.today().month
    seasonal_w=8.0 if seasonal_m in (11,12,1,2,3) else 3.0
    eia_w=min(10.0,abs(eia_data.get("wow_change") or 0)/500)
    raw_drvs=[
        ("Crude Oil (WTI)",   round(min(20,abs((wti or ho*0.6)-ho*0.6)/ho*100),2) if wti else 5.0),
        ("Crack Spread",      round(min(15,crack_v/3),2)),
        ("Seasonal Demand",   round(seasonal_w,2)),
        ("Volatility (VIX)",  round(min(12,max(0,(vix_v-15)/2)),2)),
        ("EIA Inventory",     round(eia_w,2)),
        ("USD Strength (DXY)",round(min(8,abs((dxy or 103)-103)/2),2) if dxy else 2.0),
    ]
    tot=sum(v for _,v in raw_drvs) or 1
    drivers=[{"name":n,"value":round(v,2),"pct":round(v/tot*100,1)} for n,v in raw_drvs]

    # Scenarios — dynamic signal-driven drift + VIX-scaled vol
    np.random.seed(42)
    fc_dates=[]
    d=datetime.date.today()
    while len(fc_dates)<14:
        d+=datetime.timedelta(days=1)
        if d.weekday()<5: fc_dates.append(str(d))

    vix_cur = vix or 20.0
    crack_cur = crack or 15.0
    eia_wow = eia_data.get("wow_change") or 0.0
    seasonal_month = datetime.date.today().month
    crack_signal    = np.clip((crack_cur - 15.0) / 100.0, -0.002, 0.003)
    vix_signal      = np.clip(-(vix_cur - 20.0) / 2000.0, -0.002, 0.001)
    eia_signal      = np.clip(-eia_wow / 5_000_000.0, -0.002, 0.002)
    seasonal_signal = 0.002 if seasonal_month in (11,12,1,2,3) else (-0.001 if seasonal_month in (5,6,7,8) else 0.0)
    base_dynamic_drift = crack_signal + vix_signal + eia_signal + seasonal_signal

    vix_hist = vix_d.get("history", [])
    if len(vix_hist) >= 20:
        vix_rolling_mean = float(np.mean([r["price"] for r in vix_hist[-20:]]))
    else:
        vix_rolling_mean = vix_cur
    vix_vol_mult = float(np.clip(vix_cur / max(vix_rolling_mean, 10.0), 0.5, 2.5))

    scenario_defs = {
        "Base": {
            "drift_bias": 0.000, "vol_mult_base": 1.0,
            "label": "Signals: crack={:.2f}, VIX={:.1f}, seasonal={}".format(
                crack_cur, vix_cur, "peak" if seasonal_month in (11,12,1,2,3) else "off-peak"),
        },
        "High Demand": {
            "drift_bias": +0.003, "vol_mult_base": 1.2,
            "label": "Demand surge: inventory draws, cold-snap premium",
        },
        "Supply Disruption": {
            "drift_bias": +0.007, "vol_mult_base": 1.8,
            "label": "Refinery outage or port disruption; crack spike",
        },
        "Stable Market": {
            "drift_bias": -0.001, "vol_mult_base": 0.6,
            "label": "Low VIX, balanced inventory, mild seasonal",
        },
        "Recession": {
            "drift_bias": -0.006, "vol_mult_base": 1.4,
            "label": "Demand destruction; VIX-driven vol expansion",
        },
    }
    scenario_paths = {}
    for sname, sp in scenario_defs.items():
        total_drift = base_dynamic_drift + sp["drift_bias"]
        total_vol   = sig_daily * sp["vol_mult_base"] * vix_vol_mult
        path = [ho]
        for _ in range(14):
            path.append(round(float(path[-1] * np.exp(np.random.normal(total_drift, total_vol))), 4))
        scenario_paths[sname] = {
            "dates":       fc_dates,
            "prices":      path[1:],
            "final":       round(path[-1], 4),
            "total_drift": round(total_drift * 252 * 100, 2),
            "vol_ann":     round(total_vol * np.sqrt(252) * 100, 2),
            "label":       sp["label"],
        }
    send("  Scenarios built (dynamic drift base={:.4f}, VIX mult={:.2f})".format(
        base_dynamic_drift, vix_vol_mult))

    # Regional prices — US
    regional_prices=[
        {"country":"US","region":"New England","state":"CT","lat":41.6,"lon":-72.7,"price":round(ho*1.18,4),"factor":"High demand, port logistics"},
        {"country":"US","region":"Mid-Atlantic","state":"NY","lat":40.7,"lon":-74.0,"price":round(ho*1.12,4),"factor":"Urban distribution premium"},
        {"country":"US","region":"Southeast","state":"GA","lat":33.7,"lon":-84.4,"price":round(ho*0.97,4),"factor":"Mild climate, lower demand"},
        {"country":"US","region":"Midwest","state":"IL","lat":41.8,"lon":-87.6,"price":round(ho*1.02,4),"factor":"Inland logistics cost"},
        {"country":"US","region":"Gulf Coast","state":"TX","lat":29.7,"lon":-95.4,"price":round(ho*0.91,4),"factor":"Refinery proximity"},
        {"country":"US","region":"West Coast","state":"CA","lat":34.0,"lon":-118.2,"price":round(ho*1.24,4),"factor":"State taxes + CARB spec"},
        {"country":"US","region":"Pacific NW","state":"WA","lat":47.6,"lon":-122.3,"price":round(ho*1.15,4),"factor":"Remote supply chain"},
        {"country":"US","region":"Mountain","state":"CO","lat":39.7,"lon":-104.9,"price":round(ho*1.07,4),"factor":"Altitude distribution cost"},
    ]

    # Brazil diesel/distillate regional prices
    brl_base = ho * 0.98
    brazil_regional_prices = [
        {"country":"BR","region":"São Paulo","state":"SP","lat":-23.5,"lon":-46.6,
         "price":round(brl_base*1.14,4),"factor":"High ICMS (18%), large distribution network"},
        {"country":"BR","region":"Rio de Janeiro","state":"RJ","lat":-22.9,"lon":-43.2,
         "price":round(brl_base*1.16,4),"factor":"Urban premium, port logistics costs"},
        {"country":"BR","region":"Minas Gerais","state":"MG","lat":-19.9,"lon":-43.9,
         "price":round(brl_base*1.11,4),"factor":"Inland distribution, moderate ICMS"},
        {"country":"BR","region":"Bahia","state":"BA","lat":-12.9,"lon":-38.4,
         "price":round(brl_base*1.09,4),"factor":"Refinery proximity (RLAM), lower logistics"},
        {"country":"BR","region":"Paraná","state":"PR","lat":-25.4,"lon":-49.3,
         "price":round(brl_base*1.10,4),"factor":"REPAR refinery nearby, soy belt logistics"},
        {"country":"BR","region":"Amazonas","state":"AM","lat":-3.1,"lon":-60.0,
         "price":round(brl_base*1.31,4),"factor":"Remote access, river-only supply chain"},
        {"country":"BR","region":"Pará","state":"PA","lat":-1.5,"lon":-48.5,
         "price":round(brl_base*1.22,4),"factor":"Amazon basin, limited road infrastructure"},
        {"country":"BR","region":"Rio Grande do Sul","state":"RS","lat":-30.0,"lon":-51.2,
         "price":round(brl_base*1.12,4),"factor":"Southern border, REFAP refinery"},
        {"country":"BR","region":"Ceará","state":"CE","lat":-3.7,"lon":-38.5,
         "price":round(brl_base*1.18,4),"factor":"Northeast — distance from refineries"},
        {"country":"BR","region":"Mato Grosso","state":"MT","lat":-15.6,"lon":-56.1,
         "price":round(brl_base*1.19,4),"factor":"Agricultural interior, long haul distance"},
    ]

    ev_by_horizon  = compute_ev_by_horizon(prob_table)
    var_es_1m      = compute_var_es(ho, list(r_arr), HORIZON_DAYS["1M"])
    var_es_3m      = compute_var_es(ho, list(r_arr), HORIZON_DAYS["3M"])
    crack_history  = build_crack_spread_history(history, wti_history)

    # Contract schedule, KO barrier table, expiry distribution table
    send("Building contract schedule and derivative probability tables ...")
    ho_contracts     = get_ho_contract_schedule(ho, sig_daily)
    ko_default       = round(ho * 0.85, 4)
    ko_prob_rows     = compute_ko_probabilities(ho_contracts, ko_default)
    expiry_dist_rows = compute_expiry_distributions(ho_contracts, sig_daily)
    send("  Contract schedule: {} contracts  KO default ${:.4f}".format(
        len(ho_contracts), ko_default))

    # Save report
    report_path = os.path.join(run_dir,"ho_report_{}.txt".format(ts))
    with open(report_path,"w",encoding="utf-8") as f:
        f.write(summary+"\n\n")
        f.write("--- PROBABILITY TABLES ---\n")
        for h in HORIZONS:
            f.write("\n{}:\n".format(h))
            for b,p in prob_table[h]: f.write("  {:18s} {:.2%}\n".format(b,p))
        f.write("\n--- EV BY HORIZON ---\n")
        for h,ev in ev_by_horizon.items(): f.write("  {:4s}  EV=${:.4f}\n".format(h,ev))
        f.write("\n--- KO PROBABILITY TABLE (default KO=${:.4f}) ---\n".format(ko_default))
        for r in ko_prob_rows:
            f.write("  {:12s}  exp={}  fwd=${:.4f}  P(touch)={:.1f}%\n".format(
                r["label"], r["expiry"], r["fwd_price"], r["ko_prob"]))
    send("  Report saved")
    send("=== DONE ===")

    return {
        "agent":"ho","ho_price":ho,"history":history,"is_synthetic_history":is_synthetic_history,
        "returns":[round(float(r),6) for r in r_arr.tolist()],
        "market_data":{"HO":ho,"WTI":wti,"Brent":brent,"RBOB":rbob,"DXY":dxy,"VIX":vix,"crack_spread":crack},
        "eia_data":eia_data,"regime":regime,
        "prob_table":{h:list(prob_table[h]) for h in HORIZONS},
        "custom_bands":custom_by_horizon,"scenario_weights":weights,"summary":summary,
        "ci_bands":ci_bands,"vol_heatmap":vol_heatmap,"lognorm_shape":lognorm_shape,
        "drivers":drivers,"scenario_paths":scenario_paths,"regional_prices":regional_prices,
        "brazil_regional_prices":brazil_regional_prices,
        "ev_by_horizon":ev_by_horizon,"var_es":{"1M":var_es_1m,"3M":var_es_3m},
        "crack_history":crack_history,"run_dir":run_dir,
        # Contract schedule + derivative probability tables
        "ho_contracts":     ho_contracts,
        "ko_prob_rows":     ko_prob_rows,
        "ko_price_default": ko_default,
        "expiry_dist_rows": expiry_dist_rows,
        "scenario_signals":{
            "base_dynamic_drift_ann":round(base_dynamic_drift*252*100,2),
            "vix_vol_mult":round(vix_vol_mult,3),
            "vix_rolling_mean":round(vix_rolling_mean,2),
            "crack_signal_ann":round(crack_signal*252*100,2),
            "vix_signal_ann":round(vix_signal*252*100,2),
            "eia_signal_ann":round(eia_signal*252*100,2),
            "seasonal_signal_ann":round(seasonal_signal*252*100,2),
        },
    }


if __name__=="__main__":
    result=run()
    print(result["summary"])
