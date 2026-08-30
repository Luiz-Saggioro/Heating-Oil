#!/usr/bin/env python3
"""
data_fetcher.py — Multi-source waterfall for live commodity prices.

Waterfall order (live spot price):
  1. yfinance library      — handles Yahoo auth cookies internally, most reliable
  2. Yahoo Finance JSON v8 — direct API call with browser headers
  3. FRED CSV              — daily close for WTI/Brent/VIX/DXY only

History (daily OHLC for chart + vol model):
  1. yfinance .history()   — primary, real OHLC data going back years
  2. Yahoo Finance v8 JSON — secondary
  3. Synthetic fallback     — CLEARLY FLAGGED, never silent

Intraday history (1h / 1d intervals for short-term chart):
  1. yfinance .history()   — primary, supports 1h up to 730d lookback
  2. Synthetic fallback     — CLEARLY FLAGGED, never silent

v2.2 fix: _yfinance_intraday accepts since_dt to hard-filter by timestamp,
          allowing true single-session isolation for the "1 Hour" view.

All data is flagged with its source and age in the run log.
"""

import re, json, time, datetime, warnings
import urllib.request, urllib.error
from typing import Optional

warnings.filterwarnings("ignore")

# ── TICKER MAP ────────────────────────────────────────────────────────────────
# name -> (yfinance_ticker, yahoo_v8_ticker, fred_series)
TICKER_MAP = {
    "HO":    ("HO=F",      "HO=F",      None          ),
    "WTI":   ("CL=F",      "CL=F",      "DCOILWTICO"  ),
    "Brent": ("BZ=F",      "BZ=F",      "DCOILBRENTEU"),
    "RBOB":  ("RB=F",      "RB=F",      None          ),
    "DXY":   ("DX-Y.NYB",  "DX-Y.NYB",  "DTWEXBGS"    ),
    "VIX":   ("^VIX",      "%5EVIX",    "VIXCLS"      ),
    "OVX":   ("^OVX",      "%5EOVX",    None          ),   # CBOE Crude Oil ETF Vol Index
}

# Sanity ranges — reject obviously wrong scraped values
_SANE = {
    "HO":    (1.0,   15.0),
    "WTI":   (30.0, 250.0),
    "Brent": (30.0, 260.0),
    "RBOB":  (0.5,   15.0),
    "DXY":   (70.0, 150.0),
    "VIX":   (8.0,  100.0),
    "OVX":   (5.0,  150.0),   # OVX typically ranges 20–80 in normal markets
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

def _sane(name, value):
    lo, hi = _SANE.get(name, (0, 1e12))
    return lo <= float(value) <= hi

def _get(url, timeout=15, extra=None):
    h = dict(_HEADERS)
    if extra:
        h.update(extra)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


# ── SOURCE 1: yfinance ────────────────────────────────────────────────────────

def _yfinance_live(name):
    """Use yfinance library — handles Yahoo cookie auth internally."""
    import yfinance as yf
    ticker_sym = TICKER_MAP[name][0]
    t = yf.Ticker(ticker_sym)
    info = t.fast_info
    price = getattr(info, "last_price", None)
    if price is None:
        price = getattr(info, "previous_close", None)
    if price is None or not _sane(name, price):
        raise ValueError(f"yfinance returned invalid price {price} for {name}")
    return float(price), datetime.datetime.now(), "yfinance (Yahoo Finance)"


def _yfinance_history(name, days):
    """Fetch real daily OHLC history via yfinance."""
    import yfinance as yf
    ticker_sym = TICKER_MAP[name][0]
    period_days = days + 30
    if period_days <= 30:    period = "1mo"
    elif period_days <= 90:  period = "3mo"
    elif period_days <= 180: period = "6mo"
    elif period_days <= 365: period = "1y"
    else:                    period = "2y"
    t   = yf.Ticker(ticker_sym)
    df  = t.history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance returned empty history for {name}")
    rows = []
    for idx, row in df.iterrows():
        date_str = str(idx.date())
        close    = float(row["Close"])
        if _sane(name, close):
            rows.append({"date": date_str, "price": round(close, 4)})
    if len(rows) < 5:
        raise ValueError(f"yfinance history too short for {name}: {len(rows)} rows")
    return rows[-days:]


# ── INTRADAY HISTORY (1h / 1d intervals) ─────────────────────────────────────

def _yfinance_intraday(name, interval, lookback_days, since_dt=None):
    """
    Fetch intraday OHLC via yfinance.

    interval:      '1h' (hourly) | '1d' (daily)
    lookback_days: calendar days passed to yfinance period bucket.
    since_dt:      optional datetime — rows with datetime < since_dt are dropped
                   AFTER fetching. Use this to isolate a single trading session
                   (e.g. since_dt = today 00:00 for the "1 Hour" view).

    Returns list of {datetime, price} dicts, newest last.
    datetime key is ISO string: 'YYYY-MM-DDTHH:MM' for 1h, 'YYYY-MM-DD' for 1d.
    """
    import yfinance as yf
    ticker_sym = TICKER_MAP[name][0]
    days_use   = min(lookback_days, 729)
    if days_use <= 7:     period = "7d"
    elif days_use <= 30:  period = "1mo"
    elif days_use <= 60:  period = "60d"
    elif days_use <= 90:  period = "3mo"
    elif days_use <= 180: period = "6mo"
    elif days_use <= 365: period = "1y"
    else:                 period = "2y"
    t  = yf.Ticker(ticker_sym)
    df = t.history(period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"yfinance intraday returned empty for {name} interval={interval}")
    rows = []
    for idx, row in df.iterrows():
        close = float(row["Close"])
        if _sane(name, close):
            ts_str = (idx.strftime("%Y-%m-%dT%H:%M")
                      if interval == "1h" else str(idx.date()))
            rows.append({"datetime": ts_str, "price": round(close, 4)})
    if len(rows) < 2:
        raise ValueError(f"yfinance intraday too short for {name}: {len(rows)} rows")
    # Hard timestamp filter — used to isolate a single session (e.g. "1 Hour" view)
    if since_dt is not None:
        since_str = since_dt.strftime("%Y-%m-%dT%H:%M")
        rows = [r for r in rows if r["datetime"] >= since_str]
    # Fallback trim by lookback
    max_pts = lookback_days * 24 if interval == "1h" else lookback_days
    return rows[-max_pts:]


def _synthetic_intraday(name, interval, lookback_days, send=print):
    """
    Last-resort synthetic intraday — always logs a loud warning.
    Returned rows carry 'synthetic': True so the UI can warn the user.
    """
    import numpy as np
    send(f"  *** WARNING: SYNTHETIC intraday history for {name} [{interval}]. NOT real data. ***")
    bases = {"HO": 3.50, "WTI": 75.0, "Brent": 79.0,
             "RBOB": 2.40, "DXY": 103.5, "VIX": 18.0}
    vols  = {"HO": 0.003, "WTI": 0.003, "Brent": 0.003,
             "RBOB": 0.003, "DXY": 0.001, "VIX": 0.008}
    base = bases.get(name, 50.0)
    vol  = vols.get(name, 0.003)
    rng  = __import__("numpy").random.default_rng(
        seed=int(datetime.datetime.now().strftime("%Y%m%d%H")))
    now  = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    step = (datetime.timedelta(hours=1)
            if interval == "1h" else datetime.timedelta(days=1))
    total_steps = lookback_days * (24 if interval == "1h" else 1)
    rows, price = [], base
    cur = now - step * total_steps
    while cur <= now:
        if cur.weekday() < 5:
            price = max(0.5, price * (1 + float(rng.normal(0, vol))))
            ts_str = (cur.strftime("%Y-%m-%dT%H:%M")
                      if interval == "1h" else str(cur.date()))
            rows.append({"datetime": ts_str, "price": round(price, 4), "synthetic": True})
        cur += step
    return rows


def fetch_intraday_history(name, interval="1h", lookback_days=7, since_dt=None, send=print):
    """
    Fetch short-term intraday history for the price chart.

    interval:      '1h' | '1d'
    lookback_days: calendar days to look back
    since_dt:      optional datetime lower-bound filter (isolates a single session)

    Returns (rows, is_synthetic).
    rows is a list of {datetime, price} dicts; datetime is an ISO string.
    """
    try:
        rows = _yfinance_intraday(name, interval, lookback_days, since_dt=since_dt)
        send(f"  Intraday {name} [{interval}]: {len(rows)} pts "
             f"({rows[0]['datetime']} -> {rows[-1]['datetime']})")
        return rows, False
    except Exception as e:
        send(f"  [WARN] Intraday {name} [{interval}] failed: {e} — using synthetic")
        return _synthetic_intraday(name, interval, lookback_days, send=send), True


# ── SOURCE 2: Yahoo Finance v8 JSON (direct, no library) ─────────────────────

def _yahoo_v8_live(name):
    ticker = TICKER_MAP[name][1]
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.request.quote(ticker)}?interval=1m&range=1d&includePrePost=false"
    )
    raw  = _get(url, extra={"Referer": "https://finance.yahoo.com/"})
    data = json.loads(raw)
    meta = data["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice")
    ts    = meta.get("regularMarketTime", 0)
    dt    = datetime.datetime.fromtimestamp(ts) if ts else None
    if price is None or not _sane(name, float(price)):
        raise ValueError(f"Yahoo v8 invalid price {price} for {name}")
    return float(price), dt, "Yahoo Finance v8"


def _yahoo_v8_history(name, days):
    ticker = TICKER_MAP[name][1]
    end_ts   = int(datetime.datetime.now().timestamp())
    start_ts = int((datetime.datetime.now() - datetime.timedelta(days=days + 30)).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.request.quote(ticker)}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )
    raw  = _get(url, extra={"Referer": "https://finance.yahoo.com/"})
    data = json.loads(raw)
    result     = data["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    closes     = result["indicators"]["quote"][0].get("close", [])
    rows = []
    for ts, c in zip(timestamps, closes):
        if c is None:
            continue
        date_str = str(datetime.datetime.fromtimestamp(ts).date())
        if _sane(name, float(c)):
            rows.append({"date": date_str, "price": round(float(c), 4)})
    if len(rows) < 5:
        raise ValueError(f"Yahoo v8 history too short for {name}")
    return rows[-days:]


# ── SOURCE 3: FRED CSV ────────────────────────────────────────────────────────

def _fred_live(name):
    series = TICKER_MAP[name][2]
    if series is None:
        raise ValueError(f"No FRED series for {name}")
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series}&vintage_date={datetime.date.today()}"
    )
    raw   = _get(url)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    for line in reversed(lines[1:]):
        parts = line.split(",")
        if len(parts) == 2 and parts[1] not in (".", ""):
            try:
                price = float(parts[1])
                if _sane(name, price):
                    dt = datetime.datetime.strptime(parts[0], "%Y-%m-%d")
                    return price, dt, f"FRED ({series})"
            except ValueError:
                pass
    raise ValueError(f"FRED: no valid recent data for {name}")


# ── SYNTHETIC DAILY FALLBACK ──────────────────────────────────────────────────

def _synthetic_history(name, days, send=print):
    """Last-resort synthetic daily history — always logs a loud warning."""
    import numpy as np
    send(f"  *** WARNING: SYNTHETIC daily history for {name}. NOT real data. ***")
    bases = {"HO": 3.50, "WTI": 75.0, "Brent": 79.0,
             "RBOB": 2.40, "DXY": 103.5, "VIX": 18.0}
    vols  = {"HO": 0.012, "WTI": 0.015, "Brent": 0.015,
             "RBOB": 0.013, "DXY": 0.004, "VIX": 0.04}
    base  = bases.get(name, 50.0)
    vol   = vols.get(name, 0.012)
    rng   = np.random.default_rng(seed=42)
    rows, price = [], base
    for i in range(days, 0, -1):
        d = datetime.date.today() - datetime.timedelta(days=i)
        if d.weekday() < 5:
            price = max(0.5, price * (1 + float(rng.normal(0, vol))))
            rows.append({"date": str(d), "price": round(price, 4), "synthetic": True})
    return rows


# ── MAIN FETCH FUNCTIONS ──────────────────────────────────────────────────────

def fetch_price(name, send=print):
    """
    Try each live price source in order.
    Returns (price, datetime, source_label).
    Raises RuntimeError if all sources fail.
    """
    if name not in TICKER_MAP:
        raise ValueError(f"Unknown ticker: {name}")
    sources = [
        ("yfinance",     lambda: _yfinance_live(name)),
        ("Yahoo v8",     lambda: _yahoo_v8_live(name)),
        ("FRED",         lambda: _fred_live(name)),
    ]
    errors = []
    for label, fn in sources:
        try:
            price, dt, src = fn()
            age = datetime.datetime.now() - dt if dt else None
            age_str = _fmt_age(age) if age else "unknown"
            send(f"  OK {name} = {price:.4f}  [{src} | age: {age_str}]")
            return price, dt, src
        except Exception as e:
            errors.append(f"    {label}: {e}")
            time.sleep(0.2)
    send(f"  FAILED all sources for {name}:")
    for err in errors:
        send(err)
    raise RuntimeError(f"Could not fetch live price for {name}")


def fetch_history(name, days=365, send=print):
    """
    Try each history source in order.
    Returns list of {date, price} dicts, newest last.
    Falls back to synthetic with loud warning if all fail.
    """
    sources = [
        ("yfinance history",   lambda: _yfinance_history(name, days)),
        ("Yahoo v8 history",   lambda: _yahoo_v8_history(name, days)),
    ]
    for label, fn in sources:
        try:
            rows = fn()
            send(f"  History {name}: {len(rows)} pts via {label} "
                 f"({rows[0]['date']} -> {rows[-1]['date']})")
            return rows
        except Exception as e:
            send(f"  History {label} failed for {name}: {e}")
            time.sleep(0.2)
    return _synthetic_history(name, days, send=send)


def fetch_all(names=None, send=print, history_days=365):
    """
    Fetch live prices + daily history for all tickers.
    Returns dict: name -> {current, dt, source, history, returns, is_synthetic}
    """
    import numpy as np
    if names is None:
        names = list(TICKER_MAP.keys())
    result = {}
    send("Fetching live prices ...")
    for name in names:
        if name not in TICKER_MAP:
            send(f"  [SKIP] Unknown ticker: {name}")
            continue
        try:
            price, dt, source = fetch_price(name, send=send)
        except RuntimeError as e:
            send(f"  [WARN] Skipping {name}: {e}")
            continue
        history = fetch_history(name, days=history_days, send=send)
        is_synthetic = any(r.get("synthetic") for r in history)
        # Pin today's live price as the last data point
        today_str = str(datetime.date.today())
        if history and history[-1]["date"] == today_str:
            history[-1]["price"] = price
        else:
            history.append({"date": today_str, "price": price})
        closes  = [r["price"] for r in history]
        returns = list(np.diff(np.log(closes))) if len(closes) > 1 else []
        result[name] = {
            "current":      price,
            "dt":           dt,
            "source":       source,
            "history":      history,
            "returns":      returns,
            "is_synthetic": is_synthetic,
        }
    return result


def _fmt_age(delta):
    secs = int(delta.total_seconds())
    if secs < 0:     return "unknown"
    if secs < 60:    return f"{secs}s"
    if secs < 3600:  return f"{secs//60}m"
    if secs < 86400: return f"{secs//3600}h"
    return f"{secs//86400}d"


# ── CLI TEST ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Testing data fetcher — {datetime.datetime.now()}\n")
    data = fetch_all(["HO", "WTI"])
    print("\n--- SUMMARY ---")
    for name, d in data.items():
        synth = " *** SYNTHETIC ***" if d["is_synthetic"] else ""
        dt_str = d["dt"].strftime("%Y-%m-%d %H:%M") if d["dt"] else "?"
        print(f"  {name:8s}  {d['current']:>10.4f}  {dt_str}  [{d['source']}]  "
              f"history={len(d['history'])}pts{synth}")
    print("\n--- INTRADAY TEST ---")
    # 1 Hour: today's session only
    today_start = datetime.datetime.combine(datetime.date.today(), datetime.time(0, 0))
    rows, synth = fetch_intraday_history("HO", interval="1h", lookback_days=7,
                                         since_dt=today_start)
    print(f"  HO 1h (today only): {len(rows)} pts  synthetic={synth}")
    if rows:
        print(f"  First: {rows[0]}  Last: {rows[-1]}")
    # 1 Day: last 2 calendar days
    rows, synth = fetch_intraday_history("HO", interval="1h", lookback_days=2)
    print(f"  HO 1h (2d lookback): {len(rows)} pts  synthetic={synth}")
    # 1 Week: last 7 calendar days
    rows, synth = fetch_intraday_history("HO", interval="1h", lookback_days=7)
    print(f"  HO 1h (7d lookback): {len(rows)} pts  synthetic={synth}")
