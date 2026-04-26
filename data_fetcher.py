#!/usr/bin/env python3
"""
data_fetcher.py — Multi-source waterfall for live commodity prices.
Shared by oil_agent_v2.py and ho_agent.py.

Waterfall order per ticker:
  1. Yahoo Finance JSON API  (real-time quote, ~15min delayed)
  2. Stooq CSV               (daily close, no API key)
  3. FRED API                (daily close, free, no key needed for public series)
  4. Barchart HTML scrape    (delayed quote)
  5. Hardcoded fallback      (raises a clear warning)

History (90-day OHLC for chart + vol model) is fetched via Stooq CSV
because it is the most stable free source for continuous futures history.
"""

import re, json, time, datetime, warnings
import urllib.request, urllib.error
from typing import Optional

warnings.filterwarnings("ignore")

# ── HEADERS ───────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ── TICKER MAP ────────────────────────────────────────────────────────────────
# Maps our internal name -> (yahoo_ticker, stooq_ticker, fred_series, barchart_slug)
# None means "not available from that source"

TICKER_MAP = {
    # name          yahoo      stooq      fred              barchart
    "HO":  ("HO=F",   "ho.f",    None,             "LO"),
    "WTI": ("CL=F",   "cl.f",    "DCOILWTICO",     "CL"),
    "Brent":("BZ=F",  "bz.f",    "DCOILBRENTEU",   "CB"),
    "RBOB":("RB=F",   "rb.f",    None,             "RB"),
    "DXY": ("DX-Y.NYB","dxy.b",  "DTWEXBGS",       None),
    "VIX": ("^VIX",   "^vix",    "VIXCLS",         None),
}

# Plausible price ranges for sanity-checking scraped values
_SANE_RANGES = {
    "HO":    (1.0,  15.0),
    "WTI":   (30.0, 200.0),
    "Brent": (30.0, 220.0),
    "RBOB":  (0.8,  10.0),
    "DXY":   (70.0, 140.0),
    "VIX":   (8.0,  90.0),
}

# ── LOW-LEVEL HTTP ────────────────────────────────────────────────────────────

def _get(url, timeout=12, extra_headers=None):
    h = dict(_HEADERS)
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise ConnectionError(str(e))

def _sane(name, value):
    lo, hi = _SANE_RANGES.get(name, (0, 1e9))
    return lo <= value <= hi

# ── SOURCE 1: YAHOO FINANCE JSON API ─────────────────────────────────────────

def _yahoo_live(name):
    """
    Hits Yahoo's chart endpoint with interval=1m&range=1d.
    Returns regularMarketPrice — this is the true current price,
    not a stale session close.
    """
    yahoo_ticker = TICKER_MAP[name][0]
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        "{}?interval=1m&range=1d&includePrePost=false".format(
            urllib.request.quote(yahoo_ticker)
        )
    )
    raw = _get(url, extra_headers={"Accept": "application/json"})
    data = json.loads(raw)
    result = data["chart"]["result"][0]
    meta = result["meta"]

    # regularMarketPrice is always the live current price
    price = meta.get("regularMarketPrice")
    ts    = meta.get("regularMarketTime", 0)
    dt    = datetime.datetime.fromtimestamp(ts) if ts else None

    if price is None:
        raise ValueError("No regularMarketPrice in Yahoo response")
    price = float(price)
    if not _sane(name, price):
        raise ValueError("Yahoo price {:.4f} out of sane range for {}".format(price, name))
    return price, dt, "Yahoo Finance (live)"


# ── SOURCE 2: STOOQ LIVE QUOTE ────────────────────────────────────────────────

def _stooq_live(name):
    """
    Stooq's quote page returns a simple CSV with the latest price.
    More reliable than yfinance for futures rolls.
    """
    stooq_ticker = TICKER_MAP[name][1]
    if stooq_ticker is None:
        raise ValueError("No stooq ticker for {}".format(name))
    url = "https://stooq.com/q/l/?s={}&f=sd2t2ohlcv&h&e=csv".format(stooq_ticker)
    raw = _get(url)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # Header: Symbol,Date,Time,Open,High,Low,Close,Volume
    if len(lines) < 2:
        raise ValueError("Stooq returned no data for {}".format(name))
    parts = lines[1].split(",")
    if len(parts) < 7:
        raise ValueError("Stooq bad CSV format for {}".format(name))
    close = float(parts[6])
    date_str = parts[1]
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
    if not _sane(name, close):
        raise ValueError("Stooq price {:.4f} out of sane range for {}".format(close, name))
    return close, dt, "Stooq (daily close)"


# ── SOURCE 3: FRED API ────────────────────────────────────────────────────────

def _fred_live(name):
    """
    FRED (St. Louis Fed) provides daily spot prices for WTI, Brent, VIX, DXY.
    No API key required for public series. Returns latest observation.
    """
    series = TICKER_MAP[name][2]
    if series is None:
        raise ValueError("No FRED series for {}".format(name))
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id={}&vintage_date={}".format(series, datetime.date.today())
    )
    raw = _get(url)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    # Last non-"." row
    for line in reversed(lines[1:]):
        parts = line.split(",")
        if len(parts) == 2 and parts[1] not in (".", ""):
            try:
                price = float(parts[1])
                dt = datetime.datetime.strptime(parts[0], "%Y-%m-%d")
                if not _sane(name, price):
                    raise ValueError("FRED price {:.4f} out of sane range".format(price))
                return price, dt, "FRED (St. Louis Fed)"
            except ValueError:
                continue
    raise ValueError("FRED returned no valid price for {}".format(name))


# ── SOURCE 4: BARCHART HTML SCRAPE ───────────────────────────────────────────

def _barchart_live(name):
    """
    Scrapes Barchart's futures quote page for the last price.
    Barchart shows 10-min delayed quotes for free.
    """
    slug = TICKER_MAP[name][3]
    if slug is None:
        raise ValueError("No Barchart slug for {}".format(name))
    url = "https://www.barchart.com/futures/quotes/{}*0/overview".format(slug)
    raw = _get(url)
    # Look for lastPrice in their JSON data blob
    m = re.search(r'"lastPrice"\s*:\s*"?([\d.]+)"?', raw)
    if not m:
        # Try og:description meta tag which often has the price
        m = re.search(r'<meta[^>]+og:description[^>]+content="[^"]*?([\d]+\.[\d]+)[^"]*"', raw)
    if not m:
        raise ValueError("Could not parse Barchart price for {}".format(name))
    price = float(m.group(1))
    if not _sane(name, price):
        raise ValueError("Barchart price {:.4f} out of sane range".format(price))
    return price, datetime.datetime.now(), "Barchart (10min delayed)"


# ── HISTORY: STOOQ 90-DAY CSV ────────────────────────────────────────────────

def fetch_history(name, days=90):
    """
    Fetch daily OHLC history from Stooq.
    Returns list of {"date": "YYYY-MM-DD", "price": float}.
    Falls back to synthetic series if all sources fail.
    """
    stooq_ticker = TICKER_MAP[name][1]
    if stooq_ticker is None:
        return _synthetic_history(name, days)

    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days + 20)  # buffer for weekends
    url = (
        "https://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d".format(
            stooq_ticker,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
        )
    )
    try:
        raw = _get(url)
        lines = raw.strip().splitlines()
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    price = float(parts[4])  # Close
                    if _sane(name, price):
                        rows.append({"date": parts[0], "price": round(price, 4)})
                except (ValueError, IndexError):
                    pass
        if len(rows) >= 10:
            return rows[-days:]
    except Exception as e:
        pass  # fall through to synthetic

    return _synthetic_history(name, days)


def _synthetic_history(name, days):
    """Last-resort synthetic history centered on a reasonable base price."""
    import numpy as np
    bases = {"HO": 3.80, "WTI": 97.0, "Brent": 102.0,
             "RBOB": 3.20, "DXY": 103.5, "VIX": 25.0}
    base  = bases.get(name, 50.0)
    vol   = {"HO": 0.018, "WTI": 0.014, "Brent": 0.013,
             "RBOB": 0.018, "DXY": 0.004, "VIX": 0.035}.get(name, 0.012)

    end = datetime.date.today()
    rows, price = [], base
    for i in range(days + 30, -1, -1):
        d = end - datetime.timedelta(days=i)
        if d.weekday() < 5:
            price = max(0.1, price * (1 + float(np.random.normal(0, vol))))
            rows.append({"date": str(d), "price": round(price, 4)})
    return rows[-days:]


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def fetch_price(name, send=print):
    """
    Waterfall: try each source in order, return first valid result.
    Returns (price, datetime_or_None, source_name).
    Raises RuntimeError only if ALL sources fail.
    """
    sources = [
        ("Yahoo Finance",  _yahoo_live),
        ("Stooq",          _stooq_live),
        ("FRED",           _fred_live),
        ("Barchart",       _barchart_live),
    ]
    errors = []
    for source_name, fn in sources:
        try:
            price, dt, label = fn(name)
            age = ""
            if dt:
                delta = datetime.datetime.now() - dt
                age = " | age: {}".format(_fmt_age(delta))
            send("  ✓ {} = {:.4f}  [{}{}]".format(name, price, label, age))
            return price, dt, label
        except Exception as e:
            errors.append("  {} → {}".format(source_name, e))
            time.sleep(0.3)

    # All failed — log errors and raise
    send("  ✗ ALL sources failed for {}:".format(name))
    for err in errors:
        send(err)
    raise RuntimeError("Could not fetch price for {} from any source".format(name))


def fetch_all(names=None, send=print, history_days=90):
    """
    Fetch live prices + history for all requested tickers.
    Returns dict: name -> {"current", "dt", "source", "history", "returns"}
    """
    import numpy as np

    if names is None:
        names = list(TICKER_MAP.keys())

    result = {}
    send("Fetching live prices (waterfall) ...")

    for name in names:
        if name not in TICKER_MAP:
            send("  [SKIP] Unknown ticker: {}".format(name))
            continue
        try:
            price, dt, source = fetch_price(name, send=send)
            history = fetch_history(name, days=history_days)

            # Override last history point with the live price so charts are current
            if history:
                today_str = str(datetime.date.today())
                if history[-1]["date"] == today_str:
                    history[-1]["price"] = price
                else:
                    history.append({"date": today_str, "price": price})

            closes = [r["price"] for r in history]
            returns = list(np.diff(np.log(closes))) if len(closes) > 1 else []

            result[name] = {
                "current": price,
                "dt":      dt,
                "source":  source,
                "history": history,
                "returns": returns,
            }
        except RuntimeError as e:
            send("  [WARN] Skipping {}: {}".format(name, e))

    return result


def _fmt_age(delta):
    secs = int(delta.total_seconds())
    if secs < 0:
        return "unknown"
    if secs < 60:
        return "{}s".format(secs)
    if secs < 3600:
        return "{}m".format(secs // 60)
    if secs < 86400:
        return "{}h".format(secs // 3600)
    return "{}d".format(secs // 86400)


# ── CLI TEST ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing data waterfall — {}\n".format(datetime.datetime.now()))
    data = fetch_all()
    print("\n--- SUMMARY ---")
    for name, d in data.items():
        dt_str = d["dt"].strftime("%Y-%m-%d %H:%M") if d["dt"] else "?"
        print("  {:8s}  {:>10.4f}  {}  [{}]  history={}pts".format(
            name, d["current"], dt_str, d["source"], len(d["history"])))
