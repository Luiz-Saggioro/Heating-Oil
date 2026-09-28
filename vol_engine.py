"""
vol_engine.py — HO futures curve, CME settlement implied vol, realized vol,
calendar spreads, Brazil import-parity (PPI) and Chicago basis helpers.

Implied-vol source waterfall (per user decision):
  1. CME settlement data (auto, best-effort): option settlement premiums from
     cmegroup.com, inverted with Black-76 -> settlement vol per strike.
  2. Latest CME vol file uploaded by an admin (persisted in the forecast store).
  3. Proxy: OVX x (HO realized vol / WTI realized vol) — ALWAYS labeled "Proxy".

Futures curve waterfall:
  1. yfinance individual contracts (e.g. HOX26.NYM)
  2. CME futures settlements endpoint
  3. Flat curve at front price — flagged "estimated" (no invented seasonality)
"""
from __future__ import annotations
import calendar, datetime, re, time
import numpy as np
import pandas as pd
import config as cfg

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
               7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_TO_MONTH = {v: k for k, v in MONTH_CODES.items()}
MONTH_ABBR = [calendar.month_abbr[i] for i in range(1, 13)]

_CME = "https://www.cmegroup.com/CmeWS/mvc"
_CME_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cmegroup.com/markets/energy/refined-products/heating-oil.settlements.options.html",
}


# ── Contract calendar ────────────────────────────────────────────────────────
def last_biz_day(y, m):
    d = datetime.date(y, m, calendar.monthrange(y, m)[1])
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d


def contract_calendar(today=None, n=None):
    """HO futures expire on the last business day of the month PRIOR to delivery."""
    today = today or datetime.date.today()
    n = n or cfg.get("CONTRACT_COUNT")
    off = cfg.get("OPTION_EXPIRY_BDAYS_BEFORE_FUT")
    y, m = today.year, today.month
    for _ in range(14):
        ey, em = (y - 1, 12) if m == 1 else (y, m - 1)
        if last_biz_day(ey, em) >= today:
            break
        m += 1
        if m > 12:
            m, y = 1, y + 1
    out = []
    for _ in range(n):
        ey, em = (y - 1, 12) if m == 1 else (y, m - 1)
        fx = last_biz_day(ey, em)
        ox = pd.Timestamp(np.busday_offset(np.datetime64(fx), -off, roll="backward")).date()
        t_cal = max(1, (fx - today).days)
        out.append({
            "label": f"{MONTH_ABBR[m - 1]} {y}",
            "code": f"HO{MONTH_CODES[m]}{str(y)[-2:]}",
            "yahoo": f"HO{MONTH_CODES[m]}{str(y)[-2:]}.NYM",
            "month": m, "year": y,
            "expiry_date": str(fx), "option_expiry": str(ox),
            "t_cal": t_cal, "t_days": max(1, int(np.busday_count(today, fx))),
            "t_opt_years": max(1, (ox - today).days) / 365.0,
        })
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def parse_contract(s, cal=None):
    """Map many spellings to a calendar label: HOX26, X26, X6, NOV26, Nov 2026, NOV 26, 2026-11."""
    s = str(s).strip().upper()
    y = m = None
    mt = re.match(r"^(?:HO|OH)?([FGHJKMNQUVXZ])(\d{1,2})$", s)
    if mt:
        m = CODE_TO_MONTH[mt.group(1)]
        yy = mt.group(2)
        base = datetime.date.today().year
        y = (base // 10 * 10 + int(yy)) if len(yy) == 1 else 2000 + int(yy)
        if len(yy) == 1 and y < base - 1:
            y += 10
    else:
        mt = re.match(r"^([A-Z]{3})[\s\-']*(\d{2,4})$", s)
        if mt and mt.group(1).title() in MONTH_ABBR:
            m = MONTH_ABBR.index(mt.group(1).title()) + 1
            y = int(mt.group(2)) + (2000 if len(mt.group(2)) == 2 else 0)
        else:
            mt = re.match(r"^(\d{4})[\-/](\d{1,2})", s)
            if mt:
                y, m = int(mt.group(1)), int(mt.group(2))
    if not y or not m:
        return None
    return f"{MONTH_ABBR[m - 1]} {y}"


# ── Futures curve ────────────────────────────────────────────────────────────
def _yf_curve(cal, send):
    import yfinance as yf
    tickers = [c["yahoo"] for c in cal]
    df = yf.download(tickers, period="1y", interval="1d", progress=False,
                     auto_adjust=True, group_by="ticker", threads=True)
    out = {}
    for c in cal:
        try:
            s = df[c["yahoo"]]["Close"].dropna() if isinstance(df.columns, pd.MultiIndex) \
                else df["Close"].dropna()
        except Exception:
            continue
        s = s[(s > 0.5) & (s < 15)]
        if len(s):
            out[c["label"]] = [{"date": str(i.date()), "price": round(float(v), 4)} for i, v in s.items()]
    send(f"  Curve via yfinance: {len(out)}/{len(cal)} contracts")
    return out


def _cme_json(url, timeout=10):
    import requests
    r = requests.get(url, headers=_CME_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _recent_trade_dates(n=5):
    d, out = datetime.date.today(), []
    while len(out) < n:
        d -= datetime.timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("'", "").rstrip("ABab"))
    except (TypeError, ValueError):
        return None


def _cme_futures_curve(cal, send):
    pid = cfg.get("CME_HO_FUTURES_PRODUCT_ID")
    for td in _recent_trade_dates():
        try:
            js = _cme_json(f"{_CME}/Settlements/Futures/Settlements/{pid}/FUT"
                           f"?tradeDate={td:%m/%d/%Y}&strategy=DEFAULT")
            out = {}
            for row in js.get("settlements", []):
                lbl = parse_contract(row.get("month", ""))
                px = _num(row.get("settle"))
                if lbl and px:
                    out[lbl] = [{"date": str(td), "price": px}]
            if out:
                send(f"  Curve via CME settlements {td}: {len(out)} contracts")
                return out
        except Exception as e:
            send(f"  [CME fut] {td}: {type(e).__name__}")
            break
    return {}


def fetch_curve(front_price, send=print):
    """Returns (contracts, source). contracts carry fwd_price, history, price_source."""
    cal = contract_calendar()
    hist, src = {}, "estimated (flat at front)"
    try:
        hist = _yf_curve(cal, send)
        if len(hist) >= max(2, len(cal) // 2):
            src = "yfinance (NYMEX contracts)"
    except Exception as e:
        send(f"  [WARN] yfinance curve failed: {type(e).__name__}: {e}")
    if len(hist) < max(2, len(cal) // 2) and cfg.get("CME_AUTO_FETCH"):
        cme = _cme_futures_curve(cal, send)
        if len(cme) > len(hist):
            hist, src = cme, "CME settlements"
    out = []
    for c in cal:
        h = hist.get(c["label"], [])
        c = dict(c)
        if h:
            c["fwd_price"], c["price_source"] = h[-1]["price"], src
        else:
            c["fwd_price"], c["price_source"] = round(float(front_price), 4), "estimated"
        c["history"] = h
        out.append(c)
    return out, src


# ── Realized vol ─────────────────────────────────────────────────────────────
def realized_vol(prices, window):
    p = np.asarray([x for x in prices if x and x > 0], dtype=float)
    if len(p) < window + 1:
        return None
    r = np.diff(np.log(p))[-window:]
    return float(np.std(r, ddof=1) * np.sqrt(252))


def rolling_realized(hist_rows, window):
    if len(hist_rows) < window + 2:
        return pd.DataFrame(columns=["date", "rv"])
    df = pd.DataFrame(hist_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["rv"] = np.log(df["price"].astype(float)).diff().rolling(window).std() * np.sqrt(252)
    return df.dropna(subset=["rv"])[["date", "rv"]]


# ── Black-76 ─────────────────────────────────────────────────────────────────
def black76(F, K, T, r, sig, cp):
    from scipy.stats import norm
    if T <= 0 or sig <= 0:
        return max(0.0, (F - K) if cp == "C" else (K - F)) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    df = np.exp(-r * T)
    if cp == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, cp):
    from scipy.optimize import brentq
    intrinsic = max(0.0, (F - K) if cp == "C" else (K - F)) * np.exp(-r * T)
    if not price or price <= intrinsic + 1e-6 or T <= 0:
        return None
    try:
        return float(brentq(lambda s: black76(F, K, T, r, s, cp) - price, 1e-3, 5.0, xtol=1e-6))
    except Exception:
        return None


# ── CME option settlements (auto, best-effort) ───────────────────────────────
def _discover_option_pid(send):
    pid = cfg.get("CME_HO_OPTION_PRODUCT_ID")
    if pid:
        return pid
    fut = cfg.get("CME_HO_FUTURES_PRODUCT_ID")
    try:
        js = _cme_json(f"{_CME}/Options/Categories/List/{fut}/G")
        stack = [js]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k in ("optionProductId", "productId", "id"):
                    if k in x and str(x[k]).isdigit() and str(x[k]) != str(fut):
                        send(f"  [CME opt] discovered option product id {x[k]}")
                        return str(x[k])
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    except Exception as e:
        send(f"  [CME opt] discovery failed: {type(e).__name__}")
    return ""


def _walk_rows(js):
    stack, rows = [js], []
    while stack:
        x = stack.pop()
        if isinstance(x, list) and x and isinstance(x[0], dict) and "strike" in x[0]:
            rows.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return rows


def fetch_cme_surface(contracts, rate, send=print, max_months=None):
    """Return (surface_df, trade_date) or (empty, None). Never raises."""
    if not cfg.get("CME_AUTO_FETCH"):
        return pd.DataFrame(), None
    pid = _discover_option_pid(send)
    if not pid:
        return pd.DataFrame(), None
    max_months = max_months or len(contracts)
    deadline = time.monotonic() + 40                      # hard time budget for the whole fetch
    for td in _recent_trade_dates(3):
        recs = []
        for i, c in enumerate(contracts[:max_months]):
            if time.monotonic() > deadline or (i >= 1 and not recs):
                break                                     # blocked / wrong format -> give up fast
            L, yy = c["code"][2], c["code"][-2:]
            for my in (f"OH{L}{yy}", f"OH{L}{yy[-1]}", f"{L}{yy}"):
                try:
                    js = _cme_json(f"{_CME}/Settlements/Options/Settlements/{pid}/OOF"
                                   f"?monthYear={my}&optionProductId={pid}"
                                   f"&strategy=DEFAULT&tradeDate={td:%m/%d/%Y}")
                except Exception:
                    continue
                rows = _walk_rows(js)
                if not rows:
                    continue
                F = c["fwd_price"]
                for rw in rows:
                    K = _num(rw.get("strike"))
                    if not K:
                        continue
                    K = K / 100 if K > 20 else K               # CME quotes strikes in cents
                    typ = str(rw.get("type", rw.get("putCall", ""))).upper()[:1]
                    pairs = [(typ, rw.get("settle"))] if typ in ("C", "P") else \
                            [("C", rw.get("callSettle")), ("P", rw.get("putSettle"))]
                    for cp, px in pairs:
                        px = _num(px)
                        if cp not in ("C", "P") or not px:
                            continue
                        iv = implied_vol(px, F, K, c["t_opt_years"], rate, cp)
                        if iv:
                            recs.append({"contract": c["label"], "strike": K, "cp": cp,
                                         "premium": px, "futures": F, "iv": iv})
                break
        if recs:
            send(f"  [CME opt] {len(recs)} settlement vols for {td}")
            return pd.DataFrame(recs), str(td)
        send(f"  [CME opt] no option settlements for {td}")
        break
    return pd.DataFrame(), None


# ── Uploaded CME vol file ────────────────────────────────────────────────────
_COLS = {
    "contract": ("contract", "month", "expiry", "contract month", "symbol", "series"),
    "strike":   ("strike", "strike price", "k"),
    "cp":       ("type", "cp", "put/call", "call/put", "option type", "pc"),
    "iv":       ("vol", "iv", "implied vol", "implied volatility", "settle vol",
                 "settlement vol", "settlement volatility", "atm vol", "volatility"),
    "premium":  ("settle", "premium", "price", "settlement", "settle price"),
    "futures":  ("futures", "underlying", "futures price", "fut", "underlying price"),
}


def parse_vol_upload(raw: pd.DataFrame, contracts, rate):
    """Accept a CME/QuikStrike export. Needs a contract column plus either a vol column
    (in % or decimal) or a premium column (inverted with Black-76). Strike optional
    (missing strike = ATM vol for that contract)."""
    cols = {c.lower().strip(): c for c in raw.columns}
    pick = {k: next((cols[a] for a in al if a in cols), None) for k, al in _COLS.items()}
    if not pick["contract"] or not (pick["iv"] or pick["premium"]):
        raise ValueError("File needs a contract/month column and a vol or premium column.")
    cmap = {c["label"]: c for c in contracts}
    recs, skipped = [], 0
    for _, r in raw.iterrows():
        lbl = parse_contract(r[pick["contract"]])
        c = cmap.get(lbl)
        if not c:
            skipped += 1
            continue
        F = _num(r[pick["futures"]]) if pick["futures"] else None
        F = F / 100 if F and F > 20 else F
        F = F or c["fwd_price"]
        K = _num(r[pick["strike"]]) if pick["strike"] else None
        K = (K / 100 if K and K > 20 else K) or F
        cp = str(r[pick["cp"]]).upper()[:1] if pick["cp"] else ("C" if K >= F else "P")
        cp = cp if cp in ("C", "P") else ("C" if K >= F else "P")
        iv = _num(r[pick["iv"]]) if pick["iv"] else None
        if iv is not None:
            iv = iv / 100 if iv > 3 else iv
        elif pick["premium"]:
            iv = implied_vol(_num(r[pick["premium"]]), F, K, c["t_opt_years"], rate, cp)
        if iv and 0.01 < iv < 5:
            recs.append({"contract": lbl, "strike": round(K, 4), "cp": cp,
                         "futures": F, "iv": round(iv, 5)})
        else:
            skipped += 1
    if not recs:
        raise ValueError("No valid rows matched the current 13-contract window.")
    return pd.DataFrame(recs), skipped


# ── ATM / skew / proxy ───────────────────────────────────────────────────────
def atm_from_surface(surface: pd.DataFrame, contracts):
    """ATM vol per contract, interpolating OTM-side settlement vols at the futures price."""
    out = {}
    if surface is None or surface.empty:
        return out
    for c in contracts:
        s = surface[surface["contract"] == c["label"]]
        if s.empty:
            continue
        F = c["fwd_price"]
        tol = F * 0.005                                      # treat near-ATM strikes as ATM
        otm = s[((s["cp"] == "P") & (s["strike"] <= F + tol)) | ((s["cp"] == "C") & (s["strike"] >= F - tol))]
        s = (otm if len(otm) >= 2 else s).groupby("strike", as_index=False)["iv"].mean()
        s = s.sort_values("strike")
        out[c["label"]] = float(np.interp(F, s["strike"], s["iv"])) if len(s) > 1 else float(s["iv"].iloc[0])
    return out


def proxy_vols(contracts, ovx, ho_returns, wti_returns):
    """OVX scaled by the HO/WTI realized-vol ratio, term structure shaped by each
    contract's own realized vol relative to the front. Labeled Proxy everywhere."""
    if not ovx:
        return {}
    ho_rv = np.std(ho_returns[-30:], ddof=1) if len(ho_returns) > 30 else None
    cl_rv = np.std(wti_returns[-30:], ddof=1) if len(wti_returns) > 30 else None
    ratio = (ho_rv / cl_rv) if ho_rv and cl_rv else 1.0
    front = ovx / 100 * ratio
    rv_front = realized_vol([h["price"] for h in contracts[0].get("history", [])], 30) if contracts else None
    out = {}
    for c in contracts:
        rv = realized_vol([h["price"] for h in c.get("history", [])], 30)
        out[c["label"]] = float(front * (rv / rv_front)) if rv and rv_front else float(front)
    return out


def resolve_contract_vols(contracts, atm_iv, proxy, returns):
    """label -> (sigma_ann, source). Priority: settlement IV -> proxy -> realized."""
    front_rv = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 5 else None
    out = {}
    for c in contracts:
        if c["label"] in atm_iv[0]:
            out[c["label"]] = (atm_iv[0][c["label"]], atm_iv[1])
        elif c["label"] in proxy:
            out[c["label"]] = (proxy[c["label"]], "Proxy")
        else:
            rv = realized_vol([h["price"] for h in c.get("history", [])], 30) or front_rv
            out[c["label"]] = (rv, "Realized") if rv else (None, None)
    return out


# ── Spreads ──────────────────────────────────────────────────────────────────
def calendar_spreads(contracts):
    rows = []
    for a, b in zip(contracts[:-1], contracts[1:]):
        if a["price_source"] == "estimated" or b["price_source"] == "estimated":
            continue
        rows.append({"pair": f"{a['label'][:3]}/{b['label'][:3]} {b['label'][-2:]}",
                     "near": a["label"], "far": b["label"],
                     "spread": round(a["fwd_price"] - b["fwd_price"], 4)})
    return rows


def spread_history(c1, c2):
    h1 = pd.DataFrame(c1.get("history", []))
    h2 = pd.DataFrame(c2.get("history", []))
    if h1.empty or h2.empty:
        return pd.DataFrame()
    m = h1.merge(h2, on="date", suffixes=("_1", "_2"))
    m["spread"] = m["price_1"] - m["price_2"]
    m["date"] = pd.to_datetime(m["date"])
    return m.sort_values("date")


# ── Brazil PPI ───────────────────────────────────────────────────────────────
def ppi_brl_per_liter(ho_usd_gal, usdbrl, params):
    usd_gal = (ho_usd_gal + params["PPI_GULF_BASIS_USD_GAL"] + params["PPI_FREIGHT_USD_GAL"]
               + params["PPI_PORT_COSTS_USD_GAL"])
    return usd_gal * usdbrl / cfg.LITERS_PER_GALLON + params["PPI_INTERNAL_BRL_L"]
