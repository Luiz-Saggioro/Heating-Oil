"""
scenario_engine.py — "If X, Y or Z happens, how does the HO distribution change?"

Every scenario is a set of NUMERIC shocks:
    WTI change (%), HO crack-spread change ($/bbl), volatility change (vol points)
Preset sizes are NOT hard-coded: they are percentiles of the last year's
historical 1-month moves (percentiles configurable in config.py / Secrets).

HO price under a scenario uses the crack-spread identity (no fitted coefficient):
    HO' ($/gal) = (WTI x (1 + wti_shock) + crack + crack_shock) / 42

The scenario distribution keeps the ensemble model's shape and applies a
log-space affine map around its median m:
    Q_s(u) = k * m * (Q(u) / m) ** vr      k = HO'/HO,  vr = vol_s / vol_base
"""
from __future__ import annotations
import numpy as np
import config as cfg

G = cfg.GALLONS_PER_BARREL


def _moves(hist, n, log=True):
    p = np.asarray([r for r in hist if r is not None], dtype=float)
    if len(p) <= n:
        return np.array([])
    return (np.log(p[n:] / p[:-n])) if log else (p[n:] - p[:-n])


def _rolling_rv(returns, w=30):
    r = np.asarray(returns, dtype=float)
    if len(r) < w + 5:
        return np.array([])
    return np.array([np.std(r[i - w:i], ddof=1) * np.sqrt(252) for i in range(w, len(r) + 1)])


def compute_presets(wti_hist, crack_hist, ho_returns, base_vol_ann):
    """Return list of scenario dicts with numeric shocks derived from history."""
    n = cfg.get("SCEN_LOOKBACK_DAYS")
    up, dem, dn, calm = (cfg.get("SCEN_UP_PCTL"), cfg.get("SCEN_DEMAND_PCTL"),
                         cfg.get("SCEN_DOWN_PCTL"), cfg.get("SCEN_CALM_VOL_PCTL"))
    wm = _moves(wti_hist, n, log=True)
    cm = _moves(crack_hist, n, log=False)
    rv = _rolling_rv(ho_returns)
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else 0.0
    rv_now = float(rv[-1]) if len(rv) else base_vol_ann
    dv = lambda p: (q(rv, p) - rv_now) * 100 if len(rv) else 0.0
    wpct = lambda p: (np.exp(q(wm, p)) - 1) * 100 if len(wm) else 0.0
    presets = [
        dict(name="Base", wti_pct=0.0, crack_chg=0.0, vol_pts=0.0,
             why="Current market, no shock"),
        dict(name="Cold Winter / High Demand", wti_pct=wpct(50), crack_chg=q(cm, dem),
             vol_pts=max(0.0, dv(75)),
             why=f"Crack at its {dem}th-pctl 1M widening; WTI median move; vol to 75th pctl"),
        dict(name="Supply Disruption", wti_pct=wpct(up), crack_chg=q(cm, up),
             vol_pts=max(0.0, dv(up)),
             why=f"WTI & crack at {up}th-pctl 1M moves; vol to {up}th pctl"),
        dict(name="Stable Market", wti_pct=wpct(50), crack_chg=q(cm, 50),
             vol_pts=min(0.0, dv(calm)),
             why=f"Median moves; vol compresses to {calm}th pctl"),
        dict(name="Recession", wti_pct=wpct(dn), crack_chg=q(cm, 100 - dem),
             vol_pts=max(0.0, dv(100 - dn)),
             why=f"WTI at {dn}th-pctl 1M move, crack {100 - dem}th pctl; vol to {100 - dn}th pctl"),
    ]
    for p in presets:
        p.update({k: round(float(p[k]), 2) for k in ("wti_pct", "crack_chg", "vol_pts")})
    return presets


def scenario_price(ho, wti, crack, s):
    if wti and crack is not None:
        return max(0.05, (wti * (1 + s["wti_pct"] / 100) + crack + s["crack_chg"]) / G)
    return max(0.05, ho * (1 + s["wti_pct"] / 100) + s["crack_chg"] / G)


def scenario_factors(ho, wti, crack, base_vol_ann, s):
    k = scenario_price(ho, wti, crack, s) / ho
    vs = max(0.02, base_vol_ann + s["vol_pts"] / 100)
    return k, vs / max(base_vol_ann, 1e-6), vs


def scen_cdf(grid_cdf, x, median, k, vr):
    x = np.maximum(np.asarray(x, dtype=float), 1e-9)
    return grid_cdf(median * (x / (k * median)) ** (1.0 / vr))


def bin_probs(grid_cdf, edges, median=None, k=1.0, vr=1.0):
    """Probabilities for [<e0, e0-e1, ..., >en] under (optionally) a scenario."""
    f = (lambda x: scen_cdf(grid_cdf, x, median, k, vr)) if median else grid_cdf
    c = np.array([f(e) for e in edges], dtype=float)
    return np.diff(np.concatenate([[0.0], c, [1.0]])).clip(0, 1)


def scen_quantile(grid_quantile, u, median, k, vr):
    return k * median * (grid_quantile(u) / median) ** vr
