"""
config.py — Central, override-able settings for the Energy Intelligence Dashboard.

Resolution order for every key (first hit wins):
  1. Streamlit Cloud secrets   (Manage app -> Settings -> Secrets, TOML)
  2. Environment variables     (local .env via python-dotenv)
  3. DEFAULTS below

Nothing here is a displayed market value: these are model knobs, feature flags and
integration settings. Change them in Secrets — no code edit / redeploy needed.
"""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv as _ld
    _ld(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        override=False)
except ImportError:
    pass

DEFAULTS = {
    # ── Feature flags ────────────────────────────────────────────────────────
    "EIA_ENABLED":              False,   # EIA Inventory section + EIA signal. Set true to re-enable.
    "FORECAST_AUTOSAVE":        True,    # save one forecast snapshot per day on HO run

    # ── Forecast store (GitHub) ──────────────────────────────────────────────
    "FORECAST_GITHUB_TOKEN":    "",      # fine-grained PAT (Contents: read/write, this repo only)
    "GITHUB_REPO":              "Luiz-Saggioro/Heating-Oil",
    "GITHUB_DATA_BRANCH":       "data",  # NOT the deployed branch -> no Streamlit redeploy per save
    "MODEL_VERSION":            "v4.0",

    # ── Probability table (section 03) ───────────────────────────────────────
    "PROB_GRID_STEP":           0.01,    # $/gal resolution of the internal fine grid
    "PROB_TABLE_SIGMA_RANGE":   2.0,     # default table range = ± N sigma of selected horizon
    "PROB_TABLE_BINS":          10,      # default number of interior bins

    # ── Scenario presets: percentiles of historical 1-month moves ────────────
    "SCEN_LOOKBACK_DAYS":       21,
    "SCEN_UP_PCTL":             95,      # "Supply Disruption" tail
    "SCEN_DEMAND_PCTL":         90,      # "Cold Winter / High Demand" crack tail
    "SCEN_DOWN_PCTL":           5,       # "Recession" tail
    "SCEN_CALM_VOL_PCTL":       25,      # "Stable Market" vol level

    # ── KO section ───────────────────────────────────────────────────────────
    "KO_DEFAULT_PCT_OF_SPOT":   0.85,    # default KO input = spot * this
    "KO_GRID_POINTS":           9,       # heatmap resolution (price shocks x vol levels)

    # ── Volatility / CME ─────────────────────────────────────────────────────
    "CME_AUTO_FETCH":           True,
    "CME_HO_FUTURES_PRODUCT_ID": "426",  # NY Harbor ULSD (HO) futures on cmegroup.com
    "CME_HO_OPTION_PRODUCT_ID": "",      # blank = auto-discover; set if discovery fails
    "OPTION_EXPIRY_BDAYS_BEFORE_FUT": 3, # OH options stop trading N biz days before futures
    "RISK_FREE_FALLBACK":       0.04,    # used only if ^IRX fetch fails
    "CONTRACT_COUNT":           13,      # front + next 12

    # ── Brazil PPI (import parity) components — set in Secrets or admin panel ──
    "PPI_GULF_BASIS_USD_GAL":   0.0,     # USGC ULSD vs NYMEX HO
    "PPI_FREIGHT_USD_GAL":      0.0,     # USGC -> Brazil ocean freight
    "PPI_PORT_COSTS_USD_GAL":   0.0,     # port, insurance, losses, AFRMM
    "PPI_INTERNAL_BRL_L":       0.0,     # internal logistics to the reference point
}

LITERS_PER_GALLON = 3.785411784   # physical constant
GALLONS_PER_BARREL = 42           # physical constant


def _secrets():
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return {}


def get(key: str, default=None):
    """Return a setting, cast to the type of its DEFAULTS entry."""
    base = DEFAULTS.get(key, default)
    raw = None
    try:
        s = _secrets()
        if key in s:
            raw = s[key]
    except Exception:
        raw = None
    if raw is None:
        raw = os.environ.get(key)
    if raw is None:
        return base
    try:
        if isinstance(base, bool):
            return raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(base, int) and not isinstance(base, bool):
            return int(raw)
        if isinstance(base, float):
            return float(raw)
    except (TypeError, ValueError):
        return base
    return str(raw).strip() if isinstance(raw, str) else raw
