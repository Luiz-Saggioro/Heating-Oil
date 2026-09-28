# Energy Intelligence Dashboard (v4.0)

Deterministic commodity probability engine for **Heating Oil (HO)** and **WTI Crude**.
Streamlit app, auto-deployed from `main` to Streamlit Cloud.

## File structure

```
├── streamlit_app.py    ← UI (auth + all sections). Run this.
├── ho_agent.py         ← HO engine: fine-grid ensemble, KO/expiry maths, scenarios presets
├── oil_agent_v2.py     ← WTI/Brent engine
├── data_fetcher.py     ← Live prices/history (yfinance → Yahoo v8 → FRED)
├── vol_engine.py       ← Futures curve, CME settlement vol (Black-76), realized vol, spreads, PPI
├── scenario_engine.py  ← Numeric what-if shocks → shifted probability distribution
├── forecast_store.py   ← Persistent forecast track record (GitHub `data` branch)
├── config.py           ← All knobs & feature flags (overridable in Secrets)
├── users.json / manage_users.py ← Login + 2FA
└── .streamlit/secrets.toml.example ← Template for Streamlit Cloud Secrets
```

## Streamlit Cloud Secrets (Manage app → Settings → Secrets)

See `.streamlit/secrets.toml.example`. Minimum for the track record:

```toml
FORECAST_GITHUB_TOKEN = "github_pat_..."   # fine-grained PAT: this repo only, Contents read/write
```

Without it the app still runs, but forecasts are saved to local disk only (wiped on restart)
and a warning is shown in section 12 and the sidebar.

## Dashboard sections

| # | Section | Notes |
|---|---------|-------|
| 01 | Snapshot | Live prices (30 s refresh) |
| 02 | Price History | Front contract |
| 03 | Probability Distribution | Range = ± N σ of the selected horizon, bin count adjustable, manual override |
| 03B | Scenario Impact | Editable numeric shocks (WTI %, crack $/bbl, vol pts) → probabilities, EV, 80 % range, P(< / >) |
| 04 | KO Probability | Implied vol per contract, P(KO) per scenario, price × vol sensitivity map |
| 04B | Probability at Expiration | Per-contract settlement distribution using implied vol |
| 05 | Volatility | ATM implied vs realized (10/20/30d), term structure by month, vol by strike |
| 05B | Forward Curve & Spreads | Real NYMEX contracts, consecutive-month spreads + history |
| 06 | Crack Spread | |
| 06B | Chicago Basis & Brazil PPI | Admin-entered basis / Petrobras price, computed import parity |
| 07 | EIA Inventory | **Off** by default — `EIA_ENABLED = true` in Secrets to turn back on |
| 08–11 | Seasonal, VaR, Scenario paths, Regional map | |
| 12 | Forecast Track Record | Daily immutable snapshots + hit rates vs realized |

## Data sources & fallbacks

* **Futures curve:** yfinance contracts (`HOX26.NYM`) → CME settlements → flat (flagged).
* **Implied vol:** CME option settlements inverted with Black-76 (best-effort; CME often blocks
  cloud IPs) → latest admin upload (CSV/XLSX from CME/QuikStrike) → proxy OVX × HO/WTI RV ratio
  (always labeled "Proxy").
* **Chicago basis / Petrobras price:** no free API — entered by admins in sections 06B,
  stored with date + source note.
* **PPI:** (NYMEX HO + USGC basis + freight + port costs) × USD/BRL ÷ 3.785 L/gal + internal logistics.
  Cost components are admin inputs (default 0 → warning shown).

## Forecast track record (how it works)

On each day's first HO run the app writes, to branch `data` of this repo:

* `forecasts/YYYY-MM.csv` — 5 horizon forecasts (1M…12M) + 13 contract-settlement forecasts,
  each with p05/p10/p25/p50/p75/p90/p95, reference price, vol used and its source.
* `curves/YYYY-MM.csv` — the day's futures curve (later used as the realized settlement).
* `vols/atm_YYYY-MM.csv` — ATM implied & realized vol per contract (builds IV history).

Existing rows are never overwritten (first forecast of the day is the record), and every save is a
git commit — an independent timestamp proving the forecast existed before the outcome.
Commits go to `data`, not `main`, so they do not trigger redeploys.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Contact
lsaggioro@potonmail.com
