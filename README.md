# Energy Intelligence Dashboard (v4.1)

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
├── theme.py            ← Light/dark palettes, global CSS, Plotly templates (no colours elsewhere)
├── users.json / manage_users.py ← Login + 2FA
└── .streamlit/secrets.toml.example ← Template for Streamlit Cloud Secrets
```

## Streamlit Cloud Secrets (Manage app → Settings → Secrets)

See `.streamlit/secrets.toml.example`. Minimum for the track record:

```toml
FORECAST_GITHUB_TOKEN = "github_pat_..."   # fine-grained PAT: this repo only, Contents read/write
```

Without it the app still runs, but forecasts are saved to local disk only (wiped on restart)
and a warning is shown in the Track Record tab and the sidebar.

## Dashboard layout (v4.1 — one tab per measure)

The Snapshot KPI strip sits above the tabs. Charts are grouped by what they measure;
categories with several measures get a second row of sub-tabs. Layout is declared in
`_dashboard_layout()` in `streamlit_app.py` — edit that list to move a section.

| Tab | Sub-tabs | Notes |
|-----|----------|-------|
| Prices & Averages | Price history & moving avg · Seasonal monthly averages | Front contract; MA window adapts to period |
| Probabilities | Price distribution & EV · Scenario impact · KO by contract · At expiration | Range = ± N σ of horizon; editable shocks (WTI %, crack $/bbl, vol pts) |
| Volatility & Risk (SD) | Volatility (std. dev.) · VaR & Expected Shortfall | ATM implied vs realized (10/20/30d), term structure, vol by strike |
| Curves & Spreads | Forward curve & calendar spreads · Crack spread · Chicago basis & Brazil PPI · EIA inventory | EIA **off** by default (`EIA_ENABLED = true` in Secrets) |
| Scenarios | — | Paths driven by the Scenario impact shocks |
| Regional | — | US / Brazil maps |
| Track Record | — | Daily immutable snapshots + hit rates vs realized |
| Summary & Logs | — | Market summary, run log, security audit |

Oil (WTI) runs show only the tabs that apply to that engine.

**Display mode:** sidebar *Dark mode* toggle (first visit follows the browser/Streamlit theme).
All colours come from `theme.py`; text meets WCAG AA contrast in both modes. Tables built with
`st.data_editor` keep Streamlit's own grid colours (not stylable via CSS).

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
