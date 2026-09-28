"""
forecast_store.py — Persistent, append-only storage for forecasts & market inputs.

Why: Streamlit Cloud's disk is wiped on restart. Every forecast we publish must be
kept so the track record can be proven to clients later.

Backend: GitHub Contents API on a dedicated branch (default "data"), so saves do NOT
trigger a Streamlit redeploy. Each save is a git commit => an independent, tamper-evident
timestamp proving the forecast existed before the outcome. Falls back to local disk
(ephemeral, clearly flagged) when no token is configured.

Layout on the data branch (CSV, partitioned by month to keep files small):
  forecasts/YYYY-MM.csv      one row per (forecast_date, kind, target)
  curves/YYYY-MM.csv         daily futures curve snapshot (also = realized settlements)
  vols/atm_YYYY-MM.csv       daily ATM implied/realized vol per contract
  vols/surface/YYYY-MM-DD.csv  CME settlement vol surface (auto or uploaded)
  inputs/market_inputs.csv   manual inputs (Petrobras price, Chicago basis, PPI params)

Security: token read from Secrets only, never logged; all HTTP via requests w/ timeouts.
"""
from __future__ import annotations
import base64, datetime, io, json, os, uuid
import numpy as np
import pandas as pd
import config as cfg

try:
    import requests as _rq
except ImportError:          # pragma: no cover
    _rq = None

_API = "https://api.github.com"
_LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "store")

FORECAST_KEYS = ["forecast_date", "kind", "target", "model_version"]
CURVE_KEYS    = ["date", "contract"]
ATM_KEYS      = ["date", "contract"]
INPUT_KEYS    = ["date", "field"]


# ── Backends ─────────────────────────────────────────────────────────────────
class _LocalBackend:
    name = "local (ephemeral — resets on app restart)"
    persistent = False

    def _p(self, path):
        return os.path.join(_LOCAL_ROOT, *path.split("/"))

    def read(self, path):
        try:
            with open(self._p(path), "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def write(self, path, text, message=""):
        p = self._p(path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return True

    def list_dir(self, d):
        try:
            return sorted(os.listdir(self._p(d)))
        except FileNotFoundError:
            return []

    def status(self):
        return {"backend": self.name, "ok": True, "persistent": False,
                "detail": "Set FORECAST_GITHUB_TOKEN in Streamlit Secrets to persist the track record."}


class _GitHubBackend:
    persistent = True

    def __init__(self, token, repo, branch):
        self._tok, self.repo, self.branch = token, repo, branch
        self.name = f"GitHub {repo}@{branch}"
        self._branch_ok = None
        self.last_error = ""

    def _h(self, raw=False):
        return {"Authorization": f"Bearer {self._tok}",
                "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _ensure_branch(self):
        if self._branch_ok:
            return True
        try:
            r = _rq.get(f"{_API}/repos/{self.repo}/git/ref/heads/{self.branch}",
                        headers=self._h(), timeout=15)
            if r.status_code == 200:
                self._branch_ok = True
                return True
            info = _rq.get(f"{_API}/repos/{self.repo}", headers=self._h(), timeout=15)
            info.raise_for_status()
            default = info.json().get("default_branch", "main")
            ref = _rq.get(f"{_API}/repos/{self.repo}/git/ref/heads/{default}",
                          headers=self._h(), timeout=15)
            ref.raise_for_status()
            sha = ref.json()["object"]["sha"]
            c = _rq.post(f"{_API}/repos/{self.repo}/git/refs", headers=self._h(), timeout=15,
                         json={"ref": f"refs/heads/{self.branch}", "sha": sha})
            self._branch_ok = c.status_code in (200, 201, 422)   # 422 = already exists
            if not self._branch_ok:
                self.last_error = f"create branch HTTP {c.status_code}"
            return self._branch_ok
        except Exception as e:
            self.last_error = f"branch check: {type(e).__name__}"
            return False

    def _meta(self, path):
        r = _rq.get(f"{_API}/repos/{self.repo}/contents/{path}",
                    params={"ref": self.branch}, headers=self._h(), timeout=20)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def read(self, path):
        if not self._ensure_branch():
            return None
        try:
            r = _rq.get(f"{_API}/repos/{self.repo}/contents/{path}",
                        params={"ref": self.branch}, headers=self._h(raw=True), timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception as e:
            self.last_error = f"read {path}: {type(e).__name__}"
            return None

    def write(self, path, text, message=""):
        if not self._ensure_branch():
            return False
        for _attempt in range(2):                       # retry once on sha conflict
            try:
                meta = self._meta(path)
                body = {"message": message or f"data: update {path}",
                        "content": base64.b64encode(text.encode("utf-8")).decode(),
                        "branch": self.branch}
                if meta and isinstance(meta, dict) and meta.get("sha"):
                    body["sha"] = meta["sha"]
                r = _rq.put(f"{_API}/repos/{self.repo}/contents/{path}",
                            headers=self._h(), json=body, timeout=30)
                if r.status_code in (200, 201):
                    return True
                if r.status_code not in (409, 422):
                    self.last_error = f"write {path}: HTTP {r.status_code}"
                    return False
            except Exception as e:
                self.last_error = f"write {path}: {type(e).__name__}"
                return False
        self.last_error = f"write {path}: conflict"
        return False

    def list_dir(self, d):
        if not self._ensure_branch():
            return []
        try:
            m = self._meta(d)
            return sorted(x["name"] for x in (m or []) if isinstance(x, dict))
        except Exception:
            return []

    def status(self):
        ok = self._ensure_branch()
        return {"backend": self.name, "ok": ok, "persistent": True,
                "detail": "Connected" if ok else f"Not reachable ({self.last_error or 'check token scope'})"}


_BACKEND = None


def backend():
    global _BACKEND
    if _BACKEND is None:
        tok = cfg.get("FORECAST_GITHUB_TOKEN")
        if tok and _rq is not None:
            _BACKEND = _GitHubBackend(tok, cfg.get("GITHUB_REPO"), cfg.get("GITHUB_DATA_BRANCH"))
        else:
            _BACKEND = _LocalBackend()
    return _BACKEND


def store_status():
    return backend().status()


# ── Generic table helpers ────────────────────────────────────────────────────
def read_table(path) -> pd.DataFrame:
    txt = backend().read(path)
    if not txt:
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(txt))
    except Exception:
        return pd.DataFrame()


def append_table(path, new: pd.DataFrame, keys, message="", replace=False) -> int:
    """Append rows; rows whose keys already exist are skipped (immutable history)
    unless replace=True (used only for manual inputs corrections)."""
    if new is None or new.empty:
        return 0
    old = read_table(path)
    if not old.empty and all(k in old.columns for k in keys):
        ok = old[keys].astype(str).agg("|".join, axis=1)
        nk = new[keys].astype(str).agg("|".join, axis=1)
        if replace:
            old = old[~ok.isin(set(nk))]
            add = new
        else:
            add = new[~nk.isin(set(ok))]
        if add.empty:
            return 0
        out = pd.concat([old, add], ignore_index=True)
    else:
        add, out = new, new
    return len(add) if backend().write(path, out.to_csv(index=False), message) else 0


def read_partitioned(prefix, stem="") -> pd.DataFrame:
    names = [n for n in backend().list_dir(prefix) if n.endswith(".csv") and n.startswith(stem)]
    frames = [read_table(f"{prefix}/{n}") for n in names]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _month_key(date_str):
    return str(date_str)[:7]


# ── Forecast snapshot ────────────────────────────────────────────────────────
QUANTS = [("p05", .05), ("p10", .10), ("p25", .25), ("p50", .50),
          ("p75", .75), ("p90", .90), ("p95", .95)]


def _lognorm_quantiles(F, sigma_ann, T_years):
    from scipy.stats import norm
    s = max(float(sigma_ann), 1e-6) * np.sqrt(max(T_years, 1e-6))
    mu = np.log(F) - 0.5 * s * s
    q = {k: round(float(np.exp(mu + s * norm.ppf(u))), 4) for k, u in QUANTS}
    q["mean"] = round(float(F), 4)
    return q


def build_forecast_rows(result, contract_vols=None, user=""):
    """Rows for (a) horizon forecasts of the front-month (ensemble model) and
    (b) settlement forecasts per contract (lognormal with implied or realized vol)."""
    import ho_agent as ho
    today = datetime.date.today()
    now = datetime.datetime.utcnow().isoformat(timespec="seconds")
    ver = cfg.get("MODEL_VERSION")
    spot = float(result.get("ho_price") or 0)
    rows = []
    grid = result.get("prob_grid", {})
    for h, g in grid.items():
        days = ho.HORIZON_DAYS.get(h)
        tgt = np.busday_offset(np.datetime64(today), days, roll="forward")
        q = {k: round(ho.grid_quantile(g, u), 4) for k, u in QUANTS}
        rows.append(dict(forecast_id=uuid.uuid4().hex[:12], run_ts_utc=now,
                         forecast_date=str(today), model_version=ver, kind="horizon",
                         target=h, target_date=str(tgt), ref_price=round(spot, 4),
                         vol_ann=round(float(result.get("sigma_daily", 0)) * np.sqrt(252), 4),
                         vol_source="realized", mean=result.get("ev_by_horizon", {}).get(h),
                         user=user, **q))
    cv = contract_vols or {}
    for c in result.get("ho_contracts", []):
        sig, src = cv.get(c["label"], (None, None))
        if not sig:
            sig, src = float(c.get("sigma_daily", 0)) * np.sqrt(252), "realized"
        q = _lognorm_quantiles(c["fwd_price"], sig, c["t_days"] / 252)
        rows.append(dict(forecast_id=uuid.uuid4().hex[:12], run_ts_utc=now,
                         forecast_date=str(today), model_version=ver, kind="contract",
                         target=c["label"], target_date=c["expiry_date"],
                         ref_price=c["fwd_price"], vol_ann=round(sig, 4), vol_source=src,
                         user=user, **q))
    return pd.DataFrame(rows)


def save_daily_snapshot(result, curve_rows=None, atm_rows=None, contract_vols=None, user=""):
    """Idempotent per day: the FIRST forecast of each day is the record (never overwritten)."""
    today = str(datetime.date.today())
    mk = _month_key(today)
    out = {"forecasts": 0, "curves": 0, "atm": 0}
    fc = build_forecast_rows(result, contract_vols, user)
    out["forecasts"] = append_table(f"forecasts/{mk}.csv", fc, FORECAST_KEYS,
                                    f"forecast snapshot {today}")
    if curve_rows:
        out["curves"] = append_table(f"curves/{mk}.csv", pd.DataFrame(curve_rows), CURVE_KEYS,
                                     f"curve snapshot {today}")
    if atm_rows:
        out["atm"] = append_table(f"vols/atm_{mk}.csv", pd.DataFrame(atm_rows), ATM_KEYS,
                                  f"atm vol snapshot {today}")
    return out


def load_forecasts():
    return read_partitioned("forecasts")


def load_curves():
    return read_partitioned("curves")


def load_atm_history():
    return read_partitioned("vols", stem="atm_")


# ── Vol surface (CME settlement vols) ────────────────────────────────────────
def save_vol_surface(df: pd.DataFrame, trade_date: str, source: str):
    df = df.copy()
    df["trade_date"], df["source"] = trade_date, source
    ok = backend().write(f"vols/surface/{trade_date}.csv", df.to_csv(index=False),
                         f"vol surface {trade_date} ({source})")
    return ok


def load_latest_vol_surface():
    names = [n for n in backend().list_dir("vols/surface") if n.endswith(".csv")]
    if not names:
        return pd.DataFrame(), None
    latest = sorted(names)[-1]
    return read_table(f"vols/surface/{latest}"), latest[:-4]


# ── Manual market inputs ─────────────────────────────────────────────────────
INPUT_FIELDS = {
    "petrobras_diesel_brl_l": ("Petrobras diesel A price (distributor)", "BRL/L"),
    "chicago_basis_cpg":      ("Chicago ULSD basis vs NYMEX HO", "¢/gal"),
    "PPI_GULF_BASIS_USD_GAL": ("PPI: USGC basis vs NYMEX", "$/gal"),
    "PPI_FREIGHT_USD_GAL":    ("PPI: ocean freight USGC->Brazil", "$/gal"),
    "PPI_PORT_COSTS_USD_GAL": ("PPI: port/insurance/losses", "$/gal"),
    "PPI_INTERNAL_BRL_L":     ("PPI: internal logistics", "BRL/L"),
}


def load_market_inputs() -> pd.DataFrame:
    df = read_table("inputs/market_inputs.csv")
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("date")
    return df


def save_market_inputs(rows: list, user="") -> int:
    """rows: list of {date, field, value, note}. Same (date, field) is corrected in place."""
    clean = []
    for r in rows:
        if r.get("field") not in INPUT_FIELDS:
            continue
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        clean.append({"date": str(pd.to_datetime(r["date"]).date()), "field": r["field"],
                      "value": v, "unit": INPUT_FIELDS[r["field"]][1],
                      "note": str(r.get("note", ""))[:200], "user": user,
                      "entered_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")})
    return append_table("inputs/market_inputs.csv", pd.DataFrame(clean), INPUT_KEYS,
                        "market inputs", replace=True)


def latest_input(df: pd.DataFrame, field, default=None):
    if df is None or df.empty:
        return default, None
    s = df[df["field"] == field]
    if s.empty:
        return default, None
    last = s.iloc[-1]
    return float(last["value"]), last["date"]


# ── Track-record evaluation ──────────────────────────────────────────────────
def evaluate_forecasts(fc: pd.DataFrame, front_hist: pd.DataFrame, curves: pd.DataFrame):
    """Attach realized outcomes.
    horizon  -> first HO front-month close on/after target_date
    contract -> last stored curve price for that contract on/before its expiry
    """
    if fc is None or fc.empty:
        return pd.DataFrame()
    fc = fc.copy()
    today = str(datetime.date.today())
    fh = front_hist.sort_values("date") if front_hist is not None and not front_hist.empty else None
    cv = curves.sort_values("date") if curves is not None and not curves.empty else None
    realized, rdate, status = [], [], []
    for _, r in fc.iterrows():
        val, d, st_ = np.nan, "", "pending"
        if str(r["target_date"]) <= today:
            if r["kind"] == "horizon" and fh is not None:
                m = fh[fh["date"].astype(str) >= str(r["target_date"])]
                if not m.empty:
                    val, d, st_ = float(m.iloc[0]["price"]), str(m.iloc[0]["date"]), "evaluated"
                else:
                    st_ = "awaiting data"
            elif r["kind"] == "contract" and cv is not None:
                m = cv[(cv["contract"] == r["target"]) &
                       (cv["date"].astype(str) <= str(r["target_date"]))]
                if not m.empty:
                    val, d, st_ = float(m.iloc[-1]["price"]), str(m.iloc[-1]["date"]), "evaluated"
                else:
                    st_ = "no settlement stored"
        realized.append(val); rdate.append(d); status.append(st_)
    fc["realized"], fc["realized_date"], fc["status"] = realized, rdate, status
    ev = fc["status"] == "evaluated"
    for lo, hi, name in (("p25", "p75", "in_50"), ("p10", "p90", "in_80"), ("p05", "p95", "in_90")):
        fc[name] = np.where(ev, (fc["realized"] >= fc[lo]) & (fc["realized"] <= fc[hi]), np.nan)
    fc["abs_err_median"] = np.where(ev, (fc["realized"] - fc["p50"]).abs(), np.nan)
    return fc


def track_record_summary(ev: pd.DataFrame):
    if ev is None or ev.empty:
        return {}
    e = ev[ev["status"] == "evaluated"]
    out = {"total": int(len(ev)), "evaluated": int(len(e)),
           "pending": int((ev["status"] != "evaluated").sum())}
    if len(e):
        out.update({"cov50": float(e["in_50"].mean() * 100), "cov80": float(e["in_80"].mean() * 100),
                    "cov90": float(e["in_90"].mean() * 100),
                    "mae": float(e["abs_err_median"].mean())})
    return out
