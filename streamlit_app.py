"""
Energy Intelligence Dashboard — Streamlit Edition
HO + WTI/Oil. All emojis removed from UI labels.
Security: rate-limiting, input sanitization, env-var secrets, login + 2FA.
Changes v3.0:
  - Add: Login gate with PBKDF2 password verification
  - Add: TOTP two-factor authentication (Google Authenticator / Authy)
  - Add: Admin user management panel (add / deactivate users)
  - Add: Logout button in sidebar
  - Note: users stored in users.json; manage locally via manage_users.py
Changes v3.1:
  - Add: Section 05  — KO Probability Table (Table 1)
         P(each HO futures contract touches a user-specified KO price before expiry)
  - Add: Section 05B — Probability at Expiration Table (Table 2)
         Lognormal settlement distribution across 13 monthly contracts
Changes v4.0 (market feedback — Tim):
  - 03  Probability table: range/bins now dynamic around spot (narrower), user-adjustable
  - 03B Scenario impact: numeric shocks (WTI %, crack $/bbl, vol pts) -> shifted distribution
  - 04  KO: implied vol per contract + P(KO) under each scenario + price x vol heatmap
  - 05  Volatility: CME settlement vol (auto -> upload -> labeled proxy), ATM IV vs realized,
        term structure by month, vol by strike
  - 05B Forward curve (real NYMEX contracts) + inter-month calendar spreads
  - 06B Chicago ULSD basis + Brazil PPI (import parity) vs Petrobras
  - 07  EIA inventory hidden behind EIA_ENABLED flag (config / Secrets)
  - 12  Forecast track record: daily snapshots persisted to GitHub 'data' branch + evaluation
Changes v3.2:
  - 02 Price History: title now shows HO1 — Front Contract with dynamic ticker (e.g. HOU26)
  - 01 Snapshot + 02 Price History: 30-second live spot price refresh via streamlit-autorefresh
  - 04 Volatility: added Rolling 30-day realized vol + OVX Implied Volatility overlay
  - Section order restructured: Snapshot→PriceHistory→ProbDist→KOProb→Volatility→
      CrackSpread→Inventory→Seasonal→VaR→Scenario→Map
  - KO section: interactive inputs drive all metrics on the fly (already live)
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import datetime
import os
import data_fetcher as _df
import config as cfg
import forecast_store as fs
import vol_engine as ve
import scenario_engine as se
import json
import hashlib
# 30-second auto-refresh for live market data (Snapshot + Price History)
try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _AUTOREFRESH_OK = True
except ImportError:
    _AUTOREFRESH_OK = False
# Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass
# ── INLINED SECURITY MODULE ───────────────────────────────────────────────────
# Inlined so no extra file is required on Streamlit Cloud.
import re, time, html, logging
from collections import defaultdict
logger = logging.getLogger(__name__)
_RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", 900))
_RATE_LIMIT_MAX: int    = int(os.environ.get("RATE_LIMIT_MAX_ATTEMPTS", 5))
_MAX_PAYLOAD: int       = int(os.environ.get("MAX_PAYLOAD_BYTES", 65536))
class _RateLimiter:
    def __init__(self, max_attempts=_RATE_LIMIT_MAX, window_secs=_RATE_LIMIT_WINDOW):
        self._max = max_attempts
        self._win = window_secs
        self._log: dict = defaultdict(list)
    def check(self, session_id: str):
        now = time.monotonic()
        ws  = now - self._win
        h   = self._log[session_id]
        h[:] = [t for t in h if t > ws]
        if len(h) >= self._max:
            return False, 0, int(self._win - (now - h[0])) + 1
        h.append(now)
        return True, self._max - len(h), 0
    def remaining(self, session_id: str) -> int:
        now = time.monotonic()
        ws  = now - self._win
        return max(0, self._max - len([t for t in self._log.get(session_id, []) if t > ws]))
    def reset(self, session_id: str) -> None:
        self._log.pop(session_id, None)
def sanitize_str(value: str, max_len: int = 200) -> str:
    if not isinstance(value, str):
        value = str(value)
    value = html.escape(value.strip())
    value = re.sub(r"[^\w\s\.\-\$\%\/\(\)&,]", "", value)
    return value[:max_len]
def validate_enum(value: str, allowed: set) -> str:
    if value not in allowed:
        raise ValueError(f"Invalid value: {value!r}")
    return value
def reject_oversized(obj, max_len: int = _MAX_PAYLOAD, label: str = "input") -> None:
    size = len(str(obj))
    if size > max_len:
        raise ValueError(f"{label} too large: {size} > {max_len}")
def security_audit_report() -> str:
    issues, passed = [], []
    eia = os.environ.get("EIA_API_KEY", "")
    if not cfg.get("EIA_ENABLED"):
        passed.append("EIA inventory disabled by flag (EIA_ENABLED=false)")
    elif not eia:
        issues.append("EIA_API_KEY not set — EIA inventory fetch will fail")
    elif eia == "DEMO_KEY":
        issues.append("EIA_API_KEY is still 'DEMO_KEY' — set a real key")
    else:
        passed.append("EIA_API_KEY is set")
    passed.append("Forecast store: " + fs.store_status().get("backend", "?"))
    if _RATE_LIMIT_MAX < 1 or _RATE_LIMIT_MAX > 100:
        issues.append(f"RATE_LIMIT_MAX_ATTEMPTS={_RATE_LIMIT_MAX} outside 1-100")
    else:
        passed.append(f"Rate limit: {_RATE_LIMIT_MAX} attempts / {_RATE_LIMIT_WINDOW}s")
    if _MAX_PAYLOAD < 1024:
        issues.append(f"MAX_PAYLOAD_BYTES={_MAX_PAYLOAD} very small")
    else:
        passed.append(f"Payload limit: {_MAX_PAYLOAD} bytes")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    if os.path.isdir(output_dir):
        mode = oct(os.stat(output_dir).st_mode)[-3:]
        if mode[-1] in ("6", "7"):
            issues.append(f"output/ world-writable ({mode}) — chmod o-w output/")
        else:
            passed.append(f"output/ permissions OK ({mode})")
    lines = ["=== SECURITY AUDIT REPORT ==="]
    if passed:
        lines += ["", "PASSED:"] + [f"  [OK] {p}" for p in passed]
    if issues:
        lines += ["", "WARNINGS:"] + [f"  [!!] {i}" for i in issues]
    if not issues:
        lines.append("\nAll checks passed.")
    return "\n".join(lines)
limiter = _RateLimiter()
# ── INLINED AUTH MODULE ───────────────────────────────────────────────────────
# Users stored in users.json (PBKDF2 hashed passwords + TOTP secrets).
# Manage users locally with manage_users.py, then commit to GitHub.
_PBKDF2_ITERS = 260000
_ADMIN_LOGIN  = "Luiz Saggioro"
def _resolve_users_file() -> str:
    """Try several paths so the file is found both locally and on Streamlit Cloud."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json"),
        os.path.join(os.getcwd(), "users.json"),
        "users.json",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]   # fall back to first; will raise a clear error on open
def _load_users() -> list:
    path = _resolve_users_file()
    try:
        with open(path, "r", encoding="utf-8") as _f:
            return json.load(_f).get("users", [])
    except FileNotFoundError:
        logger.error(f"[AUTH] users.json not found at {path} — check repo contains the file")
        return []
    except Exception as exc:
        logger.error(f"[AUTH] Failed to load users.json: {exc}")
        return []
def _find_user(login: str):
    for u in _load_users():
        if u.get("login", "").strip().lower() == login.strip().lower():
            return u
    return None
def _verify_password(user: dict, password: str) -> bool:
    try:
        salt     = bytes.fromhex(user["salt"])
        expected = user["password_hash"]
        key      = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
        return key.hex() == expected
    except Exception:
        return False
def _verify_totp(secret: str, code: str) -> bool:
    """Returns True if code matches.  Fails open only when pyotp is missing."""
    try:
        import pyotp
        return pyotp.TOTP(secret).verify(code.strip(), valid_window=1)
    except ImportError:
        logger.warning("[AUTH] pyotp not installed — skipping TOTP check")
        return True
def _totp_uri(secret: str, login: str) -> str:
    try:
        import pyotp
        return pyotp.TOTP(secret).provisioning_uri(name=login, issuer_name="Energy Intelligence")
    except ImportError:
        return ""
def _save_users(users: list) -> bool:
    path = _resolve_users_file()
    try:
        db = {"_comment": "Managed via manage_users.py. Commit to GitHub to persist.",
              "users": users}
        with open(path, "w", encoding="utf-8") as _f:
            json.dump(db, _f, indent=2)
        return True
    except Exception as exc:
        logger.error(f"[AUTH] save_users failed: {exc}")
        return False
def _mark_totp_enabled(login: str) -> None:
    users = _load_users()
    for u in users:
        if u.get("login", "").strip().lower() == login.strip().lower():
            u["totp_enabled"] = True
            break
    _save_users(users)
# ── Auth session helpers ──────────────────────────────────────────────────────
def _auth_init():
    defaults = dict(
        auth_logged_in    = False,
        auth_user         = None,
        auth_is_admin     = False,
        auth_step         = "login",    # "login" | "totp" | "totp_setup"
        auth_pending_user = None,
        auth_error        = "",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
def _auth_logout():
    for k in ["auth_logged_in","auth_user","auth_is_admin",
              "auth_step","auth_pending_user","auth_error"]:
        st.session_state.pop(k, None)
# ── Auth CSS ──────────────────────────────────────────────────────────────────
_AUTH_CSS = """
<style>
.auth-wrap{max-width:400px;margin:80px auto;}
.auth-card{
  padding:40px 36px;background:#ffffff;
  border:1px solid #c4d0de;border-radius:12px;
  box-shadow:0 4px 24px rgba(0,0,0,0.09);
}
.auth-title{
  font-size:22px;font-weight:800;color:#1b2a3b;
  font-family:'Syne',sans-serif;margin-bottom:4px;
}
.auth-sub{
  font-size:11px;color:#4e6880;
  font-family:'JetBrains Mono',monospace;
  letter-spacing:1.2px;text-transform:uppercase;margin-bottom:28px;
}
</style>
"""
# ── Login form (step 1) ───────────────────────────────────────────────────────
def render_login_form():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>Energy Intelligence</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-sub'>Secure access — sign in to continue</div>",
                unsafe_allow_html=True)
    # Warn immediately if users.json is missing — avoids silent failure
    if not os.path.isfile(_resolve_users_file()):
        st.warning("⚠️ users.json not found in the repo. Add the file and redeploy.")
    with st.form("login_form", clear_on_submit=False):
        login    = st.text_input("Login name", placeholder="e.g. Luiz Saggioro")
        password = st.text_input("Password", type="password")
        submit   = st.form_submit_button("Sign in", use_container_width=True)
    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)
        st.session_state.auth_error = ""
    if submit:
        login = sanitize_str(login.strip(), max_len=100)
        user  = _find_user(login)
        if not user or not user.get("is_active", False) or not _verify_password(user, password):
            st.session_state.auth_error = "Invalid credentials or account inactive."
            st.rerun()
        else:
            st.session_state.auth_pending_user = user
            st.session_state.auth_step = "totp_setup" if not user.get("totp_enabled") else "totp"
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
# ── TOTP verification (step 2, returning users) ───────────────────────────────
def render_totp_form():
    user = st.session_state.auth_pending_user
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>Two-Factor Auth</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='auth-sub'>Enter the 6-digit code for {user['login']}</div>",
                unsafe_allow_html=True)
    with st.form("totp_form", clear_on_submit=True):
        code   = st.text_input("Authenticator code", max_chars=6, placeholder="000000")
        submit = st.form_submit_button("Verify", use_container_width=True)
    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)
        st.session_state.auth_error = ""
    if st.button("Back to login", key="totp_back"):
        st.session_state.auth_step = "login"
        st.session_state.auth_pending_user = None
        st.rerun()
    if submit:
        if _verify_totp(user["totp_secret"], code):
            st.session_state.auth_logged_in    = True
            st.session_state.auth_user         = user["login"]
            st.session_state.auth_is_admin     = user.get("is_admin", False)
            st.session_state.auth_step         = "login"
            st.session_state.auth_pending_user = None
            st.rerun()
        else:
            st.session_state.auth_error = "Incorrect code — try again."
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
# ── TOTP first-time setup (step 2, new users) ────────────────────────────────
def render_totp_setup():
    user   = st.session_state.auth_pending_user
    secret = user["totp_secret"]
    uri    = _totp_uri(secret, user["login"])
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    st.markdown("<div class='auth-wrap'><div class='auth-card'>", unsafe_allow_html=True)
    st.markdown("<div class='auth-title'>Set up Two-Factor Auth</div>", unsafe_allow_html=True)
    st.markdown("<div class='auth-sub'>One-time setup — scan QR or enter key manually</div>",
                unsafe_allow_html=True)
    qr_ok = False
    if uri:
        try:
            import qrcode, io
            buf = io.BytesIO()
            qrcode.make(uri).save(buf, format="PNG")
            buf.seek(0)
            st.image(buf, caption="Scan with Google Authenticator / Authy / 1Password", width=240)
            qr_ok = True
        except ImportError:
            pass
    if not qr_ok:
        st.markdown("**Add manually in your authenticator app:**")
        if uri:
            st.code(uri, language=None)
    st.markdown("**Secret key (manual entry):**")
    st.code(secret, language=None)
    st.caption("Time-based (TOTP) · issuer: Energy Intelligence")
    with st.form("totp_setup_form", clear_on_submit=True):
        code   = st.text_input("Confirm with a 6-digit code", max_chars=6, placeholder="000000")
        submit = st.form_submit_button("Confirm & sign in", use_container_width=True)
    if st.session_state.auth_error:
        st.error(st.session_state.auth_error)
        st.session_state.auth_error = ""
    if st.button("Back to login", key="setup_back"):
        st.session_state.auth_step = "login"
        st.session_state.auth_pending_user = None
        st.rerun()
    if submit:
        if _verify_totp(secret, code):
            _mark_totp_enabled(user["login"])
            st.session_state.auth_logged_in    = True
            st.session_state.auth_user         = user["login"]
            st.session_state.auth_is_admin     = user.get("is_admin", False)
            st.session_state.auth_step         = "login"
            st.session_state.auth_pending_user = None
            st.rerun()
        else:
            st.session_state.auth_error = "Incorrect code — make sure you scanned the right key."
            st.rerun()
    st.markdown("</div></div>", unsafe_allow_html=True)
# ── Auth gate dispatcher ──────────────────────────────────────────────────────
def render_auth_gate() -> bool:
    """Call before any dashboard content. Returns True if user is authenticated."""
    _auth_init()
    if st.session_state.auth_logged_in:
        return True
    step = st.session_state.auth_step
    if step == "totp":
        render_totp_form()
    elif step == "totp_setup":
        render_totp_setup()
    else:
        render_login_form()
    return False
# ── Admin panel (sidebar) ─────────────────────────────────────────────────────
def render_admin_panel():
    """Sidebar expander for admin user management."""
    import secrets as _sec
    import datetime as _dt
    st.sidebar.divider()
    with st.sidebar.expander("User Management (Admin)", expanded=False):
        users = _load_users()
        st.markdown("**Users**")
        for u in users:
            tag = ("Active" if u.get("is_active") else "**Inactive**") + \
                  (" · 2FA on" if u.get("totp_enabled") else " · 2FA pending") + \
                  (" · Admin" if u.get("is_admin") else "")
            st.markdown(f"`{u['login']}` — {tag}")
        st.markdown("---")
        st.markdown("**Toggle active status**")
        non_admin = [u["login"] for u in users if u["login"] != _ADMIN_LOGIN]
        if non_admin:
            toggle = st.selectbox("User", non_admin, key="adm_tog")
            c1, c2 = st.columns(2)
            if c1.button("Activate", key="adm_act", use_container_width=True):
                for u in users:
                    if u["login"] == toggle: u["is_active"] = True
                ok = _save_users(users)
                st.success(f"{toggle} activated." + ("" if ok else " (session only)"))
                st.rerun()
            if c2.button("Deactivate", key="adm_deact", use_container_width=True):
                for u in users:
                    if u["login"] == toggle: u["is_active"] = False
                ok = _save_users(users)
                st.success(f"{toggle} deactivated." + ("" if ok else " (session only)"))
                st.rerun()
        else:
            st.caption("No other users to manage.")
        st.markdown("---")
        st.markdown("**Add new user**")
        nl = st.text_input("Login name", key="adm_nl")
        np = st.text_input("Password",   type="password", key="adm_np")
        if st.button("Create", key="adm_create", use_container_width=True):
            nl = sanitize_str(nl.strip(), 100)
            if not nl or not np:
                st.error("Both fields required.")
            elif len(np) < 8:
                st.error("Password must be at least 8 characters.")
            elif any(u["login"].strip().lower() == nl.lower() for u in users):
                st.error(f"'{nl}' already exists.")
            else:
                _salt = os.urandom(16)
                _key  = hashlib.pbkdf2_hmac("sha256", np.encode(), _salt, _PBKDF2_ITERS)
                _totp = "".join(_sec.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(32))
                users.append({
                    "login": nl, "password_hash": _key.hex(), "salt": _salt.hex(),
                    "is_admin": False, "is_active": True,
                    "totp_secret": _totp, "totp_enabled": False,
                    "created_at": str(_dt.date.today()),
                })
                ok = _save_users(users)
                msg = f"'{nl}' created." + ("" if ok else " Commit users.json via manage_users.py to persist.")
                st.success(msg)
                st.rerun()
# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Intelligence Dashboard",
    page_icon="energy",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── THEME / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;}
code,.stCode,pre{font-family:'JetBrains Mono',monospace!important;}
.stApp{background:#f0f4f8!important;}
.stPlotlyChart{background:transparent!important;}
.stPlotlyChart>div{background:transparent!important;}
div[data-testid="stPlotlyChart"]{background:transparent!important;}
div[data-testid="block-container"]{background:transparent!important;}
div[data-testid="stVerticalBlock"]{background:transparent!important;}
div[data-testid="column"]{background:transparent!important;}
div[data-testid="stHorizontalBlock"]{background:transparent!important;}
.element-container{background:transparent!important;}
section[data-testid="stSidebar"]{background:#1b2a3b;border-right:1px solid #26394d;}
section[data-testid="stSidebar"] .stMarkdown{color:#c8d8ea;}
section[data-testid="stSidebar"] label{color:#8aaac4!important;}
section[data-testid="stSidebar"] p{color:#c8d8ea!important;}
div[data-testid="metric-container"]{background:#ffffff;border:1px solid #c4d0de;border-radius:8px;padding:14px 18px;transition:border-color .2s;box-shadow:0 1px 4px rgba(0,0,0,0.06);}
div[data-testid="metric-container"]:hover{border-color:#8aaac4;}
div[data-testid="metric-container"] label{color:#4e6880!important;font-family:'JetBrains Mono',monospace!important;font-size:10px!important;text-transform:uppercase;letter-spacing:1px;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#1b2a3b!important;font-family:'JetBrains Mono',monospace!important;font-size:22px!important;font-weight:700!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:#1b2a3b!important;}
.stButton>button{background:linear-gradient(90deg,#b87010,#966010);color:#fff;border:none;border-radius:6px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:.5px;transition:opacity .15s;}
.stButton>button:hover{opacity:.85;}
.stSelectbox,.stRadio{color:#1b2a3b;}
.stSelectbox>div>div{background:#ffffff;border-color:#c4d0de;color:#1b2a3b;}
hr{border-color:#c4d0de;}
.status-box{background:#ffffff;border:1px solid #c4d0de;border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#4e6880;line-height:1.8;white-space:pre;overflow-x:auto;}
.js-plotly-plot{border-radius:8px;}
</style>
""", unsafe_allow_html=True)
# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
_tmpl = go.layout.Template(layout=go.Layout(
    paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc",
    font=dict(family="JetBrains Mono, monospace", color="#4e6880", size=10),
    colorway=["#1758b0","#b87010","#1a7a45","#b82828","#5438a0","#987010"],
    xaxis=dict(gridcolor="#d8e2ee", zerolinecolor="#c4d0de"),
    yaxis=dict(gridcolor="#d8e2ee", zerolinecolor="#c4d0de"),
    legend=dict(bgcolor="rgba(255,255,255,0.96)", bordercolor="#c4d0de", borderwidth=1),
    margin=dict(l=50,r=20,t=40,b=40),
))
pio.templates["energy_light"] = _tmpl
pio.templates.default = "plotly+energy_light"
PT = "plotly+energy_light"
SCEN_COLORS = ["#1758b0","#987010","#b05828","#b82828","#5438a0","#1a7a45"]
HORIZONS    = ["1M","3M","6M","9M","12M"]
_PCFG = {"displayModeBar":False,"displaylogo":False}
# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defs = dict(result=None,agent=None,sel_horizon="1M",sel_bin=None,
                sel_scenario=None,sel_region=None,sel_driver=None,log=[])
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v
    _auth_init()
init_state()
# ── AGENT RUNNERS (cached, TTL=300s) ─────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_oil_agent():
    import oil_agent_v2 as oil
    msgs=[]
    result=oil.run(send=msgs.append)
    return result, msgs
@st.cache_data(ttl=300, show_spinner=False)
def run_ho_agent():
    import ho_agent as ho
    msgs=[]
    result=ho.run(send=msgs.append)
    return result, msgs

# ── LIVE SPOT PRICE FETCH (cached 30s — fast, no history) ────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_prices():
    """
    Fetch only the latest spot prices — runs in <2s and is cached for 30 seconds.
    Used to overlay fresh quotes on the Snapshot and Price History sections without
    re-running the full agent (which fetches a year of history).
    Returns a dict {name: price} with None for any failed tickers.
    """
    names = ["HO", "WTI", "RBOB", "VIX"]
    prices = {}
    for name in names:
        try:
            price, _dt, _src = _df.fetch_price(name, send=lambda _: None)
            prices[name] = price
        except Exception:
            prices[name] = None
    # Crack spread: HO ($/gal) × 42 gal/bbl − WTI ($/bbl)
    ho_p  = prices.get("HO")
    wti_p = prices.get("WTI")
    if ho_p is not None and wti_p is not None:
        prices["crack_spread"] = round(ho_p * 42 - wti_p, 2)
    else:
        prices["crack_spread"] = None
    # OVX — CBOE Crude Oil ETF Volatility Index (energy implied vol proxy)
    try:
        ovx_price, _dt, _src = _df.fetch_price("OVX", send=lambda _: None)
        prices["OVX"] = ovx_price
    except Exception:
        prices["OVX"] = None
    return prices
# ── HELPERS ───────────────────────────────────────────────────────────────────
def section(num, title, hint=""):
    st.markdown(f"""
    <div style="margin-top:28px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #c4d0de;
                display:flex;align-items:center;justify-content:space-between">
      <span style="font-size:12px;font-weight:700;color:#4e6880;text-transform:uppercase;letter-spacing:1.8px">
        {num} {title}
      </span>
      <span style="font-size:9px;color:#7a92a8;font-family:'JetBrains Mono',monospace">{hint}</span>
    </div>""", unsafe_allow_html=True)

def next_biz_days(n):
    dates,d=[],datetime.date.today()
    while len(dates)<n:
        d+=datetime.timedelta(days=1)
        if d.weekday()<5: dates.append(str(d))
    return dates

def interp_line(start,end,n):
    return [round(start+(end-start)*(i+1)/n,5) for i in range(n)]

def _pc(key): return f"chart_{key}"

_TH = ("padding:7px 12px;background:#eaeff6;color:#4e6880;font-family:'JetBrains Mono',monospace;"
       "font-size:9px;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid #c4d0de;"
       "text-align:center;white-space:nowrap;")
_TD = ("padding:6px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;"
       "border-bottom:1px solid #d8e2ee;text-align:center;")

def _html_table(headers, rows):
    """rows: list of lists; each cell is str or (str, extra_css)."""
    head = "".join(f'<th style="{_TH}">{html.escape(str(h))}</th>' for h in headers)
    body = ""
    for r in rows:
        cells = ""
        for c in r:
            txt, css = (c if isinstance(c, tuple) else (c, "color:#1b2a3b;"))
            cells += f'<td style="{_TD}{css}">{txt}</td>'
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #c4d0de;margin-bottom:16px">'
        f'<table style="width:100%;border-collapse:collapse;background:#ffffff">'
        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
        unsafe_allow_html=True)

def _prob_color(p):
    return "#b82828" if p >= 75 else "#987010" if p >= 50 else "#b87010" if p >= 25 else "#1a7a45"

def _is_admin():
    return bool(st.session_state.get("auth_is_admin"))

def _fmt_pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}%"

# ── Persistent store reads (cached; every write clears these) ────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load_inputs():
    return fs.load_market_inputs()

@st.cache_data(ttl=300, show_spinner=False)
def _load_forecasts():
    return fs.load_forecasts()

@st.cache_data(ttl=300, show_spinner=False)
def _load_curves():
    return fs.load_curves()

@st.cache_data(ttl=300, show_spinner=False)
def _load_atm_hist():
    return fs.load_atm_history()

@st.cache_data(ttl=300, show_spinner=False)
def _store_status():
    return fs.store_status()

@st.cache_data(ttl=21600, show_spinner=False)
def _front_history_long():
    try:
        return pd.DataFrame(_df.fetch_history("HO", days=730, send=lambda _: None))
    except Exception:
        return pd.DataFrame()

def _clear_store_caches():
    for f in (_load_inputs, _load_forecasts, _load_curves, _load_atm_hist, build_vol_context):
        f.clear()

def _param(inputs, key):
    v, _d = fs.latest_input(inputs, key, None)
    return float(cfg.get(key)) if v is None else v

# ── Volatility context: CME auto -> stored upload -> proxy -> realized ───────
@st.cache_data(ttl=900, show_spinner=False)
def build_vol_context(_result, run_key):
    log = []
    contracts = _result.get("ho_contracts", [])
    rate = _result.get("risk_free") or cfg.get("RISK_FREE_FALLBACK")
    surface, src = pd.DataFrame(), None
    try:
        surface, td = ve.fetch_cme_surface(contracts, rate, send=log.append)
        if not surface.empty:
            src = f"CME settlement ({td})"
            if f"{td}.csv" not in fs.backend().list_dir("vols/surface"):
                fs.save_vol_surface(surface, td, "CME auto")
    except Exception as e:
        log.append(f"  [CME] {type(e).__name__}: {e}")
    if surface.empty:
        try:
            stored, td = fs.load_latest_vol_surface()
            if not stored.empty:
                surface = stored
                kind = str(stored["source"].iloc[0]) if "source" in stored.columns else "stored"
                src = f"CME settlement ({kind}, {td})"
        except Exception as e:
            log.append(f"  [store] vol surface: {type(e).__name__}")
    atm = ve.atm_from_surface(surface, contracts) if not surface.empty else {}
    ovx = fetch_live_prices().get("OVX")
    proxy = ve.proxy_vols(contracts, ovx, _result.get("returns", []), _result.get("wti_returns", []))
    vols = ve.resolve_contract_vols(contracts, (atm, src or "CME"), proxy, _result.get("returns", []))
    rv = {c["label"]: {w: ve.realized_vol([h["price"] for h in c.get("history", [])], w)
                       for w in (10, 20, 30)} for c in contracts}
    return {"surface": surface, "surface_src": src, "atm": atm, "proxy": proxy,
            "vols": vols, "rv": rv, "ovx": ovx, "log": log}

def _contracts_with_vol(result, vctx, implied=True):
    out = []
    for c in result.get("ho_contracts", []):
        c = dict(c)
        sig, src = vctx["vols"].get(c["label"], (None, None)) if vctx else (None, None)
        if implied and sig:
            c["sigma_daily"], c["vol_src"] = sig / np.sqrt(252), src
        else:
            c["vol_src"] = "Realized"
        out.append(c)
    return out

def _scenarios(result):
    """Scenario list shared by 03B, 04 and 10 (edited values kept in session)."""
    return st.session_state.get("scen_list") or result.get("scenario_presets", [])

def _base_vol(result):
    return float(result.get("sigma_daily", 0.015)) * np.sqrt(252)

# ── Forecast autosave (first forecast of the day is the record) ──────────────
def _autosave_snapshot(result, vctx, force=False):
    if result.get("agent") != "ho" or not (cfg.get("FORECAST_AUTOSAVE") or force):
        return None
    key = f"{datetime.date.today()}|{result.get('run_dir')}"
    if st.session_state.get("_snap_key") == key and not force:
        return st.session_state.get("_snap_res")
    today = str(datetime.date.today())
    curve_rows = [{"date": today, "contract": c["label"], "expiry": c["expiry_date"],
                   "price": c["fwd_price"], "source": c.get("price_source")}
                  for c in result.get("ho_contracts", []) if c.get("price_source") != "estimated"]
    atm_rows = []
    for c in result.get("ho_contracts", []):
        sig, src = vctx["vols"].get(c["label"], (None, None))
        rvs = vctx["rv"].get(c["label"], {})
        atm_rows.append({"date": today, "contract": c["label"], "expiry": c["expiry_date"],
                         "futures": c["fwd_price"], "iv_atm": sig, "iv_source": src,
                         "rv20": rvs.get(20), "rv30": rvs.get(30), "ovx": vctx.get("ovx")})
    try:
        res = fs.save_daily_snapshot(result, curve_rows, atm_rows, vctx["vols"],
                                     user=st.session_state.get("auth_user", ""))
    except Exception as e:
        res = {"error": f"{type(e).__name__}"}
    st.session_state["_snap_key"], st.session_state["_snap_res"] = key, res
    if any(v for k, v in res.items() if k != "error"):
        _clear_store_caches()
    return res

def _get_front_contract_ticker() -> str:
    """
    Return the CME HO front-month contract ticker (e.g. 'HOU26').
    HO futures expire on the last business day of the month PRIOR to delivery,
    so the front delivery month is the first month whose prior-month expiry
    falls on or after today.
    CME month codes: F G H J K M N Q U V X Z (Jan–Dec)
    """
    import calendar as _cal
    _MONTH_LETTERS = {1:'F',2:'G',3:'H',4:'J',5:'K',6:'M',
                      7:'N',8:'Q',9:'U',10:'V',11:'X',12:'Z'}
    today = datetime.date.today()
    year, month = today.year, today.month
    for _ in range(14):                          # scan up to 14 months forward
        # expiry = last business day of (month - 1)
        if month == 1:
            exp_y, exp_m = year - 1, 12
        else:
            exp_y, exp_m = year, month - 1
        last_day = _cal.monthrange(exp_y, exp_m)[1]
        exp_d = datetime.date(exp_y, exp_m, last_day)
        while exp_d.weekday() >= 5:              # walk back to Friday if weekend
            exp_d -= datetime.timedelta(days=1)
        if exp_d >= today:
            return f"HO{_MONTH_LETTERS[month]}{str(year)[-2:]}"
        month += 1
        if month > 12:
            month = 1
            year += 1
    return "HO1"                                 # fallback — should never reach here

# ── ① SNAPSHOT ────────────────────────────────────────────────────────────────
def render_snapshot(result, agent):
    section("01", "SNAPSHOT")
    ho   = agent == "ho"
    md   = result.get("market_data", {})
    f    = result.get("forecast", {})
    ci   = result.get("ci_bands", {})
    ci1m = ci.get("1M", {})

    # Overlay live spot prices from the 30-second TTL cache where available
    live = fetch_live_prices()

    if ho:
        ho_disp  = live.get("HO")  or md.get("HO",  0)
        wti_disp = live.get("WTI") or md.get("WTI", 0)
        rb_disp  = live.get("RBOB") or md.get("RBOB", 0)
        cs_disp  = live.get("crack_spread") or md.get("crack_spread", 0)
        vix_disp = md.get("VIX", 0)   # VIX from cached agent result

        # Deltas vs cached values (shows movement since last full run)
        ho_delta  = round(ho_disp  - md.get("HO",  ho_disp),  4) if live.get("HO")  else None
        wti_delta = round(wti_disp - md.get("WTI", wti_disp), 2) if live.get("WTI") else None
        rb_delta  = round(rb_disp  - md.get("RBOB",rb_disp),  4) if live.get("RBOB") else None

        cols = st.columns(5)
        _src_tag = " ⟳" if any(live.get(k) for k in ("HO","WTI","RBOB")) else ""
        metrics = [
            ("HO Price" + _src_tag, f"${ho_disp:.4f}",           "$/gal",
             f"+${ho_delta:.4f}"  if ho_delta and ho_delta >= 0
             else f"${ho_delta:.4f}" if ho_delta else None),
            ("WTI" + _src_tag,       f"${wti_disp:.2f}",         "$/bbl",
             f"+${wti_delta:.2f}" if wti_delta and wti_delta >= 0
             else f"${wti_delta:.2f}" if wti_delta else None),
            ("RBOB" + _src_tag,      f"${rb_disp:.4f}",          "$/gal",
             f"+${rb_delta:.4f}"  if rb_delta  and rb_delta  >= 0
             else f"${rb_delta:.4f}"  if rb_delta  else None),
            ("Crack Spread",         f"${cs_disp:.2f}",           "$/bbl", None),
            ("Volatility (VIX)",     f"{vix_disp:.1f}",           "index", None),
        ]
        for col, (label, val, sub, delta) in zip(cols, metrics):
            col.metric(label, val, delta, help=sub)
    else:
        cols = st.columns(5)
        metrics = [
            ("WTI Live",      f"${f.get('current_wti',0):.2f}",                        "per barrel"),
            ("Brent",         f"${result.get('brent',0):.2f}",                          "per barrel"),
            ("1-Wk Forecast", f"${f.get('forecast_low',0)}–${f.get('forecast_high',0)}","90% CI"),
            ("Ann. Vol",      f"{f.get('annualised_vol',0):.1f}%",                      "historical sigma"),
            ("Direction",     f.get("direction","—"),                                   "model signal"),
        ]
        for col, (label, val, sub) in zip(cols, metrics):
            col.metric(label, val, sub)

    ci_width     = round((ci1m.get("ci95",[0,0])[1] - ci1m.get("ci95",[0,0])[0]), 4 if ho else 2)
    regime_label = result.get("regime","—") if ho else f.get("direction","—")
    st.caption(f"1M 95% CI range: **${ci_width}** · Regime: **{regime_label}**")


# ── ② PRICE HISTORY ──────────────────────────────────────────────────────────
def render_price_history(result, agent):
    """
    Price history — short-term (1H / 1D / 1W) priority, longer periods available.
    Intraday periods call data_fetcher.fetch_intraday_history() live each render.
    Longer periods (1M+) use the daily history cached in result.
    Dynamic y-axis, period-aware MA, synthetic data warning.
    v2.1 fixes:
      - "1 Day"  → 2d lookback / 13 bars  (last trading session only)
      - "1 Week" → 7d lookback / 168 bars  (distinct from 1 Day)
      - "1 Hour" → 2d lookback / 13 bars   (same session, narrower tick fmt)
      - Chart height raised from 340 → 480
    """
    front_ticker = _get_front_contract_ticker()
    section("02", "PRICE HISTORY",
            f"HO1 Front Contract: {front_ticker} · Short-term periods use live intraday fetch")
    ho          = agent == "ho"
    daily_hist  = result.get("history", [])
    ticker_name = "HO" if ho else "WTI"
    color       = "#b87010" if ho else "#1758b0"
    fill_rgba   = "184,112,16" if ho else "23,88,176"
    # Period options — "1 Hour" removed (unreliable intraday source)
    PERIOD_OPTS  = ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "All"]
    INTRADAY_SET = {"1 Day", "1 Week"}
    period = st.radio("Period", PERIOD_OPTS, horizontal=True, index=1, key="period_radio")
    period_s = sanitize_str(period)
    try:
        validate_enum(period_s, set(PERIOD_OPTS))
    except ValueError:
        period_s = "1 Week"
    is_synthetic = False
    # ── Intraday path (1D / 1W) ───────────────────────────────────────────────
    if period_s in INTRADAY_SET:
        intraday_cfg = {
            #        interval  lookback  tick_fmt
            "1 Day":  ("1h",  2,        "%b %d %H:%M" ),
            "1 Week": ("1h",  7,        "%b %d"       ),
        }
        interval, lookback, tick_fmt = intraday_cfg[period_s]
        rows, is_synthetic = _df.fetch_intraday_history(
            ticker_name, interval=interval, lookback_days=lookback
        )
        labels = [r["datetime"] for r in rows]
        prices = [float(r["price"]) for r in rows]
        xaxis_cfg = dict(
            type="date",
            tickformat=tick_fmt,
            rangeslider=dict(visible=True, bgcolor="#f5f8fc"),
        )
        ma_n = min(6, max(1, len(prices) // 4))
    # ── Daily path (1M and longer) ────────────────────────────────────────────
    else:
        cutoff_map  = {"1 Month": 30, "3 Months": 90, "6 Months": 180,
                       "1 Year": 365, "All": 9999}
        cutoff_days = cutoff_map.get(period_s, 90)
        cutoff_date = datetime.date.today() - datetime.timedelta(days=cutoff_days)
        filtered    = [r for r in daily_hist if str(r["date"]) >= str(cutoff_date)]
        if len(filtered) < 2:
            filtered = daily_hist[-10:]
        labels = [r["date"] for r in filtered]
        prices = [float(r["price"]) for r in filtered]
        is_synthetic = (result.get("is_synthetic_history", False)
                        or any(r.get("synthetic") for r in filtered))
        xaxis_cfg = dict(
            type="date",
            rangeslider=dict(visible=True, bgcolor="#f5f8fc"),
        )
        ma_n = min(20, max(1, len(prices) // 2)) if ho else min(7, max(1, len(prices) // 2))
    # ── Synthetic warning ─────────────────────────────────────────────────────
    if is_synthetic:
        st.error(
            "**DATA WARNING: Price history is SYNTHETIC (simulated) — all live data sources "
            "failed to respond.** The chart below does NOT reflect real market prices. "
            "Check the Run Log at the bottom of the page for details on which sources failed."
        )
    if len(prices) < 2:
        st.info("Not enough history data for the selected period.")
        return
    # ── Moving average ────────────────────────────────────────────────────────
    ma = [np.mean(prices[max(0, i - ma_n + 1):i + 1]) if i >= ma_n - 1 else None
          for i in range(len(prices))]
    # ── Dynamic y-axis — no hard-coded range ─────────────────────────────────
    p_min, p_max = min(prices), max(prices)
    if ho:
        pad    = max(0.05, (p_max - p_min) * 0.10)
        yrange = [round(p_min - pad, 4), round(p_max + pad, 4)]
    else:
        pad    = max(1.0, (p_max - p_min) * 0.10)
        yrange = [round(p_min - pad, 2), round(p_max + pad, 2)]
    # ── Chart — full-width container, height 520 ────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=prices, name="Price",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({fill_rgba},.06)",
        hovertemplate="$%{y:.4f}<extra></extra>" if ho else "$%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=labels, y=ma, name=f"{ma_n}pt MA",
        line=dict(color="#5438a0", width=1.5, dash="dot"),
        hovertemplate="MA: $%{y:.4f}<extra></extra>" if ho else "MA: $%{y:.2f}<extra></extra>"))
    fig.update_layout(
        template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=520,
        title=dict(
            text=(f"HO1 — {front_ticker} — Price History ({period_s})"
                  if ho else f"WTI Crude — Price History ({period_s})"),
            font=dict(size=12, color="#1b2a3b")),
        xaxis=xaxis_cfg,
        yaxis=dict(tickformat="$.4f" if ho else "$.2f", range=yrange),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    with st.container():
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("price_hist"))


# ── ③ PROBABILITY DISTRIBUTION ───────────────────────────────────────────────
def display_prob_table(result, sel_h):
    """Re-bin the model's fine-grid distribution into the display bins chosen by the user.
    Returns (edges, labels, {horizon: [probabilities]})."""
    import ho_agent as H
    grid = result.get("prob_grid")
    if not grid:                                   # WTI engine / legacy result
        pt = result.get("prob_table", {})
        labels = [r[0] for r in pt.get(HORIZONS[0], [])]
        return None, labels, {h: [r[1] for r in pt.get(h, [])] for h in HORIZONS}
    ss = st.session_state
    spot, sd = result["ho_price"], result.get("sigma_daily", 0.015)
    edges = None
    if ss.get("pt_manual"):
        lo, hi, stp = ss.get("pt_min"), ss.get("pt_max"), ss.get("pt_step")
        if lo and hi and stp and hi > lo and 1 <= (hi - lo) / stp <= 60:
            edges = [round(float(x), 4) for x in np.arange(lo, hi + stp / 2, stp)]
    if not edges:
        edges = H.make_display_edges(spot, sd, H.HORIZON_DAYS[sel_h],
                                     ss.get("pt_sigma"), ss.get("pt_bins"))
    labels = H.edge_labels(edges)
    return edges, labels, {h: H.grid_bin_probs(grid[h], edges) for h in HORIZONS}


def render_prob_dist(result, agent, sel_h, sel_bin):
    section("03","PROBABILITY DISTRIBUTION","Horizon selector in sidebar · range adapts to horizon")
    ho   = agent=="ho"
    if ho and result.get("prob_grid"):
        ss = st.session_state
        ss.setdefault("pt_sigma", float(cfg.get("PROB_TABLE_SIGMA_RANGE")))
        ss.setdefault("pt_bins", int(cfg.get("PROB_TABLE_BINS")))
        k1, k2, k3 = st.columns([2, 2, 1])
        k1.slider(f"Table range (± σ of {sel_h} move)", 1.0, 4.0, step=0.5, key="pt_sigma",
                  help="Narrower = focus on the likely zone around today's price. "
                       "Tails are always shown as '<' and '>' rows so totals stay 100%.")
        k2.slider("Number of price bins", 4, 20, key="pt_bins")
        k3.checkbox("Manual range", key="pt_manual")
        if ss.get("pt_manual"):
            import ho_agent as H
            auto = H.make_display_edges(result["ho_price"], result.get("sigma_daily", 0.015),
                                        H.HORIZON_DAYS[sel_h], ss.get("pt_sigma"), ss.get("pt_bins"))
            m1, m2, m3 = st.columns(3)
            m1.number_input("Min ($/gal)", value=float(auto[0]), step=0.05, format="%.2f", key="pt_min")
            m2.number_input("Max ($/gal)", value=float(auto[-1]), step=0.05, format="%.2f", key="pt_max")
            m3.number_input("Bin width ($/gal)", value=float(round(auto[1]-auto[0], 4)) if len(auto) > 1 else 0.10,
                            min_value=0.01, step=0.01, format="%.2f", key="pt_step")
    edges, labels, table = display_prob_table(result, sel_h)
    c1,c2 = st.columns(2)
    rows = list(zip(labels, table.get(sel_h, [])))
    if not rows: return
    bins  = [r[0] for r in rows]
    probs = [round(r[1]*100,2) for r in rows]
    maxP  = max(probs)
    colors = []
    for b,p in zip(bins,probs):
        if sel_bin and b==sel_bin: colors.append("#1758b0")
        elif sel_bin:              colors.append("rgba(23,88,176,.2)")
        elif p==maxP:              colors.append("#b87010")
        else:                      colors.append("#1758b0" if ho else "#b87010")
    with c1:
        fig=go.Figure(go.Bar(y=bins,x=probs,orientation="h",marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=360,
            title=dict(text=f"Probability by Price Bin — {sel_h}",font=dict(size=11,color="#1b2a3b")),
            xaxis=dict(title="Probability (%)",ticksuffix="%"),
            yaxis=dict(autorange="reversed"),bargap=0.15,showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("prob_bar"))
    with c2:
        cum=0; cdf=[]
        for _,p in rows: cum+=p; cdf.append(round(cum*100,2))
        fig=go.Figure(go.Scatter(x=bins,y=cdf,mode="lines+markers",
            line=dict(color="#5438a0",width=2),marker=dict(size=4,color="#5438a0"),
            fill="tozeroy",fillcolor="rgba(84,56,160,.07)",
            hovertemplate="%{x}: P(<=) = %{y:.1f}%<extra></extra>"))
        fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=360,
            title=dict(text=f"Cumulative Distribution — {sel_h}",font=dict(size=11,color="#1b2a3b")),
            yaxis=dict(title="Cumulative Probability (%)",ticksuffix="%",range=[0,100]),
            xaxis=dict(title="Price Range"))
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("cdf"))
    ev = result.get("ev_by_horizon",{})
    if ev:
        st.markdown("**Expected Value (EV) by Horizon** — probability-weighted average price")
        ev_cols = st.columns(len(HORIZONS))
        for col,h in zip(ev_cols,HORIZONS):
            col.metric(h,f"${ev.get(h,0):.4f}" if ho else f"${ev.get(h,0):.2f}")
    st.markdown("**Probability Table** — all horizons")
    _render_prob_table(labels, table, sel_h, sel_bin)
    if ho:
        ls = result.get("lognorm_shape",{})
        if ls:
            st.markdown("**Log-Normal HO Price Distribution Shape (1M horizon)**")
            c_a,c_b = st.columns([3,1])
            with c_a:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=ls["x"],y=ls["y"],mode="lines",
                    line=dict(color="#b87010",width=2),fill="tozeroy",
                    fillcolor="rgba(184,112,16,.12)",
                    hovertemplate="Price: $%{x:.4f}<br>PDF: %{y:.5f}<extra></extra>"))
                fig.add_vline(x=ls["mean"],  line=dict(color="#1758b0",dash="dash",width=1.5),
                              annotation_text=f"Mean ${ls['mean']:.4f}",
                              annotation_font=dict(color="#1758b0",size=9))
                fig.add_vline(x=ls["median"],line=dict(color="#1a7a45",dash="dot",width=1.5),
                              annotation_text=f"Median ${ls['median']:.4f}",
                              annotation_font=dict(color="#1a7a45",size=9))
                fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=260,
                    title=dict(text="Log-Normal PDF — shape, skewness, kurtosis",font=dict(size=11,color="#1b2a3b")),
                    xaxis=dict(title="HO Price ($/gal)"),yaxis=dict(title="Probability Density"),showlegend=False)
                st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("lognorm"))
            with c_b:
                st.metric("Mean",    f"${ls['mean']:.4f}")
                st.metric("Median",  f"${ls['median']:.4f}")
                st.metric("Skewness",f"{ls['skewness']:.3f}")
                st.metric("Kurtosis",f"{ls['kurtosis']:.3f}")

def _render_prob_table(labels, table, sel_h, sel_bin):
    if not labels: return
    rows = []
    for i, b in enumerate(labels):
        sel = sel_bin == b
        bg = "background:#e2edf8;" if sel else ""
        r = [(b, f"{bg}color:#1b2a3b;font-weight:{'700' if sel else '400'}")]
        for h in HORIZONS:
            col = table.get(h, [])
            pct = round(col[i]*100, 1) if i < len(col) else 0.0
            top = h == sel_h and col and pct == round(max(col)*100, 1)
            r.append((f"{pct:.1f}%", f"{bg}color:{'#b87010' if top else '#1b2a3b'}"))
        rows.append(r)
    _html_table(["Bin"] + HORIZONS, rows)


# ── ③B SCENARIO IMPACT ───────────────────────────────────────────────────────
def render_scenario_impact(result, sel_h):
    section("03B", "SCENARIO IMPACT ON PROBABILITY DISTRIBUTION",
            "If X happens → numbers · preset shock sizes = historical 1-month percentile moves")
    import ho_agent as H
    presets = result.get("scenario_presets") or []
    grid = result.get("prob_grid", {}).get(sel_h)
    if not presets or not grid:
        st.info("Scenario analysis needs the HO engine with WTI history.")
        return
    md = result.get("market_data", {})
    ho, wti, crack = result["ho_price"], md.get("WTI"), md.get("crack_spread")
    base_vol = _base_vol(result)
    st.caption(
        f"Edit any shock below — every table and chart updates. HO under a scenario uses the crack identity "
        f"HO = (WTI + crack) / {cfg.GALLONS_PER_BARREL}. Today: WTI **${wti or 0:.2f}**, crack **${crack or 0:.2f}/bbl**, "
        f"model vol **{base_vol*100:.1f}%**. Shocks are assumed realised by the {sel_h} horizon.")
    df0 = pd.DataFrame([{"Scenario": p["name"], "WTI change (%)": p["wti_pct"],
                         "Crack change ($/bbl)": p["crack_chg"], "Vol change (pts)": p["vol_pts"],
                         "Rationale": p["why"]} for p in presets] +
                       [{"Scenario": "Custom", "WTI change (%)": 0.0, "Crack change ($/bbl)": 0.0,
                         "Vol change (pts)": 0.0, "Rationale": "Your own what-if"}])
    ed = st.data_editor(df0, key="scen_editor", hide_index=True, num_rows="fixed",
                        disabled=["Scenario", "Rationale"], use_container_width=True,
                        column_config={
                            "WTI change (%)": st.column_config.NumberColumn(format="%.2f", min_value=-90.0, max_value=300.0),
                            "Crack change ($/bbl)": st.column_config.NumberColumn(format="%.2f", min_value=-100.0, max_value=200.0),
                            "Vol change (pts)": st.column_config.NumberColumn(format="%.2f",
                                                                              min_value=-base_vol*100 + 2, max_value=200.0)})
    scen = [{"name": r["Scenario"], "wti_pct": float(r["WTI change (%)"] or 0),
             "crack_chg": float(r["Crack change ($/bbl)"] or 0),
             "vol_pts": float(r["Vol change (pts)"] or 0), "why": r["Rationale"]}
            for _, r in ed.iterrows()]
    st.session_state["scen_list"] = scen

    edges, labels, _tbl = display_prob_table(result, sel_h)
    med = H.grid_quantile(grid, 0.5)
    cdf = lambda x: H.grid_cdf(grid, x)
    qf  = lambda u: H.grid_quantile(grid, u)
    t1, t2 = st.columns(2)
    thr_lo = t1.number_input("Show P(price below) $/gal", value=round(float(qf(0.25)), 2),
                             step=0.05, format="%.2f", key="scen_thr_lo")
    thr_hi = t2.number_input("Show P(price above) $/gal", value=round(float(qf(0.75)), 2),
                             step=0.05, format="%.2f", key="scen_thr_hi")
    us = np.linspace(0.0025, 0.9975, 400)
    out, base_ev = [], None
    for s in scen:
        k, vr, vs = se.scenario_factors(ho, wti, crack, base_vol, s)
        q = lambda u: se.scen_quantile(qf, u, med, k, vr)
        ev = float(np.mean([q(u) for u in us]))
        base_ev = ev if base_ev is None else base_ev
        out.append(dict(s=s, k=k, vs=vs, ho_s=ho * k, ev=ev, p10=q(.10), p50=q(.50), p90=q(.90),
                        plo=se.scen_cdf(cdf, thr_lo, med, k, vr) * 100,
                        phi=(1 - se.scen_cdf(cdf, thr_hi, med, k, vr)) * 100,
                        bins=se.bin_probs(cdf, edges, med, k, vr) * 100))
    rows = []
    for i, o in enumerate(out):
        c = SCEN_COLORS[i % len(SCEN_COLORS)]
        d = o["ev"] - base_ev
        rows.append([(o["s"]["name"], f"color:{c};font-weight:700;text-align:left"),
                     f'{o["s"]["wti_pct"]:+.1f}%', f'{o["s"]["crack_chg"]:+.2f}', f'{o["vs"]*100:.1f}%',
                     f'${o["ho_s"]:.4f}', f'${o["p50"]:.4f}', f'${o["ev"]:.4f}',
                     f'${o["p10"]:.4f} – ${o["p90"]:.4f}',
                     (f'{o["plo"]:.1f}%', "color:#b82828;"), (f'{o["phi"]:.1f}%', "color:#1a7a45;"),
                     (f'{d:+.4f}', f"color:{'#1a7a45' if d >= 0 else '#b82828'};")])
    st.markdown(f"**Scenario outcomes — {sel_h} horizon**")
    _html_table(["Scenario", "WTI", "Crack Δ", "Vol", "HO implied", "Median", "EV",
                 "80% range", f"P(< ${thr_lo:.2f})", f"P(> ${thr_hi:.2f})", "EV Δ vs Base"], rows)

    fig = go.Figure()
    for i, o in enumerate(out):
        fig.add_trace(go.Scatter(x=labels, y=o["bins"], mode="lines+markers", name=o["s"]["name"],
                                 line=dict(color=SCEN_COLORS[i % len(SCEN_COLORS)],
                                           width=3 if i == 0 else 1.8, dash=None if i == 0 else "dot"),
                                 hovertemplate="%{x}: %{y:.1f}%<extra>" + o["s"]["name"] + "</extra>"))
    fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=340,
                      title=dict(text=f"Probability by Price Bin under each Scenario — {sel_h}",
                                 font=dict(size=11, color="#1b2a3b")),
                      yaxis=dict(title="Probability (%)", ticksuffix="%"), xaxis=dict(title="Price range"),
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("scen_impact"))

    rows = []
    for j, lbl in enumerate(labels):
        r = [(lbl, "color:#1b2a3b;font-weight:600")]
        for i, o in enumerate(out):
            p = o["bins"][j]
            if i == 0:
                r.append(f"{p:.1f}%")
            else:
                dp = p - out[0]["bins"][j]
                col = "#1a7a45" if dp > 0.05 else "#b82828" if dp < -0.05 else "#7a92a8"
                r.append((f'{p:.1f}% <span style="color:{col};font-size:9px">({dp:+.1f})</span>', "color:#1b2a3b;"))
        rows.append(r)
    with st.expander("Full distribution by scenario (Δ pp vs Base)", expanded=False):
        _html_table(["Bin"] + [o["s"]["name"] for o in out], rows)


# ── ⑤ VOLATILITY ─────────────────────────────────────────────────────────────
def render_volatility(result, vctx=None):
    ho = result.get("agent") == "ho"
    section("05","VOLATILITY",
            "CME settlement implied vol · ATM implied vs realized · term structure · vol by strike"
            if ho else "Rolling 10-day & 30-day annualised vol · OVX implied vol proxy (live)")
    vh = result.get("vol_heatmap",[])
    if not vh:
        st.info("Insufficient history for volatility (need > 11 trading days)")
        return
    contracts = result.get("ho_contracts", [])
    front = contracts[0] if contracts else None
    live = fetch_live_prices()
    ovx_val = live.get("OVX")

    # ── Source banner + headline metrics ──────────────────────────────────────
    iv_front, iv_src = (vctx["vols"].get(front["label"], (None, None)) if (ho and vctx and front) else (None, None))
    if ho:
        if iv_src and iv_src.startswith("CME"):
            st.success(f"Implied vol source: **{iv_src}** — Black-76 settlement vols from CME option settlements.")
        elif iv_src == "Proxy":
            st.warning("Implied vol source: **Proxy** (OVX × HO/WTI realized-vol ratio). CME settlement vols "
                       "were not reachable and none have been uploaded — an admin can upload the CME file below.")
        else:
            st.warning("No implied vol available — showing realized vol only.")
        rvf = vctx["rv"].get(front["label"], {}) if (vctx and front) else {}
        m = st.columns(5)
        m[0].metric(f"ATM IV {front['code'] if front else ''}", _fmt_pct(iv_front*100 if iv_front else None),
                    help=f"Source: {iv_src or 'n/a'}")
        m[1].metric("Realized 10d", _fmt_pct(rvf.get(10)*100 if rvf.get(10) else None))
        m[2].metric("Realized 20d", _fmt_pct(rvf.get(20)*100 if rvf.get(20) else None))
        m[3].metric("Realized 30d", _fmt_pct(rvf.get(30)*100 if rvf.get(30) else None))
        prem = (iv_front - rvf[30]) * 100 if (iv_front and rvf.get(30)) else None
        m[4].metric("IV − RV30 premium", f"{prem:+.1f} pts" if prem is not None else "—",
                    help="Positive = options price more movement than recently realized")

    df_full = pd.DataFrame(vh)
    df_full["date"] = pd.to_datetime(df_full["date"])
    period_opts = ["1M","3M","6M","1Y"]
    period_days = {"1M":30,"3M":90,"6M":180,"1Y":365}
    period = st.radio("Volatility period",period_opts,index=1,
                      horizontal=True,key="vol_period_radio")
    try:    validate_enum(sanitize_str(period),set(period_opts))
    except ValueError: period="3M"
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=period_days[period])
    df     = df_full[df_full["date"]>=cutoff].copy()
    if df.empty: df = df_full.tail(10).copy()

    # Realized from the FRONT CONTRACT when available (else continuous HO=F)
    base_hist = front["history"] if (front and len(front.get("history", [])) > 32) else result.get("history", [])
    r30 = ve.rolling_realized(base_hist, 30)
    r30 = r30[r30["date"] >= cutoff]

    # Stored daily ATM IV history (builds up from each day's snapshot)
    iv_hist = pd.DataFrame()
    if ho and front:
        ah = _load_atm_hist()
        if not ah.empty and "iv_atm" in ah.columns:
            ah = ah.dropna(subset=["iv_atm"]).copy()
            ah["date"] = pd.to_datetime(ah["date"])
            ah = ah.sort_values(["date", "expiry"]).groupby("date").first().reset_index()   # front each day
            iv_hist = ah[ah["date"] >= cutoff]

    all_vals = list(df["vol"]) + list(r30["rv"]*100)
    if not iv_hist.empty: all_vals += list(iv_hist["iv_atm"]*100)
    if ovx_val: all_vals.append(ovx_val)
    if iv_front: all_vals.append(iv_front*100)
    y_upper = round(max(all_vals) * 1.10, 2) if all_vals else None
    y_lower = round(max(0, min(all_vals) * 0.80), 2) if all_vals else 0
    avg     = float(df["vol"].mean())
    line_color = "#b87010" if ho else "#1758b0"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"],y=df["vol"],mode="lines",line=dict(color=line_color,width=2),
        name="Realized 10d", hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>10d RV</extra>"))
    if not r30.empty:
        fig.add_trace(go.Scatter(x=r30["date"],y=r30["rv"]*100,mode="lines",
            line=dict(color="#5438a0",width=1.8,dash="dot"), name="Realized 30d",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>30d RV</extra>"))
    if not iv_hist.empty:
        fig.add_trace(go.Scatter(x=iv_hist["date"], y=iv_hist["iv_atm"]*100, mode="lines+markers",
            line=dict(color="#1a7a45", width=2), marker=dict(size=5), name="ATM Implied (front, stored)",
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>ATM IV</extra>"))
    if iv_front:
        fig.add_trace(go.Scatter(x=[pd.Timestamp.today().normalize()], y=[iv_front*100], mode="markers",
            marker=dict(color="#1a7a45", size=12, symbol="diamond"), name=f"ATM IV today ({iv_src})",
            hovertemplate="Today: %{y:.1f}%<extra>ATM IV</extra>"))
    if ovx_val:
        fig.add_hline(y=ovx_val,line=dict(color="#987010",width=1,dash="dashdot"),
            annotation_text=f"OVX {ovx_val:.1f}", annotation_font=dict(color="#987010",size=9))
    fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=320,
        title=dict(text=f"Front-Month ATM Implied vs Realized Volatility — {period}",
                   font=dict(size=11,color="#1b2a3b")),
        xaxis=dict(title="",type="date"),
        yaxis=dict(title="Ann. Vol (%)",ticksuffix="%",range=[y_lower,y_upper] if y_upper else None),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("vol"))
    if not ho:
        return

    # ── Term structure: vol for each month going forward ─────────────────────
    labels = [c["label"] for c in contracts]
    ivs  = [vctx["vols"].get(l, (None, None))[0] for l in labels]
    srcs = [vctx["vols"].get(l, (None, None))[1] or "" for l in labels]
    rv30 = [vctx["rv"].get(l, {}).get(30) for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[v*100 if v else None for v in ivs], name="ATM Implied",
        marker_color=["#1a7a45" if s_.startswith("CME") else "#987010" if s_ == "Proxy" else "#7a92a8" for s_ in srcs],
        text=[f"{v*100:.1f}%" if v else "" for v in ivs], textposition="outside",
        customdata=srcs, hovertemplate="%{x}: %{y:.1f}%<br>%{customdata}<extra>ATM IV</extra>"))
    fig.add_trace(go.Scatter(x=labels, y=[v*100 if v else None for v in rv30], name="Realized 30d (contract)",
        mode="lines+markers", line=dict(color="#5438a0", width=2, dash="dot"),
        hovertemplate="%{x}: %{y:.1f}%<extra>RV30</extra>"))
    fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=300,
        title=dict(text="Volatility Term Structure — ATM Implied vs Realized by Contract Month",
                   font=dict(size=11, color="#1b2a3b")),
        yaxis=dict(title="Ann. Vol (%)", ticksuffix="%"), bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("vol_term"))
    st.caption("Green = CME settlement vol · Amber = proxy · Gray = realized fallback")

    # ── Vol by strike (smile) ─────────────────────────────────────────────────
    surf = vctx.get("surface", pd.DataFrame())
    if surf is not None and not surf.empty and "strike" in surf.columns:
        avail = [l for l in labels if l in set(surf["contract"])]
        pick = st.multiselect("Compare vol by strike — contracts", avail, default=avail[:3], key="smile_pick")
        mode = st.radio("Strike axis", ["Strike ($/gal)", "Moneyness (K/F)"], horizontal=True, key="smile_axis")
        fig = go.Figure()
        cmap = {c["label"]: c for c in contracts}
        for i, l in enumerate(pick):
            d = surf[surf["contract"] == l].groupby("strike", as_index=False)["iv"].mean().sort_values("strike")
            F = cmap[l]["fwd_price"]
            x = d["strike"] if mode.startswith("Strike") else d["strike"] / F
            fig.add_trace(go.Scatter(x=x, y=d["iv"]*100, mode="lines+markers", name=l,
                line=dict(color=SCEN_COLORS[i % len(SCEN_COLORS)], width=2), marker=dict(size=4),
                hovertemplate="K %{x:.3f}: %{y:.1f}%<extra>" + l + "</extra>"))
            fig.add_vline(x=F if mode.startswith("Strike") else 1.0,
                          line=dict(color=SCEN_COLORS[i % len(SCEN_COLORS)], width=1, dash="dot"))
        fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=320,
            title=dict(text=f"Settlement Vol by Strike — {vctx.get('surface_src')}", font=dict(size=11, color="#1b2a3b")),
            xaxis=dict(title=mode), yaxis=dict(title="Implied Vol (%)", ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("vol_smile"))
        st.caption("Dotted lines = each contract's futures price (ATM). OTM puts below, OTM calls above.")
    else:
        st.info("Vol by strike needs CME settlement data (auto-fetch or admin upload).")

    if _is_admin():
        with st.expander("Admin — upload CME settlement vol file", expanded=False):
            st.caption("CSV/XLSX from CME / QuikStrike. Required: a contract/month column and either a vol "
                       "column (% or decimal) or a settlement premium column (inverted with Black-76). "
                       "Optional: strike, put/call, futures price. No strike = ATM vol.")
            up = st.file_uploader("CME vol file", type=["csv", "xlsx"], key="cme_upload")
            td = st.date_input("Trade date", value=datetime.date.today() - datetime.timedelta(days=1), key="cme_td")
            if up is not None and st.button("Parse & save", key="cme_save"):
                try:
                    if up.size > _MAX_PAYLOAD * 64:
                        raise ValueError("File too large")
                    raw = pd.read_excel(up) if up.name.lower().endswith("xlsx") else pd.read_csv(up)
                    surf_new, skipped = ve.parse_vol_upload(raw, contracts, result.get("risk_free") or cfg.get("RISK_FREE_FALLBACK"))
                    ok = fs.save_vol_surface(surf_new, str(td), "Upload")
                    _clear_store_caches()
                    (st.success if ok else st.error)(
                        f"{len(surf_new)} vol points saved ({skipped} rows skipped)." if ok else "Save failed — check store status.")
                except Exception as e:
                    st.error(f"Could not parse file: {sanitize_str(str(e))}")


# ── ④ KO PROBABILITY TABLE ───────────────────────────────────────────────────
def render_ko_table(result, agent, vctx=None):
    """P(each HO contract touches the KO barrier before expiry), using implied vol per
    contract, plus the same probability under each numeric scenario."""
    if agent != "ho":
        return
    section("04", "KO PROBABILITY BY CONTRACT",
            "P(touch KO before expiry) · implied vol per contract · scenario map")
    if not result.get("ho_contracts"):
        st.info("Forward curve data unavailable — re-run the HO engine.")
        return
    import ho_agent as _ho_mod
    ho_spot = result.get("market_data", {}).get("HO", result.get("ho_price"))
    k1, k2 = st.columns([1, 2])
    ko_price = k1.number_input("KO Price ($/gal)", min_value=0.01, max_value=20.0,
        value=float(result.get("ko_price_default", round(ho_spot * cfg.get("KO_DEFAULT_PCT_OF_SPOT"), 4))),
        step=0.01, format="%.4f", key="ko_price_input",
        help="Knock-Out barrier — probability of touching this level at any time before each contract's expiry.")
    vmode = k2.radio("Volatility input", ["Implied (CME settlement / proxy)", "Realized (historical)"],
                     horizontal=True, key="ko_vol_mode")
    contracts = _contracts_with_vol(result, vctx, implied=vmode.startswith("Implied"))
    base_rows = _ho_mod.compute_ko_probabilities(contracts, ko_price)
    scen = [s for s in _scenarios(result) if s["name"] != "Base"]
    md = result.get("market_data", {})
    wti, crack = md.get("WTI"), md.get("crack_spread")
    base_vol = _base_vol(result)

    direction = "BELOW" if ko_price < ho_spot else "ABOVE"
    st.caption(f"KO **${ko_price:.4f}** — **{direction}** front **${ho_spot:.4f}** "
               f"({abs(ko_price-ho_spot)/ho_spot*100:.1f}% away) · GBM reflection principle, zero drift "
               f"(risk-neutral futures) · scenarios shift each futures price by the scenario HO move and add the vol shock.")
    rows, heat = [], []
    for c, r in zip(contracts, base_rows):
        sig_ann = c["sigma_daily"] * np.sqrt(252)
        row = [(r["label"], "color:#1b2a3b;font-weight:600"), (r["expiry"], "color:#4e6880;"),
               f'${r["fwd_price"]:.4f}' + ("" if c.get("price_source") != "estimated" else "*"),
               f"{sig_ann*100:.1f}%", (c["vol_src"], "color:#4e6880;font-size:9px;"),
               (f'{r["ko_prob"]:.1f}%', f'color:{_prob_color(r["ko_prob"])};font-weight:700;')]
        hrow = [r["ko_prob"]]
        for s in scen:
            k, vr, vs = se.scenario_factors(ho_spot, wti, crack, base_vol, s)
            sig_s = max(0.02, sig_ann + s["vol_pts"] / 100) / np.sqrt(252)
            p = _ho_mod.barrier_touch_prob(c["fwd_price"] * k, ko_price, c["t_days"], sig_s) * 100
            row.append((f"{p:.1f}%", f"color:{_prob_color(p)};"))
            hrow.append(p)
        rows.append(row); heat.append(hrow)
    _html_table(["Contract", "Expiry", "Futures", "Vol", "Vol source", "P(KO) Base"] +
                [f"P(KO) {s['name']}" for s in scen], rows)
    if any(c.get("price_source") == "estimated" for c in contracts):
        st.caption("* futures price estimated (contract quote unavailable)")

    fig = go.Figure(go.Heatmap(
        z=heat, x=["Base"] + [s["name"] for s in scen], y=[c["label"] for c in contracts],
        colorscale=[[0, "#eaf5ee"], [0.25, "#b9dcc5"], [0.5, "#e8c77f"], [0.75, "#d98a4a"], [1, "#b82828"]],
        zmin=0, zmax=100, text=[[f"{v:.0f}%" for v in r] for r in heat], texttemplate="%{text}",
        hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="P(KO)", font=dict(size=9)), ticksuffix="%")))
    fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=420,
        title=dict(text=f"P(Touch KO ${ko_price:.4f}) — Contract × Scenario", font=dict(size=11, color="#1b2a3b")),
        yaxis=dict(autorange="reversed"), margin=dict(l=90, r=40, t=40, b=60))
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("ko_scen"))

    # ── Price-shock × vol grid for one contract ──────────────────────────────
    st.markdown("**KO sensitivity map** — probability across price moves and implied-vol levels")
    lbl = st.selectbox("Contract", [c["label"] for c in contracts], key="ko_grid_contract")
    c = next(x for x in contracts if x["label"] == lbl)
    sig_ann = c["sigma_daily"] * np.sqrt(252)
    n = int(cfg.get("KO_GRID_POINTS"))
    move = 2 * sig_ann * np.sqrt(c["t_days"] / 252)
    shocks = np.linspace(-move, move, n)
    rets = np.asarray(result.get("returns", []))
    rv_roll = [np.std(rets[i-30:i], ddof=1)*np.sqrt(252) for i in range(30, len(rets)+1)] if len(rets) > 35 else [sig_ann]
    v_lo = min(float(np.percentile(rv_roll, 5)), sig_ann); v_hi = max(float(np.percentile(rv_roll, 95)), sig_ann)
    vols = np.linspace(max(0.02, v_lo), v_hi, n)
    z = [[_ho_mod.barrier_touch_prob(c["fwd_price"]*(1+x), ko_price, c["t_days"], v/np.sqrt(252))*100
          for x in shocks] for v in vols]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{x*100:+.0f}%" for x in shocks], y=[f"{v*100:.0f}%" for v in vols],
        colorscale=[[0, "#eaf5ee"], [0.5, "#e8c77f"], [1, "#b82828"]], zmin=0, zmax=100,
        text=[[f"{v:.0f}" for v in r] for r in z], texttemplate="%{text}",
        hovertemplate="Price move %{x} · vol %{y}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="P(KO)", font=dict(size=9)), ticksuffix="%")))
    fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=380,
        title=dict(text=f"{lbl}: P(KO) by immediate futures move (±2σ) × vol (historical 5th–95th pct range)",
                   font=dict(size=11, color="#1b2a3b")),
        xaxis=dict(title="Immediate futures price move"), yaxis=dict(title="Annualised vol"))
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("ko_grid"))
    st.caption(f"Current: futures ${c['fwd_price']:.4f}, vol {sig_ann*100:.1f}% ({c['vol_src']}), "
               f"{c['t_days']} trading days to expiry.")


# ── ⑤B PROBABILITY AT EXPIRATION TABLE ───────────────────────────────────────
def render_expiry_distribution_table(result, agent, vctx=None):
    """
    Table 2: Probability at Expiration — 12-Month Forward.
    Rows: 13 monthly contracts.
    Columns: Contract, Expiry, Fwd Price + probability in each price range.
    Model: lognormal settlement distribution, zero-drift (risk-neutral).
    Modes: Predefined Ranges (default bins) or Custom Thresholds (user-defined).
    """
    if agent != "ho":
        return
    section("04B", "PROBABILITY AT EXPIRATION (12-MONTH FORWARD)",
            "Lognormal settlement distribution by contract — predefined or custom ranges")

    contracts = result.get("ho_contracts", [])
    if not contracts:
        st.info("Forward curve data unavailable — re-run the HO engine.")
        return

    import ho_agent as _ho_mod

    r_arr     = np.array(result.get("returns", []))
    sig_daily = float(np.std(r_arr, ddof=1)) if len(r_arr) > 5 else 0.015
    sig_daily = min(sig_daily, 0.80 / np.sqrt(252))
    ho_spot   = result.get("market_data", {}).get("HO", result.get("ho_price"))

    # ── Mode selector ─────────────────────────────────────────────────────────
    mode = st.radio(
        "Display mode",
        ["Predefined Ranges", "Custom Thresholds"],
        horizontal=True,
        key="expiry_dist_mode",
    )

    if mode == "Custom Thresholds":
        st.caption(
            "Enter up to 3 price thresholds to split the distribution. "
            "The table will show P(below T1), P(T1–T2), P(T2–T3), P(above T3)."
        )
        cc1, cc2, cc3 = st.columns(3)
        t1 = cc1.number_input("Threshold 1 ($/gal)", value=round(ho_spot * 0.80, 2),
                               min_value=0.01, max_value=20.0, step=0.05, format="%.2f",
                               key="ed_thresh1")
        t2 = cc2.number_input("Threshold 2 ($/gal)", value=round(ho_spot * 0.90, 2),
                               min_value=0.01, max_value=20.0, step=0.05, format="%.2f",
                               key="ed_thresh2")
        t3 = cc3.number_input("Threshold 3 ($/gal)", value=round(ho_spot, 2),
                               min_value=0.01, max_value=20.0, step=0.05, format="%.2f",
                               key="ed_thresh3")
        thresholds = sorted(set([t1, t2, t3]))   # deduplicate & sort
        bin_labels = (
            [f"<${thresholds[0]:.2f}"]
            + [f"${thresholds[i]:.2f}-${thresholds[i+1]:.2f}"
               for i in range(len(thresholds) - 1)]
            + [f">${thresholds[-1]:.2f}"]
        )
        bin_edges = [-np.inf] + list(thresholds) + [np.inf]
    else:
        mid_t = int(np.median([c["t_days"] for c in contracts]))
        _e = _ho_mod.make_display_edges(ho_spot, sig_daily, mid_t,
                                        st.session_state.get("pt_sigma"), st.session_state.get("pt_bins"))
        bin_edges  = [-np.inf] + _e + [np.inf]
        bin_labels = _ho_mod.edge_labels(_e)

    # Implied vol per contract when available (CME settlement -> proxy), else realized
    contracts = _contracts_with_vol(result, vctx, implied=True)
    st.caption("Vol per contract: " + ", ".join(sorted({c["vol_src"] or "Realized" for c in contracts})) +
               " · predefined ranges follow the Section 03 range/bin settings")
    fresh_rows = _ho_mod.compute_expiry_distributions(
        contracts, sig_daily, bin_edges, bin_labels
    )

    # ── Styled heatmap HTML table ─────────────────────────────────────────────
    th = (
        "padding:7px 10px;background:#eaeff6;color:#4e6880;"
        "font-family:'JetBrains Mono',monospace;font-size:9px;"
        "text-transform:uppercase;letter-spacing:.8px;"
        "border-bottom:1px solid #c4d0de;text-align:center;white-space:nowrap;"
    )
    td = (
        "padding:5px 10px;font-family:'JetBrains Mono',monospace;"
        "font-size:10px;border-bottom:1px solid #d8e2ee;text-align:center;"
    )

    fixed_hdrs = ["Contract", "Expiry", "Fwd Price"]
    head_html  = (
        "".join(f'<th style="{th}">{h}</th>' for h in fixed_hdrs)
        + "".join(f'<th style="{th}">{lbl}</th>' for lbl in bin_labels)
    )

    body_html = ""
    for r in fresh_rows:
        bp     = r["bin_probs"]
        max_bp = max(bp) if bp else 1.0
        cells  = (
            f'<td style="{td}color:#1b2a3b;font-weight:600">{r["label"]}</td>'
            f'<td style="{td}color:#4e6880">{r["expiry"]}</td>'
            f'<td style="{td}color:#1b2a3b">${r["fwd_price"]:.4f}</td>'
        )
        for p in bp:
            intensity = p / max_bp if max_bp > 0 else 0
            # Heat color: highest bin in each row = bright orange, moderate = blue, low = dim
            if intensity > 0.65:
                cell_bg = "rgba(184,112,16,0.18)"
                tc = "#987010"
                fw = "700"
            elif intensity > 0.35:
                cell_bg = "rgba(0,212,255,0.10)"
                tc = "#1758b0"
                fw = "400"
            else:
                cell_bg = "transparent"
                tc = "#7a92a8"
                fw = "400"
            cells += (
                f'<td style="{td}background:{cell_bg};color:{tc};font-weight:{fw}">'
                f'{p:.1f}%</td>'
            )
        body_html += f"<tr>{cells}</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #c4d0de;margin-bottom:16px">'
        f'<table style="width:100%;border-collapse:collapse;background:#ffffff">'
        f'<thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Orange = modal price range per contract · Blue = moderate probability · "
        "Gray = low probability · "
        "Model: lognormal, risk-neutral zero-drift (futures martingale)"
    )

    # ── Heatmap chart — contract × bin probability matrix ─────────────────────
    if fresh_rows and bin_labels:
        z_matrix  = [r["bin_probs"] for r in fresh_rows]
        x_labels  = bin_labels
        y_labels  = [r["label"] for r in fresh_rows]

        fig = go.Figure(go.Heatmap(
            z=z_matrix,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0,  "#f8fafc"],
                [0.35, "#c8ddf4"],
                [0.65, "#5888c8"],
                [0.85, "#b87010"],
                [1.0,  "#987010"],
            ],
            hovertemplate="Contract: %{y}<br>Range: %{x}<br>Probability: %{z:.1f}%<extra></extra>",
            showscale=True,
            colorbar=dict(
                title=dict(text="Prob (%)", font=dict(color="#4e6880", size=9)),
                tickfont=dict(color="#4e6880", size=9),
                bgcolor="#f5f8fc",
                bordercolor="#c4d0de",
            ),
        ))
        fig.update_layout(
            template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=360,
            title=dict(
                text="Settlement Probability Heatmap — Contract × Price Range",
                font=dict(size=10, color="#1b2a3b")),
            xaxis=dict(title="Price Range", tickfont=dict(size=9)),
            yaxis=dict(title="Contract", autorange="reversed", tickfont=dict(size=9)),
            margin=dict(l=100, r=80, t=40, b=80),
        )
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("expiry_heatmap"))


# ── ⑥ SCENARIO ────────────────────────────────────────────────────────────────
def render_scenario(result, agent, sel_scen):
    section("10","SCENARIO SIMULATION","Paths driven by the Section 03B numeric scenario shocks")
    sp   = result.get("scenario_paths",{})
    ho   = agent=="ho"
    md   = result.get("market_data",{})
    f    = result.get("forecast",{})
    spot = md.get("HO",result.get("ho_price")) if ho else f.get("current_wti",result.get("wti"))
    sigs = result.get("scenario_signals",{})
    if ho and sigs:
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        base_drift = sigs.get("base_dynamic_drift_ann",0)
        sc1.metric("Base Drift (ann.)", f"{base_drift:+.2f}%",
            help="Composite dynamic drift from all live signals (annualised)")
        sc2.metric("VIX Vol Mult", f"{sigs.get('vix_vol_mult',1):.2f}×",
            help="VIX current ÷ 20d rolling mean — scales scenario volatility")
        sc3.metric("Crack Signal", f"{sigs.get('crack_signal_ann',0):+.2f}%",
            help=f"Crack vs its 1-year median ${sigs.get('crack_median',0):.2f}/bbl (annualised drift contribution)")
        sc4.metric("Seasonal Signal", f"{sigs.get('seasonal_signal_ann',0):+.2f}%",
            help="This calendar month's historical average vs the 1-year average")
        eia_on = sigs.get("eia_enabled", False)
        sc5.metric("EIA Signal", f"{sigs.get('eia_signal_ann',0):+.2f}%" if eia_on else "Off",
            help="Weekly inventory draw (+) or build (−) contribution" if eia_on
                 else "EIA inventory disabled (EIA_ENABLED=false)")
        sig_names  = ["Crack Spread","VIX","Seasonal"] + (["EIA Inventory"] if eia_on else [])
        sig_values = [sigs.get("crack_signal_ann",0),sigs.get("vix_signal_ann",0),
                      sigs.get("seasonal_signal_ann",0)] + ([sigs.get("eia_signal_ann",0)] if eia_on else [])
        sig_colors = ["#1a7a45" if v>=0 else "#b82828" for v in sig_values]
        fig_sig = go.Figure(go.Bar(
            x=sig_names, y=sig_values, marker_color=sig_colors,
            text=[f"{v:+.2f}%" for v in sig_values], textposition="outside",
            hovertemplate="%{x}: %{y:+.2f}% ann.<extra></extra>"))
        fig_sig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=180,
            title=dict(text="Signal Decomposition — Drift Contribution (ann.%)",font=dict(size=10,color="#1b2a3b")),
            yaxis=dict(title="Drift (%)", ticksuffix="%"),
            showlegend=False, bargap=0.3, margin=dict(l=40,r=20,t=36,b=30))
        st.plotly_chart(fig_sig, use_container_width=True, config=_PCFG, key=_pc("sig_decomp"))
    fig=go.Figure()
    for i,(sname,path) in enumerate(sp.items()):
        opa = 1.0 if not sel_scen or sel_scen==sname else 0.2
        wid = 2.5 if not sel_scen or sel_scen==sname else 1
        hover_lbl = path.get("label","") if ho else sname
        fig.add_trace(go.Scatter(x=["Today"]+path["dates"],y=[spot]+path["prices"],
            name=sname, line=dict(color=SCEN_COLORS[i%5],width=wid), opacity=opa,
            hovertemplate=f"<b>{sname}</b><br>${{y:.4f}}<br><i>{hover_lbl}</i><extra></extra>"))
    fmt="$.4f" if ho else "$.2f"
    fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=340,
        title=dict(text="14-Day Scenario Simulation Paths (Dynamic Drift + VIX-Scaled Vol)",
                   font=dict(size=11,color="#1b2a3b")),
        yaxis=dict(tickformat=fmt),hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("scen"))
    if ho and sp:
        st.markdown("**Scenario Parameters** — effective drift and volatility after signal adjustments")
        rows_html = ""
        th = "padding:7px 14px;background:#eaeff6;color:#4e6880;font-family:'JetBrains Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid #c4d0de;text-align:center;"
        td = "padding:6px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;border-bottom:1px solid #d8e2ee;text-align:center;"
        for i,(sname,path) in enumerate(sp.items()):
            drift = path.get("total_drift", 0)
            vol   = path.get("vol_ann", 0)
            lbl   = path.get("label","")
            bg    = "background:#e2edf8;" if sel_scen==sname else ""
            dc    = "color:#1a7a45;" if drift>=0 else "color:#b82828;"
            rows_html += f"""<tr>
              <td style="{td}{bg}color:{SCEN_COLORS[i%5]};font-weight:700">{sname}</td>
              <td style="{td}{bg}{dc}">{drift:+.1f}%</td>
              <td style="{td}{bg}color:#5438a0;">{vol:.1f}%</td>
              <td style="{td}{bg}color:#4e6880;font-size:9px;text-align:left">{lbl}</td>
            </tr>"""
        st.markdown(
            f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #c4d0de;margin-bottom:16px">'
            f'<table style="width:100%;border-collapse:collapse;background:#ffffff">'
            f'<thead><tr>'
            f'<th style="{th}text-align:left">Scenario</th>'
            f'<th style="{th}">Drift (ann.)</th>'
            f'<th style="{th}">Vol (ann.)</th>'
            f'<th style="{th}text-align:left">Driver Logic</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        names_s=[s for s in sp]; finals=[sp[s]["final"] for s in names_s]
        cols_s=[SCEN_COLORS[i%5] if (not sel_scen or sel_scen==n) else "rgba(122,146,168,.25)" for i,n in enumerate(names_s)]
        fig=go.Figure(go.Bar(x=names_s,y=finals,marker_color=cols_s,
            text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in finals],textposition="outside"))
        fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=240,
            title=dict(text="Scenario Final Prices (Day 14)",font=dict(size=10,color="#1b2a3b")),
            yaxis=dict(tickformat="$.4f" if ho else "$.2f"),showlegend=False,bargap=0.2)
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("scen_final"))
    with c2:
        cib=result.get("ci_bands",{})
        w95=[round((cib.get(h,{}).get("ci95",[0,0])[1]-cib.get(h,{}).get("ci95",[0,0])[0]),4) for h in HORIZONS]
        w80=[round((cib.get(h,{}).get("ci80",[0,0])[1]-cib.get(h,{}).get("ci80",[0,0])[0]),4) for h in HORIZONS]
        mids=[cib.get(h,{}).get("mid",0) for h in HORIZONS]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=HORIZONS,y=w95,name="95% CI Width",marker_color="rgba(26,122,69,.65)"))
        fig.add_trace(go.Bar(x=HORIZONS,y=w80,name="80% CI Width",marker_color="rgba(23,88,176,.5)"))
        fig.add_trace(go.Scatter(x=HORIZONS,y=mids,name="Midpoint",mode="lines+markers",
            line=dict(color="#987010",width=1.5,dash="dot"),marker=dict(size=5)))
        fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=240,
            title=dict(text="Forecast Uncertainty by Horizon (CI Width)",font=dict(size=10,color="#1b2a3b")),
            yaxis=dict(title="Width ($)",tickformat="$.4f" if ho else "$.2f"),
            barmode="overlay",bargap=0.2,legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("ci_width"))


# ── ⑦ REGIONAL MAP ───────────────────────────────────────────────────────────
def render_regional(result, agent, sel_reg):
    section("11","REGIONAL PRICE MAP","United States · Brazil — Green = below avg, Red = above avg")
    rp    = result.get("regional_prices",[])
    br_rp = result.get("brazil_regional_prices",[])
    ho    = agent=="ho"
    if not rp: return

    def _make_map(data, scope, title, proj, center_lat=None, center_lon=None, scale=None):
        avg_p = sum(r["price"] for r in data) / len(data)
        fig   = go.Figure()
        for r in data:
            sel  = sel_reg == r["region"]
            clr  = "#1758b0" if sel else ("#b82828" if r["price"] > avg_p else "#1a7a45")
            delt = (r["price"] - avg_p) / avg_p * 100
            fig.add_trace(go.Scattergeo(
                lat=[r["lat"]], lon=[r["lon"]],
                mode="markers+text",
                marker=dict(
                    size=24 if sel else 17,
                    color=clr,
                    opacity=1.0 if not sel_reg or sel else 0.3,
                    line=dict(width=2 if sel else 0.5,
                              color="#fff" if sel else "rgba(0,0,0,.15)")),
                text=[r["state"]],
                textfont=dict(color="#fff", size=8),
                textposition="middle center",
                customdata=[[r["region"], r["price"], round(delt,1), r["factor"]]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Price: $%{customdata[1]:.4f}/gal<br>"
                    "vs avg: %{customdata[2]:+.1f}%<br>"
                    "%{customdata[3]}<extra></extra>"),
                name=r["region"], showlegend=False))
        geo_cfg = dict(
            bgcolor="#f5f8fc", landcolor="#eaeff6",
            coastlinecolor="#c4d0de", showlakes=False,
            showrivers=False, framecolor="#c4d0de",
            showocean=True, oceancolor="#ffffff",
            showcountries=True, countrycolor="#c4d0de")
        if scope:
            geo_cfg["scope"] = scope
        if proj:
            geo_cfg["projection_type"] = proj
        if center_lat is not None:
            geo_cfg["center"] = dict(lat=center_lat, lon=center_lon)
        if scale is not None:
            geo_cfg["projection"] = dict(scale=scale)
        fig.update_geos(**geo_cfg)
        fig.update_layout(
            template=PT, paper_bgcolor="#f5f8fc", height=380,
            title=dict(text=title, font=dict(size=11, color="#1b2a3b")),
            margin=dict(l=0,r=0,t=36,b=0))
        return fig

    tab_us, tab_br = st.tabs(["United States", "Brazil"])

    with tab_us:
        us_rp = [r for r in rp if r.get("country","US")=="US"]
        if us_rp:
            fig_us = _make_map(us_rp, scope="usa",
                               title="US Regional Heating Oil Prices ($/gal)", proj="albers usa")
            st.plotly_chart(fig_us, use_container_width=True, config=_PCFG, key=_pc("map_us"))
            us_prices_all = [r["price"] for r in us_rp]
            st.caption(f"US avg: **${float(np.mean(us_prices_all)):.4f}/gal** · "
                       f"{len(us_rp)} regions · Green = below avg, Red = above avg")

    with tab_br:
        if br_rp:
            fig_br = _make_map(br_rp, scope="south america",
                               title="Brazil Regional Heating Oil Prices ($/gal)", proj="mercator")
            st.plotly_chart(fig_br, use_container_width=True, config=_PCFG, key=_pc("map_br"))
            us_rp_all = [r for r in rp if r.get("country","US")=="US"]
            us_prices_all = [r["price"] for r in us_rp_all] if us_rp_all else [0]
            br_prices_all = [r["price"] for r in br_rp]
            fig_cmp = go.Figure()
            all_regions = (
                [dict(r, label=r["region"]) for r in us_rp_all] +
                [dict(r, label=r["region"]) for r in br_rp]
            )
            all_regions.sort(key=lambda x: x["price"])
            bar_colors_cmp = ["#1758b0" if r.get("country","US")=="US" else "#b87010"
                              for r in all_regions]
            fig_cmp.add_trace(go.Bar(
                x=[r["label"] for r in all_regions],
                y=[r["price"] for r in all_regions],
                marker_color=bar_colors_cmp,
                text=[f"${r['price']:.4f}" for r in all_regions],
                textposition="outside",
                hovertemplate="%{x}: $%{y:.4f}/gal<extra></extra>"))
            fig_cmp.add_hline(y=float(np.mean(us_prices_all)),
                line=dict(color="#1758b0", width=1, dash="dot"),
                annotation_text=f"US avg ${np.mean(us_prices_all):.4f}",
                annotation_font=dict(color="#1758b0", size=8))
            fig_cmp.add_hline(y=float(np.mean(br_prices_all)),
                line=dict(color="#b87010", width=1, dash="dot"),
                annotation_text=f"BR avg ${np.mean(br_prices_all):.4f}",
                annotation_font=dict(color="#b87010", size=8))
            fig_cmp.update_layout(
                template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc",
                height=280,
                title=dict(text="All Regions Ranked by Price — Blue = US, Orange = Brazil",
                           font=dict(size=10, color="#1b2a3b")),
                yaxis=dict(title="$/gal", tickformat="$.4f"),
                showlegend=False, bargap=0.15,
                margin=dict(l=40, r=10, t=36, b=60))
            st.plotly_chart(fig_cmp, use_container_width=True, config=_PCFG, key=_pc("region_cmp"))
            us_avg_v  = float(np.mean(us_prices_all))
            br_avg_v  = float(np.mean(br_prices_all))
            delta_pct = (br_avg_v - us_avg_v) / us_avg_v * 100
            st.caption(
                f"US avg: **${us_avg_v:.4f}/gal** · Brazil avg: **${br_avg_v:.4f}/gal** · "
                f"Brazil is **{delta_pct:+.1f}%** vs US average · "
                f"Brazil prices anchored to Petrobras refinery gate + state ICMS taxes")


# ── ⑧ EIA INVENTORY DEEP DIVE ────────────────────────────────────────────────
def render_eia_deep_dive(result):
    section("07","EIA INVENTORY DEEP DIVE","Seasonal band · WoW momentum · 5-year range")
    eia  = result.get("eia_data",{})
    hist = eia.get("history",[])
    c1,c2,c3 = st.columns(3)
    s   = eia.get("stocks_mbbl")
    wow = eia.get("wow_change")
    c1.metric("Latest Stocks", f"{s:,.0f} Mbbl" if s else "N/A")
    c2.metric("Week-on-Week",  f"{wow:+,.0f} Mbbl" if wow else "N/A",
              delta=f"{wow:+,.0f}" if wow else None)
    weeks = eia.get("weeks",[])
    if weeks:
        avg4 = sum(w["value"] for w in weeks[:4])/min(4,len(weeks))
        c3.metric("4-Week Avg", f"{avg4:,.0f} Mbbl")
    else:
        c3.metric("4-Week Avg","N/A")
    if not hist:
        key_missing = eia.get("key_missing", False)
        if key_missing:
            st.warning(
                "**EIA API key not found.** "
                "On Streamlit Cloud: go to **Manage app → Settings → Secrets** and add:\n\n"
                "```toml\nEIA_API_KEY = \"your_key_here\"\n```\n\n"
                "Locally: make sure your `.env` file contains `EIA_API_KEY=your_key_here` "
                "and the file is in the same folder as `ho_agent.py`."
            )
        else:
            st.warning(
                "**EIA inventory data unavailable.** The API key was found but all three sources "
                "failed (EIA v2 → EIA v1 → FRED). Check the **Run Log** below for per-source "
                "error details. Common causes: EIA API outage, key expired, or network restriction. "
                "The rest of the dashboard is unaffected."
            )
        return
    df_full = pd.DataFrame(hist)
    df_full["period"] = pd.to_datetime(df_full["period"])
    df_full = df_full.sort_values("period")
    st.markdown("**Inventory Levels & Week-over-Week Momentum** — last 16 weeks")
    c_a, c_b = st.columns([3, 2])
    with c_a:
        df_show = df_full.tail(52)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_show["period"], y=df_show["value"],
            mode="lines+markers", line=dict(color="#1758b0", width=2),
            marker=dict(size=4), fill="tozeroy", fillcolor="rgba(23,88,176,.07)",
            hovertemplate="Week %{x}: %{y:,.0f} Mbbl<extra></extra>", name="Distillate Stocks"))
        if len(df_show) >= 4:
            avg_v = df_show["value"].mean()
            fig.add_hline(y=avg_v, line=dict(color="#987010", width=1, dash="dash"),
                annotation_text=f"1yr Avg {avg_v:,.0f}", annotation_font=dict(color="#987010", size=9))
        fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
            title=dict(text="52-Week Inventory History (Mbbl)", font=dict(size=10, color="#1b2a3b")),
            xaxis=dict(title=""), yaxis=dict(title="Mbbl"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("eia"))
    with c_b:
        df_wow = df_full.tail(17)
        if len(df_wow) >= 2:
            wow_vals  = [round(df_wow["value"].iloc[i] - df_wow["value"].iloc[i-1], 0)
                         for i in range(1, len(df_wow))]
            wow_dates = [df_wow["period"].iloc[i] for i in range(1, len(df_wow))]
            wow_colors= ["#1a7a45" if w >= 0 else "#b82828" for w in wow_vals]
            fig = go.Figure(go.Bar(
                x=wow_dates, y=wow_vals, marker_color=wow_colors,
                text=[f"{int(w):+,}" for w in wow_vals], textposition="outside",
                hovertemplate="%{x}: %{y:+,.0f} Mbbl<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#4e6880", width=1))
            fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
                title=dict(text="WoW Change — Last 16 Weeks", font=dict(size=10, color="#1b2a3b")),
                xaxis=dict(tickangle=-45), yaxis=dict(title="Mbbl"), showlegend=False, bargap=0.15)
            st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("eia_wow"))
    sb = eia.get("seasonal_bands", [])
    cy = eia.get("current_year_data", [])
    if sb:
        st.markdown("**Seasonal Band (5-year min/max/avg) vs Current Year**")
        df_sb = pd.DataFrame(sb)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=pd.concat([df_sb["week"], df_sb["week"][::-1]]),
            y=pd.concat([df_sb["max"], df_sb["min"][::-1]]),
            fill="toself", fillcolor="rgba(23,88,176,.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=True, name="5-Yr Range"))
        fig_s.add_trace(go.Scatter(
            x=df_sb["week"], y=df_sb["avg"],
            mode="lines", line=dict(color="#1758b0", width=1.5, dash="dot"),
            name="5-Yr Avg"))
        if cy:
            df_cy = pd.DataFrame(cy)
            fig_s.add_trace(go.Scatter(
                x=df_cy["week"], y=df_cy["value"],
                mode="lines+markers", line=dict(color="#b87010", width=2),
                marker=dict(size=4), name="Current Year"))
        fig_s.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
            title=dict(text="Seasonal Band vs Current Year (ISO Week)", font=dict(size=10, color="#1b2a3b")),
            xaxis=dict(title="ISO Week"), yaxis=dict(title="Mbbl"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_s, use_container_width=True, config=_PCFG, key=_pc("eia_seasonal"))


# ── ⑨ CRACK SPREAD ANALYTICS ─────────────────────────────────────────────────
def render_crack_spread(result, agent):
    section("06","CRACK SPREAD ANALYTICS","Time series · percentile bands · distribution · scatter")
    ho = agent=="ho"
    ch = result.get("crack_history",[])
    md = result.get("market_data",{})
    current_crack = md.get("crack_spread")
    if not ch:
        st.info("Crack spread history requires both HO and WTI price history.")
        return
    df_crack = pd.DataFrame(ch)
    df_crack["date"] = pd.to_datetime(df_crack["date"])
    c1, c2, c3 = st.columns(3)
    crack_vals = df_crack["crack"].dropna().tolist()
    if crack_vals:
        c1.metric("Current Crack", f"${current_crack:.2f}/bbl" if current_crack else "N/A")
        c2.metric("1Y Avg Crack",  f"${float(np.mean(crack_vals)):.2f}/bbl")
        pct_rank = round(sum(v <= (current_crack or 0) for v in crack_vals) / len(crack_vals) * 100, 1)
        c3.metric("Percentile Rank", f"{pct_rank:.0f}th",
                  help="Where today's crack sits vs the past year")
    ca, cb = st.columns(2)
    with ca:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_crack["date"], y=df_crack["crack"],
            mode="lines", line=dict(color="#1a7a45", width=1.5),
            fill="tozeroy", fillcolor="rgba(26,122,69,.07)",
            hovertemplate="%{x|%Y-%m-%d}: $%{y:.2f}<extra></extra>", name="Crack Spread"))
        if crack_vals:
            p25 = float(np.percentile(crack_vals, 25))
            p75 = float(np.percentile(crack_vals, 75))
            fig1.add_hline(y=p75, line=dict(color="#987010", width=1, dash="dot"),
                annotation_text=f"75th ${p75:.2f}", annotation_font=dict(color="#987010", size=9))
            fig1.add_hline(y=p25, line=dict(color="#5438a0", width=1, dash="dot"),
                annotation_text=f"25th ${p25:.2f}", annotation_font=dict(color="#5438a0", size=9))
        fig1.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
            title=dict(text="Crack Spread History — HO 3:2:1 ($/bbl)", font=dict(size=10, color="#1b2a3b")),
            xaxis=dict(title=""), yaxis=dict(title="$/bbl"), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True, config=_PCFG, key=_pc("crack_ts"))
    with cb:
        if crack_vals:
            fig2 = go.Figure(go.Histogram(
                x=crack_vals, nbinsx=20,
                marker_color="rgba(26,122,69,.6)",
                hovertemplate="$%{x:.2f}: %{y} obs<extra></extra>"))
            if current_crack:
                fig2.add_vline(x=current_crack,
                    line=dict(color="#b82828", width=2, dash="dash"),
                    annotation_text=f"Now ${current_crack:.2f}",
                    annotation_font=dict(color="#b82828", size=9))
            fig2.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
                title=dict(text="Crack Spread Distribution", font=dict(size=10, color="#1b2a3b")),
                xaxis=dict(title="$/bbl"), yaxis=dict(title="Observations"), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config=_PCFG, key=_pc("crack_dist"))
    if ho and len(ch) >= 10:
        ho_prices  = [r.get("ho") for r in ch if r.get("ho") and r.get("crack")]
        crack_x    = [r.get("crack") for r in ch if r.get("ho") and r.get("crack")]
        if len(ho_prices) >= 5:
            m, b = np.polyfit(crack_x, ho_prices, 1)
            x_line = [min(crack_x), max(crack_x)]
            y_line = [m * x + b for x in x_line]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=crack_x, y=ho_prices, mode="markers",
                marker=dict(color="#5438a0", size=5, opacity=0.6),
                name="Historical",
                hovertemplate="Crack $%{x:.2f} → HO $%{y:.4f}<extra></extra>"))
            fig3.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="#987010", width=1.5, dash="dot"),
                name=f"Regression (slope={m:.4f})", hoverinfo="skip"))
            if current_crack and current_crack in crack_x:
                cur_ho = md.get("HO", 0)
                fig3.add_trace(go.Scatter(
                    x=[current_crack], y=[cur_ho], mode="markers",
                    marker=dict(color="#b82828", size=14, symbol="star",
                                line=dict(width=2, color="#ffffff")),
                    name="Today",
                    hovertemplate=f"Today: crack ${current_crack:.2f} → HO ${cur_ho:.4f}<extra></extra>"))
            fig3.update_layout(
                template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
                title=dict(text="Crack Spread vs HO Price — Historical Relationship",
                           font=dict(size=10, color="#1b2a3b")),
                xaxis=dict(title="Crack Spread ($/bbl)"),
                yaxis=dict(title="HO Price ($/gal)", tickformat="$.4f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig3, use_container_width=True, config=_PCFG, key=_pc("crack_scatter"))
        else:
            st.info("Insufficient data for crack vs HO scatter plot.")


# ── ⑩ SEASONAL PATTERN ANALYSIS ──────────────────────────────────────────────
def render_seasonal_pattern(result, agent):
    if agent != "ho":
        return
    section("08","SEASONAL PATTERN ANALYSIS","Monthly average vs current — cycle positioning")
    history = result.get("history", [])
    if len(history) < 90:
        st.info("Seasonal pattern needs at least 90 days of history.")
        return
    from collections import defaultdict
    import calendar
    month_buckets = defaultdict(list)
    for row in history:
        try:
            dt = datetime.datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
            month_buckets[dt.month].append(float(row["price"]))
        except Exception:
            pass
    months     = list(range(1, 13))
    month_abbr = [calendar.month_abbr[m] for m in months]
    avgs       = [round(float(np.mean(month_buckets[m])), 4) if month_buckets[m] else None for m in months]
    current_m  = datetime.date.today().month
    cur_ho     = result.get("market_data", {}).get("HO", result.get("ho_price", 0))
    overall_avg   = float(np.mean([p for v in month_buckets.values() for p in v])) if month_buckets else cur_ho
    current_m_avg = avgs[current_m - 1] or cur_ho
    delta_vs_seasonal = round((cur_ho - current_m_avg) / current_m_avg * 100, 2) if current_m_avg else 0
    d1, d2 = st.columns(2)
    d1.metric("Current Month Avg (hist.)", f"${current_m_avg:.4f}/gal",
              help=f"Historical avg for {calendar.month_name[current_m]}")
    d2.metric("Live vs Seasonal Avg", f"{delta_vs_seasonal:+.2f}%",
              delta=f"{delta_vs_seasonal:+.2f}%",
              help="Positive = currently trading above seasonal norm")
    avgs_plot  = [a if a is not None else 0 for a in avgs]
    bar_colors = ["#b87010" if i+1==current_m else "#1758b0" for i in range(12)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=month_abbr, y=avgs_plot, marker_color=bar_colors,
        text=[f"${a:.4f}" if a else "N/A" for a in avgs],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.4f}/gal<extra></extra>", name="Monthly Avg"))
    fig.add_hline(y=overall_avg,
        line=dict(color="#987010", width=1.5, dash="dash"),
        annotation_text=f"Overall avg ${overall_avg:.4f}",
        annotation_font=dict(color="#987010", size=9))
    if cur_ho:
        fig.add_hline(y=cur_ho,
            line=dict(color="#b82828", width=1.5, dash="dot"),
            annotation_text=f"Live ${cur_ho:.4f}",
            annotation_font=dict(color="#b82828", size=9))
    fig.update_layout(
        template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=300,
        title=dict(text="Seasonal Price Pattern — Monthly Historical Average",
                   font=dict(size=11, color="#1b2a3b")),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Avg Price ($/gal)", tickformat="$.4f"),
        showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("seasonal"))
    st.caption(
        f"Highlighted bar = current month ({calendar.month_name[current_m]}) · "
        f"Avg: **${current_m_avg:.4f}** · "
        f"Live price: **${cur_ho:.4f}** · vs seasonal avg: **{delta_vs_seasonal:+.2f}%**")


# ── ⑪ VALUE AT RISK & EXPECTED SHORTFALL ─────────────────────────────────────
def render_var_es(result, agent):
    section("09","VALUE AT RISK (VaR) & EXPECTED SHORTFALL","Monte Carlo — 10,000 simulations")
    ho       = agent=="ho"
    var_data = result.get("var_es",{})
    if not var_data:
        return
    horizons_shown = ["1M","3M"]
    cols = st.columns(len(horizons_shown)*2)
    col_i=0
    for h in horizons_shown:
        d = var_data.get(h,{})
        if not d: continue
        conf = int(d.get("confidence",95))
        cols[col_i].metric(f"VaR {h} ({conf}%)",  f"${d['var']:.4f}/gal" if ho else f"${d['var']:.2f}/bbl",
            help="Max loss at this confidence level over the horizon")
        cols[col_i+1].metric(f"ES {h} ({conf}%)", f"${d['es']:.4f}/gal" if ho else f"${d['es']:.2f}/bbl",
            help="Average loss beyond VaR (Conditional VaR / CVaR)")
        col_i+=2
    c1,c2 = st.columns(2)
    for ci,h in enumerate(horizons_shown):
        d = var_data.get(h,{})
        if not d: continue
        pnl   = d.get("pnl_distribution",[])
        plbls = d.get("percentile_labels",[])
        if not pnl: continue
        bar_colors=["#b82828" if v<0 else "#1a7a45" for v in pnl]
        fig=go.Figure(go.Bar(x=plbls,y=pnl,marker_color=bar_colors,
            text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in pnl],textposition="outside",
            hovertemplate="%{x}: $%{y:.4f}<extra></extra>"))
        fig.add_hline(y=0,line=dict(color="#4e6880",width=1))
        fig.update_layout(template=PT,paper_bgcolor="#f5f8fc",plot_bgcolor="#f5f8fc",height=240,
            title=dict(text=f"P&L Percentile Distribution — {h}",font=dict(size=10,color="#1b2a3b")),
            yaxis=dict(title="P&L ($/gal)" if ho else "P&L ($/bbl)"),
            showlegend=False,bargap=0.15)
        (c1 if ci==0 else c2).plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc(f"var_{h}"))


# ── ⑤B FORWARD CURVE & CALENDAR SPREADS ─────────────────────────────────────
def render_curve_spreads(result):
    section("05B", "FORWARD CURVE & INTER-MONTH SPREADS",
            f"Source: {result.get('curve_source', 'n/a')} · spread = near − far (positive = backwardation)")
    contracts = result.get("ho_contracts", [])
    if not contracts:
        st.info("Forward curve unavailable.")
        return
    if all(c.get("price_source") == "estimated" for c in contracts):
        st.warning("Contract quotes unavailable — curve shown flat at the front price; spreads hidden.")
    lbl = [c["label"] for c in contracts]
    px  = [c["fwd_price"] for c in contracts]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=lbl, y=px, mode="lines+markers", line=dict(color="#b87010", width=2),
            marker=dict(size=7, color=["#b87010" if c.get("price_source") != "estimated" else "#c4d0de" for c in contracts]),
            hovertemplate="%{x}: $%{y:.4f}<extra></extra>"))
        fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=300,
            title=dict(text="HO Futures Curve ($/gal)", font=dict(size=11, color="#1b2a3b")),
            yaxis=dict(tickformat="$.4f"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("curve"))
    spreads = ve.calendar_spreads(contracts)
    with c2:
        if spreads:
            vals = [x["spread"]*100 for x in spreads]
            fig = go.Figure(go.Bar(x=[x["pair"] for x in spreads], y=vals,
                marker_color=["#1a7a45" if v >= 0 else "#b82828" for v in vals],
                text=[f"{v:+.2f}¢" for v in vals], textposition="outside",
                hovertemplate="%{x}: %{y:+.2f}¢/gal<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#4e6880", width=1))
            fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=300,
                title=dict(text="Consecutive-Month Spreads (¢/gal)", font=dict(size=11, color="#1b2a3b")),
                yaxis=dict(title="¢/gal"), showlegend=False, bargap=0.25)
            st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("spreads"))
    if spreads:
        pairs = [f"{a['label']} / {b['label']}" for a, b in zip(contracts[:-1], contracts[1:])]
        pick = st.selectbox("Spread history", pairs, index=0, key="spread_pick")
        i = pairs.index(pick)
        h = ve.spread_history(contracts[i], contracts[i+1])
        if not h.empty:
            fig = go.Figure(go.Scatter(x=h["date"], y=h["spread"]*100, mode="lines",
                line=dict(color="#5438a0", width=2), fill="tozeroy", fillcolor="rgba(84,56,160,.07)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}¢<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#4e6880", width=1))
            fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
                title=dict(text=f"{pick} spread history (¢/gal)", font=dict(size=11, color="#1b2a3b")),
                yaxis=dict(title="¢/gal"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("spread_hist"))
            st.caption(f"Current {h['spread'].iloc[-1]*100:+.2f}¢ · 1Y range {h['spread'].min()*100:+.2f}¢ "
                       f"to {h['spread'].max()*100:+.2f}¢ · percentile "
                       f"{(h['spread'] <= h['spread'].iloc[-1]).mean()*100:.0f}th")
        else:
            st.caption("Not enough overlapping history for this pair.")


# ── ⑥B CHICAGO BASIS & BRAZIL PPI ───────────────────────────────────────────
def _input_form(fields, key, allow_csv_field=None):
    """Admin data entry for manual market inputs (persisted, versioned in the store)."""
    with st.form(key, clear_on_submit=False):
        d = st.date_input("Effective date", value=datetime.date.today(), key=f"{key}_d")
        vals = {}
        for f in fields:
            label, unit = fs.INPUT_FIELDS[f]
            vals[f] = st.number_input(f"{label} ({unit})", value=None, format="%.4f", key=f"{key}_{f}")
        note = st.text_input("Source / note", max_chars=200, key=f"{key}_n")
        up = st.file_uploader(f"…or CSV history (columns: date, value)", type=["csv"], key=f"{key}_csv") \
            if allow_csv_field else None
        if st.form_submit_button("Save"):
            rows = [{"date": d, "field": f, "value": v, "note": sanitize_str(note)}
                    for f, v in vals.items() if v is not None]
            if up is not None:
                try:
                    df = pd.read_csv(up)
                    cols = {c.lower().strip(): c for c in df.columns}
                    rows += [{"date": r[cols["date"]], "field": allow_csv_field, "value": r[cols["value"]],
                              "note": "csv upload"} for _, r in df.iterrows()]
                except Exception as e:
                    st.error(f"CSV not read: {sanitize_str(str(e))}")
            n = fs.save_market_inputs(rows, user=st.session_state.get("auth_user", "")) if rows else 0
            _clear_store_caches()
            (st.success if n else st.warning)(f"Saved {n} value(s)." if n else "Nothing saved.")


def render_basis_ppi(result):
    section("06B", "CHICAGO BASIS & BRAZIL PPI",
            "Chicago ULSD vs NYMEX HO · Petrobras vs import parity (PPI)")
    inputs = _load_inputs()
    ho = result["ho_price"]
    tab_chi, tab_br = st.tabs(["Chicago basis", "Brazil PPI vs Petrobras"])
    with tab_chi:
        basis, bdate = fs.latest_input(inputs, "chicago_basis_cpg")
        if basis is None:
            st.info("No Chicago basis stored yet. Chicago ULSD basis comes from OPIS/Argus/broker quotes — "
                    "an admin can enter it below (history is kept for the track record).")
        else:
            c = st.columns(3)
            c[0].metric("Chicago basis", f"{basis:+.2f}¢/gal", help=f"As of {bdate}")
            c[1].metric("NYMEX HO front", f"${ho:.4f}")
            c[2].metric("Chicago implied price", f"${ho + basis/100:.4f}/gal")
            hist = inputs[inputs["field"] == "chicago_basis_cpg"]
            if len(hist) > 1:
                fig = go.Figure(go.Scatter(x=pd.to_datetime(hist["date"]), y=hist["value"], mode="lines+markers",
                    line=dict(color="#1758b0", width=2, shape="hv"), hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}¢<extra></extra>"))
                fig.add_hline(y=0, line=dict(color="#4e6880", width=1))
                fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=260,
                    title=dict(text="Chicago ULSD Basis History (¢/gal vs NYMEX HO)", font=dict(size=11, color="#1b2a3b")),
                    yaxis=dict(title="¢/gal"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("chi_basis"))
        if _is_admin():
            with st.expander("Admin — enter Chicago basis", expanded=False):
                _input_form(["chicago_basis_cpg"], "form_chi", allow_csv_field="chicago_basis_cpg")
    with tab_br:
        usdbrl = result.get("market_data", {}).get("USDBRL")
        params = {k: _param(inputs, k) for k in ("PPI_GULF_BASIS_USD_GAL", "PPI_FREIGHT_USD_GAL",
                                                 "PPI_PORT_COSTS_USD_GAL", "PPI_INTERNAL_BRL_L")}
        if not usdbrl:
            st.warning("USD/BRL unavailable — PPI cannot be computed this run.")
        else:
            ppi = ve.ppi_brl_per_liter(ho, usdbrl, params)
            pb, pdate = fs.latest_input(inputs, "petrobras_diesel_brl_l")
            c = st.columns(4)
            c[0].metric("PPI (import parity)", f"R$ {ppi:.4f}/L")
            c[1].metric("USD/BRL", f"{usdbrl:.4f}")
            c[2].metric("Petrobras diesel A", f"R$ {pb:.4f}/L" if pb else "—", help=f"As of {pdate}" if pdate else "Not entered")
            if pb:
                gap = (pb / ppi - 1) * 100
                c[3].metric("Petrobras vs PPI", f"{gap:+.1f}%",
                            help="Negative = Petrobras sells below import parity (import window closed)")
            if all(v == 0 for k, v in params.items()):
                st.warning("PPI logistics components are all zero — the figure is the NYMEX FOB parity only. "
                           "Enter USGC basis, freight, port and internal costs below (or in Secrets).")
            _html_table(["Component", "Value"], [
                ["NYMEX HO front", f"${ho:.4f}/gal"],
                ["+ USGC basis", f"${params['PPI_GULF_BASIS_USD_GAL']:.4f}/gal"],
                ["+ Ocean freight", f"${params['PPI_FREIGHT_USD_GAL']:.4f}/gal"],
                ["+ Port / insurance / losses", f"${params['PPI_PORT_COSTS_USD_GAL']:.4f}/gal"],
                ["× USD/BRL ÷ L/gal", f"{usdbrl:.4f} ÷ {cfg.LITERS_PER_GALLON:.4f}"],
                ["+ Internal logistics", f"R$ {params['PPI_INTERNAL_BRL_L']:.4f}/L"],
                [("= PPI", "font-weight:700;"), (f"R$ {ppi:.4f}/L", "font-weight:700;color:#b87010;")]])
            hh = pd.DataFrame(result.get("history", [])); bh = pd.DataFrame(result.get("usdbrl_history", []))
            if not hh.empty and not bh.empty:
                m = hh.merge(bh, on="date", suffixes=("_ho", "_brl"))
                m["ppi"] = [ve.ppi_brl_per_liter(a, b, params) for a, b in zip(m["price_ho"], m["price_brl"])]
                m["date"] = pd.to_datetime(m["date"])
                fig = go.Figure(go.Scatter(x=m["date"], y=m["ppi"], mode="lines", name="PPI (current cost params)",
                    line=dict(color="#b87010", width=2), hovertemplate="%{x|%Y-%m-%d}: R$ %{y:.4f}<extra>PPI</extra>"))
                ph = inputs[inputs["field"] == "petrobras_diesel_brl_l"] if not inputs.empty else pd.DataFrame()
                if not ph.empty:
                    fig.add_trace(go.Scatter(x=pd.to_datetime(ph["date"]), y=ph["value"], mode="lines+markers",
                        name="Petrobras", line=dict(color="#1a7a45", width=2, shape="hv"),
                        hovertemplate="%{x|%Y-%m-%d}: R$ %{y:.4f}<extra>Petrobras</extra>"))
                fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=300,
                    title=dict(text="Diesel PPI vs Petrobras price (R$/L)", font=dict(size=11, color="#1b2a3b")),
                    yaxis=dict(title="R$/L"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("ppi"))
        if _is_admin():
            with st.expander("Admin — Petrobras price & PPI cost components", expanded=False):
                _input_form(["petrobras_diesel_brl_l", "PPI_GULF_BASIS_USD_GAL", "PPI_FREIGHT_USD_GAL",
                             "PPI_PORT_COSTS_USD_GAL", "PPI_INTERNAL_BRL_L"], "form_ppi",
                            allow_csv_field="petrobras_diesel_brl_l")


# ── ⑫ FORECAST TRACK RECORD ──────────────────────────────────────────────────
def render_track_record(result, vctx):
    section("12", "FORECAST TRACK RECORD",
            "Every day's first forecast is stored immutably and scored against what actually happened")
    ss = _store_status()
    snap = st.session_state.get("_snap_res")
    if not ss.get("persistent"):
        st.warning(f"Store: {ss.get('backend')}. {ss.get('detail')}")
    elif not ss.get("ok"):
        st.error(f"Store unreachable: {ss.get('detail')}")
    else:
        st.caption(f"Store: **{ss.get('backend')}** · today's save: "
                   + (f"{snap.get('forecasts', 0)} forecast rows, {snap.get('curves', 0)} curve rows"
                      if snap and "error" not in snap else "already recorded / not run" if not snap
                      else f"error {snap.get('error')}"))
    if _is_admin() and st.button("Save today's snapshot now", key="snap_now"):
        res = _autosave_snapshot(result, vctx, force=True)
        st.success(f"Saved: {res}")
    fc = _load_forecasts()
    if fc.empty:
        st.info("No forecasts stored yet — the first snapshot is written on today's HO run.")
        return
    fh = _front_history_long()
    hist_now = pd.DataFrame(result.get("history", []))
    fh = pd.concat([fh, hist_now]).drop_duplicates("date", keep="last") if not fh.empty else hist_now
    ev = fs.evaluate_forecasts(fc, fh, _load_curves())
    sm = fs.track_record_summary(ev)
    m = st.columns(6)
    m[0].metric("Forecasts stored", f"{sm.get('total', 0):,}")
    m[1].metric("Evaluated", f"{sm.get('evaluated', 0):,}")
    m[2].metric("Pending", f"{sm.get('pending', 0):,}")
    m[3].metric("Hit rate 80% range", _fmt_pct(sm.get("cov80")), help="Share of outcomes inside the p10–p90 range. Well-calibrated ≈ 80%.")
    m[4].metric("Hit rate 90% range", _fmt_pct(sm.get("cov90")), help="p05–p95 range. Well-calibrated ≈ 90%.")
    m[5].metric("Median abs error", f"${sm['mae']:.4f}" if sm.get("mae") is not None else "—")
    k1, k2 = st.columns(2)
    kind = k1.radio("Forecast type", ["contract", "horizon"], horizontal=True, key="tr_kind",
                    format_func=lambda x: "Contract settlement (e.g. Nov range)" if x == "contract" else "Front-month horizon")
    sub = ev[ev["kind"] == kind]
    targets = sorted(sub["target"].unique(), key=lambda t: str(sub[sub["target"] == t]["target_date"].iloc[0]))
    if not targets:
        return
    tgt = k2.selectbox("Target", targets, key="tr_target")
    d = sub[sub["target"] == tgt].sort_values("forecast_date")
    fig = go.Figure()
    x = pd.to_datetime(d["forecast_date"])
    fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]), y=list(d["p95"])+list(d["p05"][::-1]), fill="toself",
        fillcolor="rgba(23,88,176,.08)", line=dict(width=0), name="90% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=list(x)+list(x[::-1]), y=list(d["p90"])+list(d["p10"][::-1]), fill="toself",
        fillcolor="rgba(23,88,176,.18)", line=dict(width=0), name="80% range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=d["p50"], mode="lines+markers", line=dict(color="#1758b0", width=2),
        marker=dict(size=4), name="Forecast median",
        hovertemplate="Forecast %{x|%Y-%m-%d}: median $%{y:.4f}<extra></extra>"))
    real = d.dropna(subset=["realized"])
    if not real.empty:
        fig.add_trace(go.Scatter(x=x.loc[real.index], y=real["realized"], mode="markers",
            marker=dict(size=9, symbol="star", color=["#1a7a45" if v == 1 else "#b82828" for v in real["in_80"]]),
            name="Realized (green = inside 80%)", hovertemplate="Realized $%{y:.4f}<extra></extra>"))
    fig.update_layout(template=PT, paper_bgcolor="#f5f8fc", plot_bgcolor="#f5f8fc", height=320,
        title=dict(text=f"{tgt}: forecast ranges by forecast date vs realized outcome "
                        f"(target {d['target_date'].iloc[-1]})", font=dict(size=11, color="#1b2a3b")),
        yaxis=dict(tickformat="$.4f"), xaxis=dict(title="Forecast date"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("track"))
    show = ev.sort_values(["forecast_date", "kind", "target_date"], ascending=[False, True, True]).head(60)
    rows = []
    for _, r in show.iterrows():
        hit = r["in_80"]
        rows.append([r["forecast_date"], r["kind"], r["target"], r["target_date"], f"${r['ref_price']:.4f}",
                     f"${r['p10']:.4f} – ${r['p90']:.4f}", f"${r['p50']:.4f}",
                     f"${r['realized']:.4f}" if pd.notna(r["realized"]) else "—",
                     ("✓" if hit == 1 else "✗" if hit == 0 else r["status"],
                      f"color:{'#1a7a45' if hit == 1 else '#b82828' if hit == 0 else '#7a92a8'};font-weight:700;")])
    with st.expander("Forecast log (latest 60)", expanded=False):
        _html_table(["Made", "Type", "Target", "Target date", "Ref price", "80% range", "Median",
                     "Realized", "In 80%?"], rows)
    st.download_button("Download full track record (CSV)", ev.to_csv(index=False).encode(),
                       file_name=f"ho_track_record_{datetime.date.today()}.csv", mime="text/csv", key="tr_dl")


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:16px 0 8px">
      <div style="font-size:18px;font-weight:800;color:#1b2a3b;font-family:'Syne',sans-serif">
        Energy Intelligence
      </div>
      <div style="font-size:9px;color:#4e6880;font-family:'JetBrains Mono',monospace;letter-spacing:1.2px;text-transform:uppercase">
        Commodity Probability Engine
      </div>
    </div>""", unsafe_allow_html=True)
    # ── Logged-in user + logout ───────────────────────────────────────────────
    auth_user = st.session_state.get("auth_user", "")
    if auth_user:
        st.sidebar.markdown(
            f"<div style='font-size:10px;color:#4e6880;font-family:\'JetBrains Mono\',monospace;"
            f"padding:4px 0 8px'>{auth_user}</div>",
            unsafe_allow_html=True
        )
        if st.sidebar.button("Sign out", use_container_width=True, key="sidebar_logout"):
            _auth_logout()
            st.rerun()
    st.sidebar.divider()
    sess = st.session_state.get("_session_id", id(st.session_state))
    run_oil = st.sidebar.button("Oil (WTI/Brent)", use_container_width=True)
    run_ho  = st.sidebar.button("HO (Heating Oil)", use_container_width=True)
    if run_oil:
        allowed,_,reset_in=limiter.check(sess)
        if not allowed:
            st.error(f"Rate limit reached. Try again in {reset_in}s.")
        else:
            with st.spinner("Fetching oil market data..."):
                try:
                    result,log=run_oil_agent()
                    st.session_state.update(result=result,agent="oil",log=log,
                        sel_horizon="1M",sel_bin=None,sel_scenario=None,sel_region=None,sel_driver=None)
                except Exception as e:
                    st.error(f"Error: {e}")
    if run_ho:
        allowed,_,reset_in=limiter.check(sess)
        if not allowed:
            st.error(f"Rate limit reached. Try again in {reset_in}s.")
        else:
            with st.spinner("Fetching heating oil data (1-year history)..."):
                try:
                    result,log=run_ho_agent()
                    st.session_state.update(result=result,agent="ho",log=log,
                        sel_horizon="1M",sel_bin=None,sel_scenario=None,sel_region=None,sel_driver=None)
                except Exception as e:
                    st.error(f"Error: {e}")
    # ── Refresh button — busts the 5-min cache on demand ─────────────────────
    if st.session_state.result is not None:
        if st.sidebar.button("Refresh Data", use_container_width=True,
                             help="Force-fetch latest prices and recompute probabilities"):
            allowed, _, reset_in = limiter.check(sess)
            if not allowed:
                st.error(f"Rate limit reached. Try again in {reset_in}s.")
            else:
                agent_now = st.session_state.agent
                run_oil_agent.clear()   # bust TTL cache
                run_ho_agent.clear()
                with st.spinner("Refreshing market data..."):
                    try:
                        if agent_now == "ho":
                            result, log = run_ho_agent()
                        else:
                            result, log = run_oil_agent()
                        st.session_state.update(result=result, log=log)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Refresh error: {e}")
    result=st.session_state.result
    if result:
        st.sidebar.divider()
        st.sidebar.markdown("**Cross-Filters**")
        st.sidebar.caption("Selections update all charts simultaneously")
        h = st.sidebar.radio("Horizon",HORIZONS,index=HORIZONS.index(st.session_state.sel_horizon),horizontal=True)
        try: h=validate_enum(sanitize_str(h),set(HORIZONS))
        except ValueError: h="1M"
        st.session_state.sel_horizon=h
        sp=result.get("scenario_paths",{})
        scen_opts=["(All)"]+list(sp.keys())
        sel_s=st.sidebar.selectbox("Scenario",scen_opts,index=0 if not st.session_state.sel_scenario else
            (scen_opts.index(st.session_state.sel_scenario) if st.session_state.sel_scenario in scen_opts else 0))
        st.session_state.sel_scenario=None if sel_s=="(All)" else sanitize_str(sel_s)
        rp=result.get("regional_prices",[])
        reg_opts=["(All)"]+[r["region"] for r in rp]
        sel_r=st.sidebar.selectbox("Region",reg_opts,index=0 if not st.session_state.sel_region else
            (reg_opts.index(st.session_state.sel_region) if st.session_state.sel_region in reg_opts else 0))
        st.session_state.sel_region=None if sel_r=="(All)" else sanitize_str(sel_r)
        _e,_labels,_t = display_prob_table(result, h)
        bins=["(All)"]+list(_labels)
        sel_b=st.sidebar.selectbox("Price Bin",bins,index=0 if not st.session_state.sel_bin else
            (bins.index(st.session_state.sel_bin) if st.session_state.sel_bin in bins else 0))
        st.session_state.sel_bin=None if sel_b=="(All)" else sanitize_str(sel_b)
        if st.sidebar.button("Clear all filters",use_container_width=True):
            st.session_state.update(sel_horizon="1M",sel_bin=None,sel_scenario=None,sel_region=None,sel_driver=None)
            st.rerun()
    if st.session_state.get("auth_is_admin"):
        render_admin_panel()
    if result and result.get("agent") == "ho":
        _ss = _store_status()
        st.sidebar.caption(("Track record store: " if _ss.get("persistent") else "Track record store (NOT persistent): ")
                           + f"{_ss.get('backend')} — {_ss.get('detail')}")
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="font-size:9px;color:#7a92a8;font-family:'JetBrains Mono',monospace;line-height:1.8">
    Contact:<br><a href="mailto:lsaggioro@potonmail.com" style="color:#1758b0;text-decoration:none">lsaggioro@potonmail.com</a>
    </div>""",unsafe_allow_html=True)


def render_dashboard():
    # ── 30-second live-price auto-refresh (non-blocking) ──────────────────────
    if _AUTOREFRESH_OK:
        _st_autorefresh(interval=30_000, limit=None, key="mkt_live_refresh")

    result  = st.session_state.result
    agent   = st.session_state.agent
    sel_h   = st.session_state.sel_horizon
    sel_bin = st.session_state.sel_bin
    sel_scen= st.session_state.sel_scenario
    sel_reg = st.session_state.sel_region
    if not result:
        st.markdown("""
        <div style="text-align:center;padding:80px 0">
          <div style="font-size:22px;font-weight:800;color:#1b2a3b;font-family:'Syne',sans-serif;margin-bottom:8px">
            Energy Intelligence Dashboard
          </div>
          <div style="font-size:11px;color:#4e6880;font-family:'JetBrains Mono',monospace;letter-spacing:.8px">
            DETERMINISTIC COMMODITY PROBABILITY ENGINE
          </div>
          <div style="margin-top:32px;font-size:13px;color:#7a92a8">
            Click <strong style="color:#b87010">Oil</strong> or <strong style="color:#1758b0">HO</strong> in the sidebar to begin
          </div>
        </div>""",unsafe_allow_html=True)
        return
    ho = agent == "ho"
    # Section order (v4.0):
    # 01 Snapshot · 02 Price History · 03 Prob Distribution · 03B Scenario Impact
    # 04 KO Prob · 04B Expiry Dist · 05 Volatility · 05B Curve & Spreads · 06 Crack
    # 06B Chicago basis & Brazil PPI · 07 EIA (flag) · 08 Seasonal · 09 VaR
    # 10 Scenario paths · 11 Regional Map · 12 Track Record
    vctx = None
    if ho:
        vctx = build_vol_context(result, result.get("run_dir", ""))
        _autosave_snapshot(result, vctx)
    render_snapshot(result, agent)
    render_price_history(result, agent)
    render_prob_dist(result, agent, sel_h, sel_bin)
    if ho:
        render_scenario_impact(result, sel_h)
        render_ko_table(result, agent, vctx)
        render_expiry_distribution_table(result, agent, vctx)
    render_volatility(result, vctx)
    if ho:
        render_curve_spreads(result)
        render_crack_spread(result, agent)
        render_basis_ppi(result)
        if cfg.get("EIA_ENABLED"):
            render_eia_deep_dive(result)
        render_seasonal_pattern(result, agent)
        render_var_es(result, agent)
    render_scenario(result, agent, sel_scen)
    render_regional(result, agent, sel_reg)
    if ho:
        render_track_record(result, vctx)
    section("--","MARKET SUMMARY")
    with st.expander("View full summary",expanded=False):
        st.code(result.get("summary",""),language=None)
    with st.expander("Run log",expanded=False):
        _vlog = (vctx or {}).get("log", []) if ho else []
        st.markdown('<div class="status-box">'+html.escape("\n".join((st.session_state.log or []) + _vlog))+"</div>",
            unsafe_allow_html=True)
    with st.expander("Security audit",expanded=False):
        st.code(security_audit_report(),language=None)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    if not render_auth_gate():
        return
    render_sidebar()
    render_dashboard()

if __name__=="__main__":
    main()
