"""
Energy Intelligence Dashboard — Streamlit Edition
HO + WTI/Oil. All emojis removed from UI labels.
Security: rate-limiting, input sanitization, env-var secrets.
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

# Load .env if present (local dev)
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=True)
except ImportError:
    pass

# ── INLINED SECURITY MODULE ───────────────────────────────────────────────────
# Inlined so no extra file is required on Streamlit Cloud.
import re, html, time, logging
from collections import defaultdict
from typing import Any, Collection

_RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", 900))
_RATE_LIMIT_MAX: int    = int(os.environ.get("RATE_LIMIT_MAX_ATTEMPTS", 5))
_MAX_PAYLOAD: int       = int(os.environ.get("MAX_PAYLOAD_BYTES", 65536))
_SAFE_PATTERN           = re.compile(r"[^a-zA-Z0-9 $.\-+%/(),]")
_MAX_STR_LEN            = 128

class _RateLimiter:
    def __init__(self, max_attempts=_RATE_LIMIT_MAX, window_secs=_RATE_LIMIT_WINDOW):
        self._max = max_attempts
        self._win = window_secs
        self._log: dict = defaultdict(list)

    def check(self, session_id: str):
        now = time.monotonic()
        history = self._log[session_id]
        history[:] = [t for t in history if t > now - self._win]
        if len(history) >= self._max:
            reset_in = int(self._win - (now - history[0])) + 1
            return False, 0, reset_in
        history.append(now)
        return True, self._max - len(history), 0

    def remaining(self, session_id: str) -> int:
        now = time.monotonic()
        history = [t for t in self._log.get(session_id, []) if t > now - self._win]
        return max(0, self._max - len(history))

def sanitize_str(value: Any, field: str = "input") -> str:
    if value is None:
        return ""
    cleaned = _SAFE_PATTERN.sub("", html.escape(str(value), quote=True)).strip()
    return cleaned[:_MAX_STR_LEN]

def validate_enum(value: Any, allowed: Collection, field: str = "field") -> Any:
    if value not in allowed:
        raise ValueError(f"Invalid {field}: {value!r}")
    return value

def security_audit_report() -> str:
    issues, passed = [], []
    eia = os.environ.get("EIA_API_KEY", "")
    if not eia:
        issues.append("EIA_API_KEY not set — EIA inventory fetch will fail")
    elif eia == "DEMO_KEY":
        issues.append("EIA_API_KEY is still 'DEMO_KEY' — set a real key")
    else:
        passed.append("EIA_API_KEY is set")
    passed.append(f"Rate limit: {_RATE_LIMIT_MAX} attempts / {_RATE_LIMIT_WINDOW}s window")
    passed.append(f"Payload limit: {_MAX_PAYLOAD} bytes")
    lines = ["=== SECURITY AUDIT REPORT ==="]
    if passed:
        lines += ["", "PASSED:"] + [f"  [OK] {p}" for p in passed]
    if issues:
        lines += ["", "WARNINGS:"] + [f"  [!!] {i}" for i in issues]
    if not issues:
        lines.append("\nAll checks passed.")
    return "\n".join(lines)

limiter = _RateLimiter()

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
.stApp{background:#07090f!important;}
.stPlotlyChart{background:#07090f!important;}
.stPlotlyChart>div{background:#07090f!important;}
div[data-testid="stPlotlyChart"]{background:#07090f!important;}
div[data-testid="block-container"]{background:#07090f!important;}
div[data-testid="stVerticalBlock"]{background:#07090f!important;}
div[data-testid="column"]{background:#07090f!important;}
div[data-testid="stHorizontalBlock"]{background:#07090f!important;}
.element-container{background:transparent!important;}
section[data-testid="stSidebar"]{background:#0a0e18;border-right:1px solid #1a2540;}
section[data-testid="stSidebar"] .stMarkdown{color:#c8d8ec;}
div[data-testid="metric-container"]{background:#07090f;border:1px solid #1a2540;border-radius:8px;padding:12px 16px;transition:border-color .2s;}
div[data-testid="metric-container"]:hover{border-color:#243660;}
div[data-testid="metric-container"] label{color:#4a6080!important;font-family:'JetBrains Mono',monospace!important;font-size:9px!important;text-transform:uppercase;letter-spacing:1px;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:#c8d8ec!important;font-family:'JetBrains Mono',monospace!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:#c8d8ec!important;}
.stButton>button{background:linear-gradient(90deg,#f5a623,#d4850e);color:#000;border:none;border-radius:6px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:.5px;transition:opacity .15s;}
.stButton>button:hover{opacity:.85;}
.stSelectbox,.stRadio{color:#c8d8ec;}
.stSelectbox>div>div{background:#07090f;border-color:#1a2540;color:#c8d8ec;}
hr{border-color:#1a2540;}
.status-box{background:#07090f;border:1px solid #1a2540;border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#4a6080;line-height:1.8;white-space:pre;overflow-x:auto;}
.js-plotly-plot{border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
_tmpl = go.layout.Template(layout=go.Layout(
    paper_bgcolor="#07090f", plot_bgcolor="#07090f",
    font=dict(family="JetBrains Mono, monospace", color="#4a6080", size=10),
    colorway=["#00d4ff","#f5a623","#1df5a0","#ff3d5a","#9d7aff","#ffd060"],
    xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
    yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
    legend=dict(bgcolor="rgba(7,9,15,.85)", bordercolor="#1a2540", borderwidth=1),
    margin=dict(l=50,r=20,t=40,b=40),
))
pio.templates["energy_dark"] = _tmpl
pio.templates.default = "plotly_dark+energy_dark"
PT = "plotly_dark+energy_dark"
SCEN_COLORS = ["#00d4ff","#ffd060","#ff6b35","#ff3d5a","#9d7aff"]
HORIZONS    = ["1M","3M","6M","9M","12M"]
_PCFG = {"displayModeBar":False,"displaylogo":False}

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defs = dict(result=None,agent=None,sel_horizon="1M",sel_bin=None,
                sel_scenario=None,sel_region=None,sel_driver=None,log=[])
    for k,v in defs.items():
        if k not in st.session_state: st.session_state[k]=v
init_state()

# ── AGENT RUNNERS (cached) ────────────────────────────────────────────────────
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

# ── HELPERS ───────────────────────────────────────────────────────────────────
def section(num, title, hint=""):
    st.markdown(f"""
    <div style="margin-top:28px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a2540;
                display:flex;align-items:center;justify-content:space-between">
      <span style="font-size:12px;font-weight:700;color:#4a6080;text-transform:uppercase;letter-spacing:1.8px">
        {num} {title}
      </span>
      <span style="font-size:9px;color:#2a3850;font-family:'JetBrains Mono',monospace">{hint}</span>
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

# ── ① SNAPSHOT (renamed from KPI Snapshot; no Retail/Margin for HO) ──────────
def render_snapshot(result, agent):
    section("01", "SNAPSHOT")
    ho = agent=="ho"
    md = result.get("market_data",{})
    f  = result.get("forecast",{})
    ci = result.get("ci_bands",{})
    ci1m = ci.get("1M",{})

    if ho:
        cols = st.columns(5)
        metrics = [
            ("HO Price",          f"${md.get('HO',0):.4f}",       "$/gal"),
            ("WTI",               f"${md.get('WTI',0):.2f}",       "$/bbl"),
            ("RBOB",              f"${md.get('RBOB',0):.4f}",      "$/gal"),
            ("Crack Spread",      f"${md.get('crack_spread',0):.2f}","$/bbl"),
            ("Volatility (VIX)",  f"{md.get('VIX',0):.1f}",        "index"),
        ]
    else:
        cols = st.columns(5)
        metrics = [
            ("WTI Live",      f"${f.get('current_wti',0):.2f}",         "per barrel"),
            ("Brent",         f"${result.get('brent',0):.2f}",           "per barrel"),
            ("1-Wk Forecast", f"${f.get('forecast_low',0)}–${f.get('forecast_high',0)}","90% CI"),
            ("Ann. Vol",      f"{f.get('annualised_vol',0):.1f}%",       "historical sigma"),
            ("Direction",     f.get('direction','—'),                    "model signal"),
        ]
    for col,(label,val,sub) in zip(cols,metrics):
        col.metric(label,val,sub)

    ci_width = round((ci1m.get("ci95",[0,0])[1]-ci1m.get("ci95",[0,0])[0]),4 if ho else 2)
    regime_label = result.get("regime","—") if ho else f.get("direction","—")
    st.caption(f"1M 95% CI range: **${ci_width}** · Regime: **{regime_label}**")


# ── ② PRICE HISTORY ──────────────────────────────────────────────────────────
def render_price_history(result, agent):
    """Price history — dynamic y-axis, period selector, synthetic data warning."""
    section("02", "PRICE HISTORY", "Select time period below")
    ho   = agent=="ho"
    hist = result.get("history",[])

    # Warn loudly if data is synthetic (all real sources failed)
    is_synthetic = result.get("is_synthetic_history", False)
    if not is_synthetic:
        is_synthetic = any(r.get("synthetic") for r in hist)
    if is_synthetic:
        st.error(
            "**DATA WARNING: Price history is SYNTHETIC (simulated) — all live data sources "
            "failed to respond.** The chart below does NOT reflect real market prices. "
            "Check the Run Log at the bottom of the page for details on which sources failed."
        )

    if len(hist)<2:
        st.info("Not enough history data yet.")
        return

    # Time-period filter (sanitized)
    period_opts = ["1 Week","1 Month","3 Months","6 Months","1 Year","All"]
    period = st.radio("Period", period_opts, horizontal=True, index=2,
                      key="period_radio")
    period_s = sanitize_str(period)
    try: validate_enum(period_s, set(period_opts))
    except ValueError: period_s = "3 Months"

    cutoff_map = {"1 Week":7,"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"All":99999}
    cutoff_days = cutoff_map.get(period_s, 90)
    cutoff_date = datetime.date.today() - datetime.timedelta(days=cutoff_days)

    filtered = [r for r in hist if str(r["date"])>=str(cutoff_date)]
    if len(filtered)<2: filtered = hist[-10:]

    labels  = [r["date"] for r in filtered]
    prices  = [float(r["price"]) for r in filtered]
    maN     = min(20, len(prices)//2) if ho else min(7, len(prices)//2)
    ma      = [np.mean(prices[max(0,i-maN+1):i+1]) if i>=maN-1 else None for i in range(len(prices))]
    color   = "#f5a623" if ho else "#00d4ff"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels,y=prices,name="Price",
        line=dict(color=color,width=2),
        fill="tozeroy",fillcolor=f"rgba({'245,166,35' if ho else '0,212,255'},.06)"))
    fig.add_trace(go.Scatter(x=labels,y=ma,name=f"{maN}d MA",
        line=dict(color="#9d7aff",width=1.5,dash="dot")))

    if ho:
        p_min = min(prices)
        p_max = max(prices)
        yrange = [round(p_min - 0.25, 4), round(p_max + 0.25, 4)]
    else:
        yrange = None
    fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=340,
        title=dict(text=f"{'Heating Oil' if ho else 'WTI Crude'} — Price History",
                   font=dict(size=12,color="#c8d8ec")),
        xaxis=dict(rangeslider=dict(visible=True,bgcolor="#07090f"),type="date"),
        yaxis=dict(tickformat="$.4f" if ho else "$.2f",range=yrange),
        hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("price_hist"))


# ── ③ PROBABILITY DISTRIBUTION ───────────────────────────────────────────────
def render_prob_dist(result, agent, sel_h, sel_bin):
    section("03","PROBABILITY DISTRIBUTION","Horizon selector in sidebar")
    ho   = agent=="ho"
    c1,c2 = st.columns(2)
    rows = result.get("prob_table",{}).get(sel_h,[])
    if not rows: return

    bins  = [r[0] for r in rows]
    probs = [round(r[1]*100,2) for r in rows]
    maxP  = max(probs)
    colors = []
    for b,p in zip(bins,probs):
        if sel_bin and b==sel_bin: colors.append("#00d4ff")
        elif sel_bin:              colors.append("rgba(0,212,255,.2)")
        elif p==maxP:              colors.append("#f5a623")
        else:                      colors.append("#00d4ff" if ho else "#f5a623")

    with c1:
        fig=go.Figure(go.Bar(y=bins,x=probs,orientation="h",marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],textposition="outside",
            hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=360,
            title=dict(text=f"Probability by Price Bin — {sel_h}",font=dict(size=11,color="#c8d8ec")),
            xaxis=dict(title="Probability (%)",ticksuffix="%"),
            yaxis=dict(autorange="reversed"),bargap=0.15,showlegend=False)
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("prob_bar"))

    with c2:
        # CDF
        cum=0; cdf=[]
        for _,p in rows: cum+=p; cdf.append(round(cum*100,2))
        fig=go.Figure(go.Scatter(x=bins,y=cdf,mode="lines+markers",
            line=dict(color="#9d7aff",width=2),marker=dict(size=4,color="#9d7aff"),
            fill="tozeroy",fillcolor="rgba(157,122,255,.07)",
            hovertemplate="%{x}: P(<=) = %{y:.1f}%<extra></extra>"))
        fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=360,
            title=dict(text=f"Cumulative Distribution — {sel_h}",font=dict(size=11,color="#c8d8ec")),
            yaxis=dict(title="Cumulative Probability (%)",ticksuffix="%",range=[0,100]),
            xaxis=dict(title="Price Range"))
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("cdf"))

    # EV row
    ev = result.get("ev_by_horizon",{})
    if ev:
        st.markdown("**Expected Value (EV) by Horizon** — probability-weighted average price")
        ev_cols = st.columns(len(HORIZONS))
        for col,h in zip(ev_cols,HORIZONS):
            col.metric(h,f"${ev.get(h,0):.4f}" if ho else f"${ev.get(h,0):.2f}")

    # Probability table
    st.markdown("**Probability Table** — all horizons")
    _render_prob_table(result, agent, sel_h, sel_bin)

    # Log-normal shape (HO only)
    if ho:
        ls = result.get("lognorm_shape",{})
        if ls:
            st.markdown("**Log-Normal HO Price Distribution Shape (1M horizon)**")
            c_a,c_b = st.columns([3,1])
            with c_a:
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=ls["x"],y=ls["y"],mode="lines",
                    line=dict(color="#f5a623",width=2),fill="tozeroy",
                    fillcolor="rgba(245,166,35,.10)",
                    hovertemplate="Price: $%{x:.4f}<br>PDF: %{y:.5f}<extra></extra>"))
                fig.add_vline(x=ls["mean"],  line=dict(color="#00d4ff",dash="dash",width=1.5),
                              annotation_text=f"Mean ${ls['mean']:.4f}",
                              annotation_font=dict(color="#00d4ff",size=9))
                fig.add_vline(x=ls["median"],line=dict(color="#1df5a0",dash="dot",width=1.5),
                              annotation_text=f"Median ${ls['median']:.4f}",
                              annotation_font=dict(color="#1df5a0",size=9))
                fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=260,
                    title=dict(text="Log-Normal PDF — shape, skewness, kurtosis",font=dict(size=11,color="#c8d8ec")),
                    xaxis=dict(title="HO Price ($/gal)"),yaxis=dict(title="Probability Density"),showlegend=False)
                st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("lognorm"))
            with c_b:
                st.markdown("""<div style="padding:16px;background:#0d1220;border:1px solid #1a2540;border-radius:8px;
                    font-family:'JetBrains Mono',monospace;font-size:11px;line-height:2;color:#c8d8ec">""",
                    unsafe_allow_html=True)
                st.metric("Mean",    f"${ls['mean']:.4f}")
                st.metric("Median",  f"${ls['median']:.4f}")
                st.metric("Skewness",f"{ls['skewness']:.3f}")
                st.metric("Kurtosis",f"{ls['kurtosis']:.3f}")
                st.markdown("</div>",unsafe_allow_html=True)


# ── ④ VOLATILITY (dynamic y-axis + period filter) ────────────────────────────
def render_volatility(result):
    section("04","VOLATILITY","Rolling 10-day annualised vol — line chart")
    vh = result.get("vol_heatmap",[])
    if not vh:
        st.info("Insufficient history for volatility (need > 11 trading days)")
        return

    df_full = pd.DataFrame(vh)
    df_full["date"] = pd.to_datetime(df_full["date"])

    # ── Period selector ───────────────────────────────────────────────────────
    period_opts = ["1M","3M","6M","1Y"]
    period_days = {"1M":30,"3M":90,"6M":180,"1Y":365}
    period = st.radio("Volatility period",period_opts,index=1,
                      horizontal=True,key="vol_period_radio")
    try:    validate_enum(sanitize_str(period),set(period_opts))
    except ValueError: period="3M"

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=period_days[period])
    df     = df_full[df_full["date"]>=cutoff].copy()
    if df.empty: df = df_full.tail(10).copy()

    # ── Dynamic y-axis: ceiling = max+5%, floor = min-15% ────────────────────
    v_max   = float(df["vol"].max())
    v_min   = float(df["vol"].min())
    y_upper = round(v_max * 1.05, 2)
    y_lower = round(v_min * 0.85, 2)
    avg     = float(df["vol"].mean())

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"],y=df["vol"],mode="lines+markers",
        line=dict(color="#f5a623" if result.get("agent")=="ho" else "#00d4ff",width=2),
        marker=dict(size=4),name="Rolling 10d Ann. Vol",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=avg,line=dict(color="#ffd060",width=1,dash="dash"),
        annotation_text=f"Avg {avg:.1f}%",annotation_font=dict(color="#ffd060",size=9))
    fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=280,
        title=dict(text=f"Rolling 10-Day Annualised Volatility — {period}",
                   font=dict(size=11,color="#c8d8ec")),
        xaxis=dict(title="",type="date"),
        yaxis=dict(title="Ann. Vol (%)",ticksuffix="%",range=[y_lower,y_upper]),
        showlegend=False)
    st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("vol"))


# ── ⑥ SCENARIO (dynamic signals + VIX-scaled vol) ────────────────────────────
def render_scenario(result, agent, sel_scen):
    section("06","SCENARIO SIMULATION","Dynamic signals: crack spread · VIX · EIA · seasonal")
    sp   = result.get("scenario_paths",{})
    ho   = agent=="ho"
    md   = result.get("market_data",{})
    f    = result.get("forecast",{})
    spot = md.get("HO",result.get("ho_price",3.5)) if ho else f.get("current_wti",result.get("wti",80))
    sigs = result.get("scenario_signals",{})

    # Signal dashboard (HO only)
    if ho and sigs:
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        base_drift = sigs.get("base_dynamic_drift_ann",0)
        sc1.metric("Base Drift (ann.)", f"{base_drift:+.2f}%",
            help="Composite dynamic drift from all live signals (annualised)")
        sc2.metric("VIX Vol Mult", f"{sigs.get('vix_vol_mult',1):.2f}×",
            help="VIX current ÷ 20d rolling mean — scales scenario volatility")
        sc3.metric("Crack Signal", f"{sigs.get('crack_signal_ann',0):+.2f}%",
            help="Crack spread above/below $15 threshold (annualised drift contribution)")
        sc4.metric("Seasonal Signal", f"{sigs.get('seasonal_signal_ann',0):+.2f}%",
            help="Heating season (Nov–Mar) = bullish, summer = bearish")
        sc5.metric("EIA Signal", f"{sigs.get('eia_signal_ann',0):+.2f}%",
            help="Weekly inventory draw (+) or build (−) contribution")

        # Signal decomposition bar
        sig_names  = ["Crack Spread","VIX","EIA Inventory","Seasonal"]
        sig_values = [
            sigs.get("crack_signal_ann",0),
            sigs.get("vix_signal_ann",0),
            sigs.get("eia_signal_ann",0),
            sigs.get("seasonal_signal_ann",0),
        ]
        sig_colors = ["#1df5a0" if v>=0 else "#ff3d5a" for v in sig_values]
        fig_sig = go.Figure(go.Bar(
            x=sig_names, y=sig_values, marker_color=sig_colors,
            text=[f"{v:+.2f}%" for v in sig_values], textposition="outside",
            hovertemplate="%{x}: %{y:+.2f}% ann. drift<extra></extra>"))
        fig_sig.add_hline(y=0, line=dict(color="#4a6080", width=1))
        fig_sig.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=180,
            title=dict(text="Live Signal Decomposition — Annualised Drift Contribution (%)",
                       font=dict(size=10, color="#c8d8ec")),
            yaxis=dict(title="Ann. drift (%)", ticksuffix="%"),
            showlegend=False, bargap=0.3, margin=dict(l=40,r=20,t=36,b=30))
        st.plotly_chart(fig_sig, use_container_width=True, config=_PCFG, key=_pc("sig_decomp"))

    # Scenario paths
    fig=go.Figure()
    for i,(sname,path) in enumerate(sp.items()):
        opa = 1.0 if not sel_scen or sel_scen==sname else 0.2
        wid = 2.5 if not sel_scen or sel_scen==sname else 1
        hover_lbl = path.get("label","") if ho else sname
        fig.add_trace(go.Scatter(x=["Today"]+path["dates"],y=[spot]+path["prices"],
            name=sname, line=dict(color=SCEN_COLORS[i%5],width=wid), opacity=opa,
            hovertemplate=f"<b>{sname}</b><br>${{y:.4f}}<br><i>{hover_lbl}</i><extra></extra>"))
    fmt="$.4f" if ho else "$.2f"
    fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=340,
        title=dict(text="14-Day Scenario Simulation Paths (Dynamic Drift + VIX-Scaled Vol)",
                   font=dict(size=11,color="#c8d8ec")),
        yaxis=dict(tickformat=fmt),hovermode="x unified",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("scen"))

    # Scenario metadata table (HO only — shows drift/vol per scenario)
    if ho and sp:
        st.markdown("**Scenario Parameters** — effective drift and volatility after signal adjustments")
        rows_html = ""
        th = "padding:7px 14px;background:#0d1220;color:#4a6080;font-family:'JetBrains Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid #1a2540;text-align:center;"
        td = "padding:6px 14px;font-family:'JetBrains Mono',monospace;font-size:11px;border-bottom:1px solid #0f1825;text-align:center;"
        for i,(sname,path) in enumerate(sp.items()):
            drift = path.get("total_drift", 0)
            vol   = path.get("vol_ann", 0)
            lbl   = path.get("label","")
            bg    = "background:#0a1e30;" if sel_scen==sname else ""
            dc    = "color:#1df5a0;" if drift>=0 else "color:#ff3d5a;"
            rows_html += f"""<tr>
              <td style="{td}{bg}color:{SCEN_COLORS[i%5]};font-weight:700">{sname}</td>
              <td style="{td}{bg}{dc}">{drift:+.1f}%</td>
              <td style="{td}{bg}color:#9d7aff;">{vol:.1f}%</td>
              <td style="{td}{bg}color:#4a6080;font-size:9px;text-align:left">{lbl}</td>
            </tr>"""
        st.markdown(
            f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #1a2540;margin-bottom:16px">'
            f'<table style="width:100%;border-collapse:collapse;background:#07090f">'
            f'<thead><tr>'
            f'<th style="{th}text-align:left">Scenario</th>'
            f'<th style="{th}">Drift (ann.)</th>'
            f'<th style="{th}">Vol (ann.)</th>'
            f'<th style="{th}text-align:left">Driver Logic</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    # Final prices bar
    with c1:
        names_s=[s for s in sp]; finals=[sp[s]["final"] for s in names_s]
        _DIM=["rgba(0,212,255,.2)","rgba(255,208,96,.2)","rgba(255,107,53,.2)","rgba(255,61,90,.2)","rgba(157,122,255,.2)"]
        cols_s=[SCEN_COLORS[i%5] if (not sel_scen or sel_scen==n) else _DIM[i%5] for i,n in enumerate(names_s)]
        fig=go.Figure(go.Bar(x=names_s,y=finals,marker_color=cols_s,
            text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in finals],textposition="outside"))
        fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=240,
            title=dict(text="Scenario Final Prices (Day 14)",font=dict(size=10,color="#c8d8ec")),
            yaxis=dict(tickformat="$.4f" if ho else "$.2f"),showlegend=False,bargap=0.2)
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("scen_final"))
    # CI width
    with c2:
        cib=result.get("ci_bands",{})
        w95=[round((cib.get(h,{}).get("ci95",[0,0])[1]-cib.get(h,{}).get("ci95",[0,0])[0]),4) for h in HORIZONS]
        w80=[round((cib.get(h,{}).get("ci80",[0,0])[1]-cib.get(h,{}).get("ci80",[0,0])[0]),4) for h in HORIZONS]
        mids=[cib.get(h,{}).get("mid",0) for h in HORIZONS]
        fig=go.Figure()
        fig.add_trace(go.Bar(x=HORIZONS,y=w95,name="95% CI Width",marker_color="rgba(29,245,160,.65)"))
        fig.add_trace(go.Bar(x=HORIZONS,y=w80,name="80% CI Width",marker_color="rgba(0,212,255,.5)"))
        fig.add_trace(go.Scatter(x=HORIZONS,y=mids,name="Midpoint",mode="lines+markers",
            line=dict(color="#ffd060",width=1.5,dash="dot"),marker=dict(size=5)))
        fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=240,
            title=dict(text="Forecast Uncertainty by Horizon (CI Width)",font=dict(size=10,color="#c8d8ec")),
            yaxis=dict(title="Width ($)",tickformat="$.4f" if ho else "$.2f"),
            barmode="overlay",bargap=0.2,legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc("ci_width"))


# ── ⑦ REGIONAL MAP ───────────────────────────────────────────────────────────
def render_regional(result, agent, sel_reg):
    section("07","REGIONAL PRICE MAP","United States · Brazil — Green = below avg, Red = above avg")
    rp    = result.get("regional_prices",[])       # US
    br_rp = result.get("brazil_regional_prices",[]) # Brazil
    ho    = agent=="ho"
    if not rp: return

    # ── Helper: build one Scattergeo figure ───────────────────────────────────
    def _make_map(data, scope, title, proj, center_lat=None, center_lon=None, scale=None):
        avg_p = sum(r["price"] for r in data) / len(data)
        fig   = go.Figure()
        for r in data:
            sel  = sel_reg == r["region"]
            clr  = "#00d4ff" if sel else ("#ff3d5a" if r["price"] > avg_p else "#1df5a0")
            delt = (r["price"] - avg_p) / avg_p * 100
            fig.add_trace(go.Scattergeo(
                lat=[r["lat"]], lon=[r["lon"]],
                mode="markers+text",
                marker=dict(
                    size=24 if sel else 17,
                    color=clr,
                    opacity=1.0 if not sel_reg or sel else 0.3,
                    line=dict(width=2 if sel else 0.5,
                              color="#fff" if sel else "rgba(255,255,255,.3)")),
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
            bgcolor="#07090f", landcolor="#0d1220",
            coastlinecolor="#1a2540", showlakes=False,
            showrivers=False, framecolor="#1a2540",
            showocean=True, oceancolor="#07090f",
            showcountries=True, countrycolor="#1a2540")
        if scope:
            geo_cfg["scope"] = scope
        if proj:
            geo_cfg["projection_type"] = proj
        if center_lat is not None:
            geo_cfg["center"] = dict(lat=center_lat, lon=center_lon)
        if scale is not None:
            geo_cfg["projection_scale"] = scale
        fig.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            height=320,
            title=dict(text=title, font=dict(size=11, color="#c8d8ec")),
            geo=geo_cfg,
            margin=dict(l=0, r=0, t=36, b=0))
        return fig, avg_p

    # ── Helper: price bar chart ────────────────────────────────────────────────
    def _make_bar(data, avg_p, title, key):
        sorted_d  = sorted(data, key=lambda r: r["price"], reverse=True)
        names_r   = [r["state"] + " (" + r["region"].split()[0] + ")" for r in sorted_d]
        prices_r  = [r["price"] for r in sorted_d]
        colors_r  = [
            "#00d4ff" if sel_reg == r["region"]
            else ("#ff3d5a" if r["price"] > avg_p else "#1df5a0")
            for r in sorted_d]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=names_r, y=prices_r, marker_color=colors_r,
            text=[f"${p:.4f}" for p in prices_r], textposition="outside"))
        fig.add_hline(y=avg_p,
            line=dict(color="#ffd060", width=1.5, dash="dash"),
            annotation_text=f"Avg ${avg_p:.4f}",
            annotation_font=dict(color="#ffd060", size=9))
        fig.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            height=260,
            title=dict(text=title, font=dict(size=10, color="#c8d8ec")),
            yaxis=dict(tickformat="$.4f"), showlegend=False, bargap=0.2,
            margin=dict(l=40, r=10, t=36, b=60))
        return fig

    # ── US Map ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:10px;font-weight:700;color:#4a6080;font-family:\'JetBrains Mono\','
        'monospace;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px">'
        'UNITED STATES</div>', unsafe_allow_html=True)

    us_m1, us_m2 = st.columns([3, 2])
    with us_m1:
        fig_us, us_avg = _make_map(
            rp, scope="usa",
            title="US Regional Distillate Price Distribution",
            proj=None)
        st.plotly_chart(fig_us, use_container_width=True, config=_PCFG, key=_pc("map_us"))
    with us_m2:
        fig_us_bar = _make_bar(rp, us_avg, "US Regional Prices", "us_bar")
        st.plotly_chart(fig_us_bar, use_container_width=True, config=_PCFG, key=_pc("region_bar_us"))

    # ── Brazil Map ────────────────────────────────────────────────────────────
    if br_rp:
        st.markdown(
            '<div style="font-size:10px;font-weight:700;color:#4a6080;font-family:\'JetBrains Mono\','
            'monospace;text-transform:uppercase;letter-spacing:1.2px;margin:16px 0 6px">'
            'BRAZIL</div>', unsafe_allow_html=True)

        br_m1, br_m2 = st.columns([3, 2])
        with br_m1:
            fig_br, br_avg = _make_map(
                br_rp, scope=None,
                title="Brazil Regional Diesel/Distillate Price Distribution",
                proj="mercator",
                center_lat=-14.0, center_lon=-52.0,
                scale=2.8)
            st.plotly_chart(fig_br, use_container_width=True, config=_PCFG, key=_pc("map_br"))
        with br_m2:
            fig_br_bar = _make_bar(br_rp, br_avg, "Brazil Regional Prices", "br_bar")
            st.plotly_chart(fig_br_bar, use_container_width=True, config=_PCFG, key=_pc("region_bar_br"))

        # ── Cross-country comparison bar ───────────────────────────────────────
        st.markdown("**US vs Brazil — Regional Price Comparison** ($/gal equivalent)")
        all_data = [(r["state"], r["price"], "US", "#00d4ff") for r in rp] + \
                   [(r["state"], r["price"], "BR", "#f5a623") for r in br_rp]
        all_data.sort(key=lambda x: x[1], reverse=True)
        fig_cmp = go.Figure()
        us_prices_all = [r["price"] for r in rp]
        br_prices_all = [r["price"] for r in br_rp]
        fig_cmp.add_trace(go.Bar(
            x=[d[0] for d in all_data],
            y=[d[1] for d in all_data],
            marker_color=[d[3] for d in all_data],
            text=[f"${d[1]:.4f} ({d[2]})" for d in all_data],
            textposition="outside",
            hovertemplate="%{x}: $%{y:.4f}/gal<extra></extra>"))
        fig_cmp.add_hline(y=float(np.mean(us_prices_all)),
            line=dict(color="#00d4ff", width=1, dash="dot"),
            annotation_text=f"US avg ${np.mean(us_prices_all):.4f}",
            annotation_font=dict(color="#00d4ff", size=8))
        fig_cmp.add_hline(y=float(np.mean(br_prices_all)),
            line=dict(color="#f5a623", width=1, dash="dot"),
            annotation_text=f"BR avg ${np.mean(br_prices_all):.4f}",
            annotation_font=dict(color="#f5a623", size=8))
        fig_cmp.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f",
            height=280,
            title=dict(text="All Regions Ranked by Price — Blue = US, Orange = Brazil",
                       font=dict(size=10, color="#c8d8ec")),
            yaxis=dict(title="$/gal", tickformat="$.4f"),
            showlegend=False, bargap=0.15,
            margin=dict(l=40, r=10, t=36, b=60))
        st.plotly_chart(fig_cmp, use_container_width=True, config=_PCFG, key=_pc("region_cmp"))

        # Caption with delta
        us_avg_v = float(np.mean(us_prices_all))
        br_avg_v = float(np.mean(br_prices_all))
        delta_pct = (br_avg_v - us_avg_v) / us_avg_v * 100
        st.caption(
            f"US avg: **${us_avg_v:.4f}/gal** · Brazil avg: **${br_avg_v:.4f}/gal** · "
            f"Brazil is **{delta_pct:+.1f}%** vs US average · "
            f"Brazil prices anchored to Petrobras refinery gate + state ICMS taxes")


# ── ⑧ EIA INVENTORY DEEP DIVE ────────────────────────────────────────────────
def render_eia_deep_dive(result):
    section("08","EIA INVENTORY DEEP DIVE","Seasonal band · WoW momentum · 5-year range")
    eia = result.get("eia_data",{})
    hist = eia.get("history",[])

    c1,c2,c3 = st.columns(3)
    s  = eia.get("stocks_mbbl")
    wow= eia.get("wow_change")
    c1.metric("Latest Stocks", f"{s:,.0f} Mbbl" if s else "N/A")
    c2.metric("Week-on-Week",  f"{wow:+,.0f} Mbbl" if wow else "N/A",
              delta=f"{wow:+,.0f}" if wow else None)
    weeks = eia.get("weeks",[])
    if weeks:
        avg4  = sum(w["value"] for w in weeks[:4])/min(4,len(weeks))
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

    # ── Chart 1: Seasonal Inventory Band (Most Important) ─────────────────────
    st.markdown("**Seasonal Inventory Band** — current year vs 5-year historical range")
    seasonal_bands = eia.get("seasonal_bands", [])
    current_yr_data = eia.get("current_year_data", [])

    if seasonal_bands:
        wks  = [b["week"] for b in seasonal_bands]
        avgs = [b["avg"]  for b in seasonal_bands]
        mins = [b["min"]  for b in seasonal_bands]
        maxs = [b["max"]  for b in seasonal_bands]

        fig_sb = go.Figure()
        # Shaded band (min to max)
        fig_sb.add_trace(go.Scatter(
            x=wks+wks[::-1], y=maxs+mins[::-1],
            fill="toself", fillcolor="rgba(0,212,255,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="5yr Min-Max Range",
            hoverinfo="skip"))
        # 5-yr average line
        fig_sb.add_trace(go.Scatter(
            x=wks, y=avgs, mode="lines",
            line=dict(color="#00d4ff", width=1.5, dash="dot"),
            name="5yr Average",
            hovertemplate="Week %{x}: avg %{y:,.0f} Mbbl<extra></extra>"))
        # Current year line
        if current_yr_data:
            cy_wks  = [r["week"]  for r in current_yr_data]
            cy_vals = [r["value"] for r in current_yr_data]
            fig_sb.add_trace(go.Scatter(
                x=cy_wks, y=cy_vals, mode="lines+markers",
                line=dict(color="#f5a623", width=2.5),
                marker=dict(size=5, color="#f5a623"),
                name=f"{datetime.date.today().year}",
                hovertemplate="Week %{x}: %{y:,.0f} Mbbl<extra></extra>"))
            # Current week dot
            if cy_wks:
                fig_sb.add_trace(go.Scatter(
                    x=[cy_wks[-1]], y=[cy_vals[-1]], mode="markers",
                    marker=dict(size=12, color="#ffd060", symbol="star",
                                line=dict(width=2, color="#07090f")),
                    name="Current", showlegend=False,
                    hovertemplate=f"Now: {cy_vals[-1]:,.0f} Mbbl<extra></extra>"))
        fig_sb.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=300,
            title=dict(text="EIA Distillate Stocks — Seasonal Band (Week of Year)",
                       font=dict(size=11, color="#c8d8ec")),
            xaxis=dict(title="ISO Week Number", dtick=4),
            yaxis=dict(title="Mbbl"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_sb, use_container_width=True, config=_PCFG, key=_pc("eia_seasonal"))
    else:
        st.info("Seasonal band requires 5+ years of EIA data. Showing available history instead.")

    # ── Chart 2: WoW Change Bar + Time-Series (side by side) ─────────────────
    st.markdown("**Inventory Levels & Week-over-Week Momentum** — last 16 weeks")
    c_a, c_b = st.columns([3, 2])
    with c_a:
        df_show = df_full.tail(52)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_show["period"], y=df_show["value"],
            mode="lines+markers", line=dict(color="#00d4ff", width=2),
            marker=dict(size=4), fill="tozeroy", fillcolor="rgba(0,212,255,.07)",
            hovertemplate="Week %{x}: %{y:,.0f} Mbbl<extra></extra>", name="Distillate Stocks"))
        if len(df_show) >= 4:
            avg_v = df_show["value"].mean()
            fig.add_hline(y=avg_v, line=dict(color="#ffd060", width=1, dash="dash"),
                annotation_text=f"1yr Avg {avg_v:,.0f}", annotation_font=dict(color="#ffd060", size=9))
        fig.update_layout(template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=260,
            title=dict(text="52-Week Inventory History (Mbbl)", font=dict(size=10, color="#c8d8ec")),
            xaxis=dict(title=""), yaxis=dict(title="Mbbl"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("eia"))

    with c_b:
        df_wow = df_full.tail(17)
        if len(df_wow) >= 2:
            wow_vals  = [round(df_wow["value"].iloc[i] - df_wow["value"].iloc[i-1], 0)
                         for i in range(1, len(df_wow))]
            wow_dates = [df_wow["period"].iloc[i] for i in range(1, len(df_wow))]
            wow_colors= ["#1df5a0" if w >= 0 else "#ff3d5a" for w in wow_vals]
            fig = go.Figure(go.Bar(
                x=wow_dates, y=wow_vals, marker_color=wow_colors,
                text=[f"{int(w):+,}" for w in wow_vals], textposition="outside",
                hovertemplate="%{x}: %{y:+,.0f} Mbbl<extra></extra>"))
            fig.add_hline(y=0, line=dict(color="#4a6080", width=1))
            fig.update_layout(template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=260,
                title=dict(text="WoW Change — Last 16 Weeks", font=dict(size=10, color="#c8d8ec")),
                xaxis=dict(tickangle=-45), yaxis=dict(title="Mbbl"), showlegend=False,
                bargap=0.15)
            st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("eia_wow"))


# ── ⑨ CRACK SPREAD HISTORY & ANALYTICS ───────────────────────────────────────
def render_crack_spread(result, agent):
    section("09","CRACK SPREAD ANALYTICS","Time series · percentile bands · distribution · scatter")
    ho = agent=="ho"
    ch = result.get("crack_history",[])
    md = result.get("market_data",{})
    current_crack = md.get("crack_spread")

    if not ch:
        st.info("Crack spread history requires both HO and WTI price history.")
        return

    df_crack = pd.DataFrame(ch)
    cracks   = df_crack["crack"].values

    # Stats sidebar
    pct_rank = float(np.mean(cracks < (current_crack or 0)) * 100) if len(cracks) > 1 else 50.0
    p25, p50, p75 = np.percentile(cracks, [25, 50, 75])

    stat_cols = st.columns(5)
    stat_cols[0].metric("Current Crack",   f"${current_crack:.2f}/bbl" if current_crack else "N/A")
    stat_cols[1].metric("Percentile Rank", f"{pct_rank:.0f}th")
    stat_cols[2].metric("90d Median (P50)",f"${p50:.2f}/bbl")
    stat_cols[3].metric("90d High (P75)",  f"${p75:.2f}/bbl")
    stat_cols[4].metric("90d Low (P25)",   f"${p25:.2f}/bbl")

    # ── Chart 1: Time-series with percentile bands + 21d MA ──────────────────
    ma21 = df_crack["crack"].rolling(21, min_periods=1).mean()
    fig1 = go.Figure()
    # P25/P75 band
    fig1.add_trace(go.Scatter(
        x=list(df_crack["date"]) + list(df_crack["date"])[::-1],
        y=[p75]*len(df_crack) + [p25]*len(df_crack),
        fill="toself", fillcolor="rgba(29,245,160,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="P25–P75 Band", hoverinfo="skip"))
    # P50 line
    fig1.add_hline(y=float(p50), line=dict(color="#1df5a0", width=1, dash="dot"),
        annotation_text=f"P50 ${p50:.2f}", annotation_font=dict(color="#1df5a0", size=9))
    fig1.add_hline(y=float(p25), line=dict(color="#4a6080", width=0.8, dash="dot"),
        annotation_text=f"P25 ${p25:.2f}", annotation_font=dict(color="#4a6080", size=8))
    fig1.add_hline(y=float(p75), line=dict(color="#4a6080", width=0.8, dash="dot"),
        annotation_text=f"P75 ${p75:.2f}", annotation_font=dict(color="#4a6080", size=8))
    # Crack spread line
    fig1.add_trace(go.Scatter(x=df_crack["date"], y=df_crack["crack"],
        mode="lines", line=dict(color="#1df5a0", width=2),
        fill="tozeroy", fillcolor="rgba(29,245,160,.06)",
        hovertemplate="%{x}: $%{y:.2f}/bbl<extra></extra>", name="Crack Spread"))
    # 21d MA
    fig1.add_trace(go.Scatter(x=df_crack["date"], y=ma21,
        mode="lines", line=dict(color="#9d7aff", width=1.5, dash="dot"),
        name="21d MA", hovertemplate="%{x}: MA $%{y:.2f}<extra></extra>"))
    # Current marker
    if current_crack:
        fig1.add_hline(y=current_crack, line=dict(color="#ff3d5a", width=1.5, dash="dash"),
            annotation_text=f"Now ${current_crack:.2f}",
            annotation_font=dict(color="#ff3d5a", size=9))
    fig1.update_layout(
        template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=280,
        title=dict(text="Crack Spread History with Percentile Bands (90d)",
                   font=dict(size=11, color="#c8d8ec")),
        xaxis=dict(type="date"), yaxis=dict(title="Crack Spread ($/bbl)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True, config=_PCFG, key=_pc("crack_ts"))

    c1, c2 = st.columns(2)

    # ── Chart 2: Histogram with current percentile marker ────────────────────
    with c1:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=cracks, nbinsx=20,
            marker=dict(color="#1df5a0", opacity=0.75, line=dict(color="#07090f", width=1)),
            name="Distribution",
            hovertemplate="Crack $%{x:.2f}: %{y} days<extra></extra>"))
        if current_crack:
            fig2.add_vline(x=current_crack,
                line=dict(color="#ff3d5a", width=2, dash="dash"),
                annotation_text=f"Now ${current_crack:.2f} ({pct_rank:.0f}th pct)",
                annotation_font=dict(color="#ff3d5a", size=9))
        fig2.add_vline(x=float(p50), line=dict(color="#1df5a0", width=1, dash="dot"),
            annotation_text="P50", annotation_font=dict(color="#1df5a0", size=8))
        fig2.update_layout(
            template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=260,
            title=dict(text="Crack Spread Distribution — 90 Days",
                       font=dict(size=10, color="#c8d8ec")),
            xaxis=dict(title="Crack Spread ($/bbl)"),
            yaxis=dict(title="Days"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True, config=_PCFG, key=_pc("crack_hist"))

    # ── Chart 3: Crack vs HO Price Scatter with regression ───────────────────
    with c2:
        ho_hist = result.get("history", [])
        # Build aligned crack/HO series by date
        ho_map = {r["date"]: r["price"] for r in ho_hist}
        scatter_x, scatter_y = [], []
        for row in ch:
            if row["date"] in ho_map:
                scatter_x.append(row["crack"])
                scatter_y.append(ho_map[row["date"]])

        if len(scatter_x) >= 5:
            # Regression line
            m, b = np.polyfit(scatter_x, scatter_y, 1)
            x_line = [min(scatter_x), max(scatter_x)]
            y_line = [m*x+b for x in x_line]

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=scatter_x, y=scatter_y, mode="markers",
                marker=dict(color="#9d7aff", size=5, opacity=0.6),
                name="Historical",
                hovertemplate="Crack $%{x:.2f} → HO $%{y:.4f}<extra></extra>"))
            fig3.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color="#ffd060", width=1.5, dash="dot"),
                name=f"Regression (slope={m:.4f})",
                hoverinfo="skip"))
            # Current day marker
            if current_crack and current_crack in [row["crack"] for row in ch]:
                cur_ho = result.get("market_data",{}).get("HO", 0)
                fig3.add_trace(go.Scatter(
                    x=[current_crack], y=[cur_ho], mode="markers",
                    marker=dict(color="#ff3d5a", size=14, symbol="star",
                                line=dict(width=2, color="#07090f")),
                    name="Today",
                    hovertemplate=f"Today: crack ${current_crack:.2f} → HO ${cur_ho:.4f}<extra></extra>"))
            fig3.update_layout(
                template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=260,
                title=dict(text="Crack Spread vs HO Price — Historical Relationship",
                           font=dict(size=10, color="#c8d8ec")),
                xaxis=dict(title="Crack Spread ($/bbl)"),
                yaxis=dict(title="HO Price ($/gal)", tickformat="$.4f"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig3, use_container_width=True, config=_PCFG, key=_pc("crack_scatter"))
        else:
            st.info("Insufficient data for crack vs HO scatter plot.")


# ── ⑩-A SEASONAL PATTERN ANALYSIS ────────────────────────────────────────────
def render_seasonal_pattern(result, agent):
    if agent != "ho":
        return
    section("10","SEASONAL PATTERN ANALYSIS","Monthly average vs current — cycle positioning")
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

    # Overall seasonal avg for delta
    overall_avg = float(np.mean([p for v in month_buckets.values() for p in v])) if month_buckets else cur_ho
    current_m_avg = avgs[current_m - 1] or cur_ho
    delta_vs_seasonal = round((cur_ho - current_m_avg) / current_m_avg * 100, 2) if current_m_avg else 0

    d1, d2 = st.columns(2)
    d1.metric("Current Month Avg (hist.)", f"${current_m_avg:.4f}/gal")
    d2.metric("Live vs Seasonal Avg", f"{delta_vs_seasonal:+.2f}%",
        delta=f"{delta_vs_seasonal:+.2f}%")

    # Bar chart — monthly average with current month highlighted
    bar_colors = []
    for m in months:
        if m == current_m:
            bar_colors.append("#ffd060")
        elif m in (11, 12, 1, 2, 3):
            bar_colors.append("rgba(245,166,35,0.7)")   # heating season
        else:
            bar_colors.append("rgba(0,212,255,0.45)")   # off-season

    valid_months = [m for m in months if avgs[m-1] is not None]
    valid_avgs   = [avgs[m-1] for m in valid_months]
    valid_labels = [month_abbr[m-1] for m in valid_months]
    valid_colors = [bar_colors[m-1] for m in valid_months]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=valid_labels, y=valid_avgs, marker_color=valid_colors,
        text=[f"${v:.4f}" for v in valid_avgs], textposition="outside",
        hovertemplate="%{x}: avg $%{y:.4f}/gal<extra></extra>", name="Monthly Avg"))
    # Live price horizontal line
    fig.add_hline(y=cur_ho, line=dict(color="#ff3d5a", width=1.5, dash="dash"),
        annotation_text=f"Live ${cur_ho:.4f}",
        annotation_font=dict(color="#ff3d5a", size=9))
    # Overall avg
    fig.add_hline(y=overall_avg, line=dict(color="#4a6080", width=1, dash="dot"),
        annotation_text=f"All-period avg ${overall_avg:.4f}",
        annotation_font=dict(color="#4a6080", size=8))
    fig.update_layout(
        template=PT, paper_bgcolor="#07090f", plot_bgcolor="#07090f", height=300,
        title=dict(
            text="Monthly Average HO Price — Seasonal Cycle  |  Orange = heating season  |  Gold = current month",
            font=dict(size=11, color="#c8d8ec")),
        xaxis=dict(title=""), yaxis=dict(title="Avg Price ($/gal)", tickformat="$.4f"),
        showlegend=False, bargap=0.2)
    st.plotly_chart(fig, use_container_width=True, config=_PCFG, key=_pc("seasonal_bar"))

    st.caption(
        f"Current month ({calendar.month_name[current_m]}) historical avg: **${current_m_avg:.4f}** · "
        f"Live price: **${cur_ho:.4f}** · vs seasonal avg: **{delta_vs_seasonal:+.2f}%**")


# ── ⑪ VALUE AT RISK & EXPECTED SHORTFALL ─────────────────────────────────────
def render_var_es(result, agent):
    section("11","VALUE AT RISK (VaR) & EXPECTED SHORTFALL","Monte Carlo — 10,000 simulations")
    ho    = agent=="ho"
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
        bar_colors=["#ff3d5a" if v<0 else "#1df5a0" for v in pnl]
        fig=go.Figure(go.Bar(x=plbls,y=pnl,marker_color=bar_colors,
            text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in pnl],
            textposition="outside",hovertemplate="%{x}: $%{y:.4f}<extra></extra>"))
        fig.add_hline(y=0,line=dict(color="#4a6080",width=1))
        fig.update_layout(template=PT,paper_bgcolor="#07090f",plot_bgcolor="#07090f",height=240,
            title=dict(text=f"P&L Percentile Distribution — {h}",font=dict(size=10,color="#c8d8ec")),
            yaxis=dict(title="P&L ($/gal)" if ho else "P&L ($/bbl)"),
            showlegend=False,bargap=0.15)
        (c1 if ci==0 else c2).plotly_chart(fig,use_container_width=True,config=_PCFG,key=_pc(f"var_{h}"))


# ── Probability table (HTML, dark) ────────────────────────────────────────────
def _render_prob_table(data, agent, sel_horizon, sel_bin):
    rows = data.get("prob_table",{})
    if not rows: return
    ho        = agent=="ho"
    all_bins  = [r[0] for r in rows.get(HORIZONS[0],[])]
    bin_probs = {}
    for h in HORIZONS:
        for bl,prob in rows.get(h,[]):
            if bl not in bin_probs: bin_probs[bl]={}
            bin_probs[bl][h]=prob

    th=("background:#0d1220;color:#4a6080;font-family:'JetBrains Mono',monospace;"
        "font-size:9px;text-transform:uppercase;letter-spacing:.8px;"
        "padding:9px 12px;border-bottom:1px solid #1a2540;text-align:center;white-space:nowrap;")
    th_l=th+"text-align:left;"
    th_a=th+"color:#00d4ff;border-bottom:2px solid #00d4ff;"
    hcells=f'<th style="{th_l}">PRICE RANGE</th>'
    for h in HORIZONS:
        hcells+=f'<th style="{th_a if h==sel_horizon else th}">{h}</th>'

    tbody=""
    for bl in sorted(all_bins,key=lambda b:(b!=sel_bin)):
        is_sel  = bl==sel_bin
        probs   = bin_probs.get(bl,{})
        h_prob  = probs.get(sel_horizon,0)
        is_best = h_prob==max((bin_probs.get(b,{}).get(sel_horizon,0) for b in all_bins),default=0)
        row_bg  = "background:#0a1e30;" if is_sel else ("background:#0a1e18;" if is_best else "")
        td_b    = f"padding:7px 12px;border-bottom:1px solid #0f1825;font-family:'JetBrains Mono',monospace;font-size:11px;text-align:center;{row_bg}"
        lc      = "color:#00d4ff;font-weight:700;" if is_sel else ("color:#1df5a0;font-weight:600;" if is_best else "color:#c8d8ec;")
        cells   = f'<td style="{td_b}text-align:left;{lc}">{bl}</td>'
        for h in HORIZONS:
            p   = probs.get(h,0); pct=f"{p*100:.2f}%"
            w   = min(60,int(p*(400 if ho else 500)))
            bc  = "#00d4ff" if h==sel_horizon else "#1a3050"
            cc  = "color:#00d4ff;" if is_sel else ("color:#c8d8ec;" if h==sel_horizon else "color:#4a6080;")
            bar=(f'<div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">'
                 f'<div style="width:60px;height:4px;background:#0d1220;border-radius:2px;">'
                 f'<div style="width:{w}px;height:100%;background:{bc};border-radius:2px;"></div></div>'
                 f'<span style="min-width:38px;text-align:right;{cc}">{pct}</span></div>')
            cells+=f'<td style="{td_b}">{bar}</td>'
        tbody+=f"<tr>{cells}</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid #1a2540;margin-bottom:16px">'
        f'<table style="width:100%;border-collapse:collapse;background:#07090f;">'
        f'<thead><tr>{hcells}</tr></thead><tbody>{tbody}</tbody></table></div>',
        unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:16px 0 8px">
      <div style="font-size:18px;font-weight:800;color:#c8d8ec;font-family:'Syne',sans-serif">Energy Intel</div>
      <div style="font-size:9px;color:#4a6080;font-family:'JetBrains Mono',monospace;margin-top:3px;letter-spacing:.8px">COMMODITY PROBABILITY ENGINE</div>
    </div>""",unsafe_allow_html=True)
    st.sidebar.divider()

    # Rate-limit info
    sess = st.session_state.get("_session_id","default")
    if "_session_id" not in st.session_state:
        import uuid as _uuid
        st.session_state["_session_id"] = _uuid.uuid4().hex
        sess = st.session_state["_session_id"]
    rem = limiter.remaining(sess)
    st.sidebar.markdown(f'<div style="font-size:9px;color:#2a3850;font-family:\'JetBrains Mono\',monospace">Rate limit: {rem}/5 attempts remaining (15 min window)</div>',unsafe_allow_html=True)

    st.sidebar.markdown("**Run Agent**")
    col1,col2=st.sidebar.columns(2)
    run_oil=col1.button("Oil",  use_container_width=True)
    run_ho =col2.button("HO",   use_container_width=True)

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
        scen_opts_clean=[sanitize_str(s) for s in scen_opts]
        sel_s=st.sidebar.selectbox("Scenario",scen_opts,index=0 if not st.session_state.sel_scenario else
            (scen_opts.index(st.session_state.sel_scenario) if st.session_state.sel_scenario in scen_opts else 0))
        st.session_state.sel_scenario=None if sel_s=="(All)" else sanitize_str(sel_s)

        rp=result.get("regional_prices",[])
        reg_opts=["(All)"]+[r["region"] for r in rp]
        sel_r=st.sidebar.selectbox("Region",reg_opts,index=0 if not st.session_state.sel_region else
            (reg_opts.index(st.session_state.sel_region) if st.session_state.sel_region in reg_opts else 0))
        st.session_state.sel_region=None if sel_r=="(All)" else sanitize_str(sel_r)

        rows=result.get("prob_table",{}).get(h,[])
        bins=["(All)"]+[r[0] for r in rows]
        sel_b=st.sidebar.selectbox("Price Bin",bins,index=0 if not st.session_state.sel_bin else
            (bins.index(st.session_state.sel_bin) if st.session_state.sel_bin in bins else 0))
        st.session_state.sel_bin=None if sel_b=="(All)" else sanitize_str(sel_b)

        if st.sidebar.button("Clear all filters",use_container_width=True):
            st.session_state.update(sel_horizon="1M",sel_bin=None,sel_scenario=None,sel_region=None,sel_driver=None)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="font-size:9px;color:#2a3850;font-family:'JetBrains Mono',monospace;line-height:1.8">
    Contact:<br><a href="mailto:lsaggioro@potonmail.com" style="color:#00d4ff;text-decoration:none">lsaggioro@potonmail.com</a>
    </div>""",unsafe_allow_html=True)


# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
def render_dashboard():
    result  = st.session_state.result
    agent   = st.session_state.agent
    sel_h   = st.session_state.sel_horizon
    sel_bin = st.session_state.sel_bin
    sel_scen= st.session_state.sel_scenario
    sel_reg = st.session_state.sel_region
    sel_drv = st.session_state.sel_driver

    if not result:
        st.markdown("""
        <div style="text-align:center;padding:80px 0">
          <div style="font-size:22px;font-weight:800;color:#c8d8ec;font-family:'Syne',sans-serif;margin-bottom:8px">
            Energy Intelligence Dashboard
          </div>
          <div style="font-size:11px;color:#4a6080;font-family:'JetBrains Mono',monospace;letter-spacing:.8px">
            DETERMINISTIC COMMODITY PROBABILITY ENGINE
          </div>
          <div style="margin-top:32px;font-size:13px;color:#2a3850">
            Click <strong style="color:#f5a623">Oil</strong> or <strong style="color:#00d4ff">HO</strong> in the sidebar to begin
          </div>
        </div>""",unsafe_allow_html=True)
        return

    ho = agent=="ho"

    render_snapshot(result, agent)
    render_price_history(result, agent)
    render_prob_dist(result, agent, sel_h, sel_bin)
    render_volatility(result)
    render_scenario(result, agent, sel_scen)
    render_regional(result, agent, sel_reg)

    if ho:
        render_eia_deep_dive(result)
        render_crack_spread(result, agent)
        render_seasonal_pattern(result, agent)
        render_var_es(result, agent)

    # Summary
    section("--","MARKET SUMMARY")
    with st.expander("View full summary",expanded=False):
        st.code(result.get("summary",""),language=None)

    # Run log
    with st.expander("Run log",expanded=False):
        st.markdown('<div class="status-box">'+"\n".join(st.session_state.log or [])+"</div>",
            unsafe_allow_html=True)

    # Security audit (collapsed)
    with st.expander("Security audit",expanded=False):
        st.code(security_audit_report(),language=None)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_dashboard()

if __name__=="__main__":
    main()
