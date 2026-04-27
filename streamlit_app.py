"""
Energy Intelligence Dashboard — Streamlit Edition
Runs on Streamlit Cloud (free tier) at streamlit.io
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .stCode, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark background override */
.stApp { background: #07090f !important; }

/* Plotly chart iframes - this is the key fix for white chart backgrounds */
.stPlotlyChart { background: #07090f !important; }
.stPlotlyChart > div { background: #07090f !important; }
.stPlotlyChart iframe { background: #07090f !important; }
div[data-testid="stPlotlyChart"] { background: #07090f !important; }
div[data-testid="stPlotlyChart"] > div { background: #07090f !important; }

/* Block containers */
div[data-testid="block-container"] { background: #07090f !important; }
div[data-testid="stVerticalBlock"] { background: #07090f !important; }
div[data-testid="column"] { background: #07090f !important; }

/* Remove any white card backgrounds Streamlit adds */
div[data-testid="stHorizontalBlock"] { background: #07090f !important; }
.element-container { background: transparent !important; }

.stApp { background: #07090f; }
section[data-testid="stSidebar"] { background: #0a0e18; border-right: 1px solid #1a2540; }
section[data-testid="stSidebar"] .stMarkdown { color: #c8d8ec; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #07090f;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 12px 16px;
    transition: border-color .2s;
}
div[data-testid="metric-container"]:hover { border-color: #243660; }
div[data-testid="metric-container"] label {
    color: #4a6080 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c8d8ec !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Section headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #c8d8ec !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #f5a623, #d4850e);
    color: #000;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .5px;
    transition: opacity .15s;
}
.stButton > button:hover { opacity: .85; }

/* Selectbox, radio */
.stSelectbox, .stRadio { color: #c8d8ec; }
.stSelectbox > div > div { background: #07090f; border-color: #1a2540; color: #c8d8ec; }

/* Dividers */
hr { border-color: #1a2540; }

/* Status box */
.status-box {
    background: #07090f;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a6080;
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
}

/* Section badge */
.sec-badge {
    display: inline-block;
    background: #0d1220;
    border: 1px solid #243660;
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

/* Plotly chart containers */
.js-plotly-plot { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
_tmpl_obj = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f",
        font=dict(family="JetBrains Mono, monospace", color="#4a6080", size=10),
        colorway=["#00d4ff", "#f5a623", "#1df5a0", "#ff3d5a", "#9d7aff", "#ffd060"],
        xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
        yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
        legend=dict(bgcolor="rgba(7,9,15,.85)", bordercolor="#1a2540", borderwidth=1),
        margin=dict(l=50, r=20, t=40, b=40),
    )
)
pio.templates["energy_dark"] = _tmpl_obj
pio.templates.default = "plotly_dark+energy_dark"
PLOTLY_TEMPLATE = "plotly_dark+energy_dark"

SCEN_COLORS = ["#00d4ff", "#ffd060", "#ff6b35", "#ff3d5a", "#9d7aff"]
HORIZONS    = ["1M", "3M", "6M", "9M", "12M"]


# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "result": None,
        "agent": None,
        "sel_horizon": "1M",
        "sel_bin": None,
        "sel_scenario": None,
        "sel_region": None,
        "sel_driver": None,
        "log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── AGENT RUNNER ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def run_oil_agent():
    import oil_agent_v2 as oil
    msgs = []
    def capture(m): msgs.append(str(m))
    result = oil.run(send=capture)
    return result, msgs

@st.cache_data(ttl=300, show_spinner=False)
def run_ho_agent():
    import ho_agent as ho
    msgs = []
    def capture(m): msgs.append(str(m))
    result = ho.run(send=capture)
    return result, msgs


# ── HELPERS ───────────────────────────────────────────────────────────────────
def section(icon, title, hint=""):
    st.markdown(f"""
    <div style="margin-top:28px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a2540;
                display:flex;align-items:center;justify-content:space-between">
      <span style="font-size:12px;font-weight:700;color:#4a6080;text-transform:uppercase;letter-spacing:1.8px">
        {icon} {title}
      </span>
      <span style="font-size:9px;color:#2a3850;font-family:'JetBrains Mono',monospace">{hint}</span>
    </div>""", unsafe_allow_html=True)

def next_biz_days(n):
    dates, d = [], datetime.date.today()
    while len(dates) < n:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            dates.append(str(d))
    return dates

def interp_line(start, end, n):
    return [round(start + (end - start) * (i+1)/n, 5) for i in range(n)]


# ── CHART BUILDERS ────────────────────────────────────────────────────────────

def chart_price_ci(data, agent):
    """Price history + multi-band CI forecast — zoomable, pannable."""
    ho  = agent == "ho"
    md  = data.get("market_data", {})
    f   = data.get("forecast", {})
    cib = data.get("ci_bands", {})
    spot = md.get("HO", data.get("ho_price", 3.5)) if ho else f.get("current_wti", data.get("wti", 80))

    hist   = data.get("history", [])
    labels = [r["date"] for r in hist]
    prices = [float(r["price"]) for r in hist]
    if len(prices) < 2:
        st.info("Not enough history data yet.")
        return

    maN = 20 if ho else 7
    ma  = [np.mean(prices[max(0,i-maN+1):i+1]) if i>=maN-1 else None for i in range(len(prices))]

    fc_n     = 14 if ho else 5
    fc_dates = next_biz_days(fc_n)
    ci1m     = cib.get("1M", {})
    fc_mid   = interp_line(spot, ci1m.get("mid", spot), fc_n)
    fc_l80   = interp_line(spot, ci1m.get("ci80", [spot,spot])[0], fc_n)
    fc_h80   = interp_line(spot, ci1m.get("ci80", [spot,spot])[1], fc_n)
    fc_l95   = interp_line(spot, ci1m.get("ci95", [spot,spot])[0], fc_n)
    fc_h95   = interp_line(spot, ci1m.get("ci95", [spot,spot])[1], fc_n)

    all_dates = labels + fc_dates
    pad_none  = [None] * (len(labels) + 1)

    color = "#f5a623" if ho else "#00d4ff"
    fig   = go.Figure()

    # 95% CI fill
    fig.add_trace(go.Scatter(
        x=all_dates, y=pad_none + fc_h95, name="95% CI Hi",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=all_dates, y=pad_none + fc_l95, name="95% CI",
        fill="tonexty", fillcolor="rgba(29,245,160,.08)",
        line=dict(width=0), hoverinfo="skip"))

    # 80% CI fill
    fig.add_trace(go.Scatter(
        x=all_dates, y=pad_none + fc_h80, name="80% CI Hi",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=all_dates, y=pad_none + fc_l80, name="80% CI",
        fill="tonexty", fillcolor="rgba(29,245,160,.18)",
        line=dict(width=0), hoverinfo="skip"))

    # Forecast midpoint
    fig.add_trace(go.Scatter(
        x=all_dates, y=pad_none + fc_mid,
        name="Forecast", line=dict(color="#1df5a0", width=2, dash="dash")))

    # Price line
    fig.add_trace(go.Scatter(
        x=labels, y=prices,
        name="Price", line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({'245,166,35' if ho else '0,212,255'},.06)"))

    # MA
    fig.add_trace(go.Scatter(
        x=labels, y=ma, name=f"{maN}d MA",
        line=dict(color="#9d7aff", width=1.5, dash="dot")))

    # Vertical divider at today
    fig.add_vline(x=labels[-1], line=dict(color="#2a3850", width=1, dash="dash"))
    fig.add_annotation(x=labels[-1], y=1, yref="paper", text="TODAY",
                        showarrow=False, font=dict(color="#2a3850", size=8))

    fmt = "$.4f" if ho else "$.2f"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f",
        height=340,
        title=dict(text=f"{'Heating Oil' if ho else 'WTI Crude'} — {'1-Year' if ho else '90-Day'} History + Multi-Band CI Forecast",
                   font=dict(size=12, color="#c8d8ec")),
        xaxis=dict(rangeslider=dict(visible=True, bgcolor="#07090f"), type="date"),
        yaxis=dict(tickformat=fmt),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_1")


def chart_prob_distribution(data, agent, sel_horizon, sel_bin):
    """Probability bar chart — click triggers cross-filter."""
    ho   = agent == "ho"
    rows = data.get("prob_table", {}).get(sel_horizon, [])
    if not rows:
        return
    bins  = [r[0] for r in rows]
    probs = [round(r[1]*100, 2) for r in rows]
    maxP  = max(probs)

    colors = []
    for b, p in zip(bins, probs):
        if sel_bin and b == sel_bin:
            colors.append("#00d4ff")
        elif sel_bin:
            colors.append("rgba(0,212,255,.2)")
        elif p == maxP:
            colors.append("#f5a623")
        else:
            colors.append("#00d4ff" if ho else "#f5a623")

    fig = go.Figure(go.Bar(
        y=bins, x=probs, orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f",
        height=360,
        title=dict(text=f"Probability by Price Bin — {sel_horizon}", font=dict(size=11, color="#c8d8ec")),
        xaxis=dict(title="Probability (%)", ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        bargap=0.15,
        showlegend=False,
    )
    return fig


def chart_cdf(data, sel_horizon):
    """Cumulative distribution function."""
    rows = data.get("prob_table", {}).get(sel_horizon, [])
    if not rows:
        return
    bins  = [r[0] for r in rows]
    probs = [r[1] for r in rows]
    cdf   = []
    cum   = 0
    for p in probs:
        cum += p
        cdf.append(round(cum * 100, 2))

    fig = go.Figure(go.Scatter(
        x=bins, y=cdf,
        mode="lines+markers",
        line=dict(color="#9d7aff", width=2),
        marker=dict(size=4, color="#9d7aff"),
        fill="tozeroy", fillcolor="rgba(157,122,255,.07)",
        hovertemplate="%{x}: P(≤) = %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f",
        height=360,
        title=dict(text=f"Cumulative Distribution — {sel_horizon}", font=dict(size=11, color="#c8d8ec")),
        yaxis=dict(title="Cumulative Probability (%)", ticksuffix="%", range=[0, 100]),
        xaxis=dict(title="Price Range"),
    )
    return fig


def chart_vol_heatmap(data):
    """Rolling volatility heatmap as scatter/calendar."""
    vh = data.get("vol_heatmap", [])
    if not vh:
        st.info("Insufficient history for volatility heatmap (need > 11 trading days)")
        return

    df   = pd.DataFrame(vh)
    df["date_dt"] = pd.to_datetime(df["date"])
    df["week"]    = df["date_dt"].dt.isocalendar().week.astype(int)
    df["weekday"] = df["date_dt"].dt.dayofweek
    df["label"]   = df["date_dt"].dt.strftime("%b %d") + " — " + df["vol"].astype(str) + "% ann.vol"

    fig = go.Figure(go.Scatter(
        x=df["date"],
        y=df["vol"],
        mode="markers",
        marker=dict(
            size=10,
            color=df["vol"],
            colorscale=[[0, "#1df5a0"], [0.5, "#ffd060"], [1, "#ff3d5a"]],
            showscale=True,
            colorbar=dict(title="Ann. Vol %", ticksuffix="%", len=0.6),
        ),
        text=df["label"],
        hovertemplate="%{text}<extra></extra>",
    ))

    # Add line connecting dots
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["vol"],
        mode="lines",
        line=dict(color="rgba(100,130,170,.3)", width=1),
        showlegend=False, hoverinfo="skip"
    ))

    # Average line
    avg = df["vol"].mean()
    fig.add_hline(y=avg, line=dict(color="#ffd060", width=1, dash="dash"),
                  annotation_text=f"Avg {avg:.1f}%",
                  annotation_font=dict(color="#ffd060", size=9))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f",
        height=260,
        title=dict(text="Rolling 10-Day Annualised Volatility", font=dict(size=11, color="#c8d8ec")),
        xaxis=dict(title="", type="date"),
        yaxis=dict(title="Ann. Vol (%)", ticksuffix="%"),
        showlegend=False,
    )
    return fig


def chart_vol_histogram(data):
    """Histogram of daily vol levels."""
    vh = data.get("vol_heatmap", [])
    if not vh:
        return
    vols = [x["vol"] for x in vh]
    fig  = go.Figure(go.Histogram(
        x=vols, nbinsx=12,
        marker_color="rgba(157,122,255,.7)",
        hovertemplate="Vol: %{x:.1f}% — %{y} days<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=220,
        title=dict(text="Vol Distribution", font=dict(size=10, color="#c8d8ec")),
        xaxis=dict(title="Annualised Vol (%)", ticksuffix="%"),
        yaxis=dict(title="Days"),
        showlegend=False,
    )
    return fig


def chart_drivers(data, agent, sel_driver):
    """Driver / explainability bar chart."""
    drivers = data.get("drivers", [])
    if not drivers:
        return
    ho = agent == "ho"

    names  = [d["name"] for d in drivers]
    values = [d["pct"] for d in drivers]
    colors = []
    for d in drivers:
        if sel_driver and d["name"] == sel_driver:
            colors.append("#00d4ff")
        elif sel_driver:
            colors.append("rgba(0,212,255,.2)")
        else:
            colors.append("#f5a623" if ho else "#00d4ff")

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}% weight<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=260,
        title=dict(text="Price Driver Contribution (rule-based)", font=dict(size=11, color="#c8d8ec")),
        yaxis=dict(title="Relative Weight (%)", ticksuffix="%"),
        xaxis=dict(title=""),
        showlegend=False, bargap=0.2,
    )
    return fig


def chart_driver_donut(data, sel_driver):
    """Driver pie / donut."""
    drivers = data.get("drivers", [])
    if not drivers:
        return
    DCOLS = ["#00d4ff", "#f5a623", "#1df5a0", "#ff3d5a", "#9d7aff", "#ffd060"]
    pull  = [0.12 if (sel_driver and d["name"] == sel_driver) else 0 for d in drivers]

    fig = go.Figure(go.Pie(
        labels=[d["name"] for d in drivers],
        values=[d["pct"] for d in drivers],
        hole=0.55,
        marker_colors=DCOLS[:len(drivers)],
        pull=pull,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=260,
        title=dict(text="Driver Share", font=dict(size=10, color="#c8d8ec")),
        showlegend=True,
        legend=dict(font=dict(size=8)),
    )
    return fig


def chart_scenarios(data, agent, sel_scenario):
    """Scenario simulation paths — line chart."""
    sp   = data.get("scenario_paths", {})
    ho   = agent == "ho"
    md   = data.get("market_data", {})
    f    = data.get("forecast", {})
    spot = md.get("HO", data.get("ho_price", 3.5)) if ho else f.get("current_wti", data.get("wti", 80))

    fc_dates = next_biz_days(14)
    fig = go.Figure()

    for i, (sname, path) in enumerate(sp.items()):
        opa  = 1.0 if not sel_scenario or sel_scenario == sname else 0.2
        wid  = 2.5 if not sel_scenario or sel_scenario == sname else 1
        fig.add_trace(go.Scatter(
            x=["Today"] + path["dates"],
            y=[spot] + path["prices"],
            name=sname,
            line=dict(color=SCEN_COLORS[i % 5], width=wid),
            opacity=opa,
            hovertemplate=f"{sname}: $%{{y:.4f}}<extra></extra>",
        ))

    fmt = "$.4f" if ho else "$.2f"
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=320,
        title=dict(text="14-Day Scenario Simulation Paths", font=dict(size=11, color="#c8d8ec")),
        yaxis=dict(tickformat=fmt),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_scenario_final(data, agent, sel_scenario):
    """Final price bar for each scenario."""
    sp = data.get("scenario_paths", {})
    ho = agent == "ho"
    names  = list(sp.keys())
    finals = [sp[s]["final"] for s in names]
    colors = [SCEN_COLORS[i%5] if (not sel_scenario or sel_scenario==n) else SCEN_COLORS[i%5][:7]+"30"
              for i, n in enumerate(names)]
    fig = go.Figure(go.Bar(
        x=names, y=finals,
        marker_color=colors,
        text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in finals],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.4f}<extra></extra>" if ho else "%{x}: $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
        title=dict(text="Scenario Final Prices (Day 14)", font=dict(size=10, color="#c8d8ec")),
        yaxis=dict(tickformat="$.4f" if ho else "$.2f"),
        showlegend=False, bargap=0.2,
    )
    return fig


def chart_scenario_weights(data, sel_scenario):
    """Scenario weights donut."""
    sw = data.get("scenario_weights", {})
    if not sw:
        return
    labels = [k.replace("_", " ").title() for k in sw]
    values = [round(v*100, 1) for v in sw.values()]
    pull   = [0.12 if (sel_scenario and k.replace("_"," ").title() == sel_scenario) else 0
              for k in sw]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker_colors=SCEN_COLORS[:len(labels)],
        pull=pull,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
        title=dict(text="Scenario Weights (rule-based)", font=dict(size=10, color="#c8d8ec")),
        legend=dict(font=dict(size=8)),
    )
    return fig


def chart_region_map(data, agent, sel_region):
    """US regional price bubble map."""
    rp  = data.get("regional_prices", [])
    ho  = agent == "ho"
    if not rp:
        return
    avg = sum(r["price"] for r in rp) / len(rp)
    df  = pd.DataFrame(rp)
    df["delta_pct"] = ((df["price"] - avg) / avg * 100).round(2)
    df["color"]     = df.apply(lambda r: "#ff3d5a" if r["price"] > avg else "#1df5a0", axis=1)
    df["size"]      = 18
    fmt_price = lambda p: f"${p:.4f}" if ho else f"${p:.3f}"
    df["label"]     = df.apply(lambda r: f"{r['region']}\n{fmt_price(r['price'])}\n{r['delta_pct']:+.1f}% vs avg", axis=1)
    df["selected"]  = df["region"].apply(lambda r: sel_region == r)
    df["opacity"]   = df["region"].apply(lambda r: 1.0 if not sel_region or sel_region == r else 0.3)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Scattergeo(
            lat=[row["lat"]], lon=[row["lon"]],
            mode="markers+text",
            marker=dict(
                size=22 if row["selected"] else 16,
                color=row["color"],
                opacity=row["opacity"],
                line=dict(width=2 if row["selected"] else 0.5, color="#fff" if row["selected"] else "rgba(255,255,255,.3)"),
            ),
            text=[row["state"]],
            textfont=dict(color="#fff", size=8),
            textposition="middle center",
            customdata=[[row["region"], row["price"], row["delta_pct"], row["factor"]]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + ("Price: $%{customdata[1]:.4f}/gal<br>" if ho else "Price: $%{customdata[1]:.3f}/gal<br>")
                + "vs avg: %{customdata[2]:+.1f}%<br>"
                + "%{customdata[3]}<extra></extra>"
            ),
            name=row["region"],
            showlegend=False,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=320,
        title=dict(text="US Regional Price Distribution", font=dict(size=11, color="#c8d8ec")),
        geo=dict(
            scope="usa",
            bgcolor="#07090f",
            landcolor="#0d1220",
            coastlinecolor="#1a2540",
            showlakes=False,
            showrivers=False,
            framecolor="#1a2540",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def chart_region_bar(data, agent, sel_region):
    """Regional price bar chart."""
    rp  = data.get("regional_prices", [])
    ho  = agent == "ho"
    if not rp:
        return
    avg    = sum(r["price"] for r in rp) / len(rp)
    sorted_rp = sorted(rp, key=lambda r: r["price"], reverse=True)
    names  = [r["state"]+" ("+r["region"].split()[0]+")" for r in sorted_rp]
    prices = [r["price"] for r in sorted_rp]
    colors = ["#00d4ff" if sel_region == r["region"] else ("#ff3d5a" if r["price"] > avg else "#1df5a0") for r in sorted_rp]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=prices, marker_color=colors,
        text=[f"${p:.4f}" if ho else f"${p:.3f}" for p in prices],
        textposition="outside",
        hovertemplate=[f"{r['region']}<br>${r['price']:.4f}<br>{r['factor']}<extra></extra>" if ho
                       else f"{r['region']}<br>${r['price']:.3f}<br>{r['factor']}<extra></extra>"
                       for r in sorted_rp],
    ))
    fig.add_hline(y=avg, line=dict(color="#ffd060", width=1.5, dash="dash"),
                  annotation_text=f"Avg ${avg:.4f}" if ho else f"Avg ${avg:.3f}",
                  annotation_font=dict(color="#ffd060", size=9))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=280,
        title=dict(text="Regional Price Comparison", font=dict(size=11, color="#c8d8ec")),
        yaxis=dict(tickformat="$.4f" if ho else "$.3f"),
        showlegend=False, bargap=0.2,
    )
    return fig


def chart_profit_timeline(data, agent):
    """Cost, retail and margin over time."""
    pi  = data.get("profit_impact", {})
    ho  = agent == "ho"
    ch  = pi.get("cost_history", [])
    if not ch:
        return
    df = pd.DataFrame(ch)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["retail"],
        name="Retail Price", line=dict(color="#f5a623", width=2),
        fill="tozeroy", fillcolor="rgba(245,166,35,.06)",
        hovertemplate="Retail: $%{y:.4f}<extra></extra>" if ho else "Retail: $%{y:.2f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cost"],
        name="Cost (excl. margin)", line=dict(color="#00d4ff", width=1.5, dash="dot"),
        hovertemplate="Cost: $%{y:.4f}<extra></extra>" if ho else "Cost: $%{y:.2f}<extra></extra>",
    ), secondary_y=False)
    fig.add_hline(y=pi.get("breakeven", 0),
                  line=dict(color="rgba(255,61,90,.5)", width=1, dash="dash"),
                  annotation_text=f"Breakeven ${pi.get('breakeven', 0):.4f}" if ho
                                  else f"Breakeven ${pi.get('breakeven', 0):.2f}",
                  annotation_font=dict(color="#ff3d5a", size=9))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["margin_pct"],
        name="Margin %", line=dict(color="#1df5a0", width=1.5),
        hovertemplate="Margin: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=300,
        title=dict(text="Cost, Retail & Margin Over Time (90-Day Rolling)", font=dict(size=11, color="#c8d8ec")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", tickformat="$.4f" if ho else "$.2f", secondary_y=False)
    fig.update_yaxes(title_text="Margin %", ticksuffix="%", secondary_y=True)
    return fig


def chart_scenario_cost(data, agent, sel_scenario):
    """Retail price by scenario."""
    pi = data.get("profit_impact", {})
    sr = pi.get("scenario_retail", {})
    ho = agent == "ho"
    if not sr:
        return
    names  = list(sr.keys())
    vals   = list(sr.values())
    colors = [SCEN_COLORS[i%5] if (not sel_scenario or sel_scenario == n) else SCEN_COLORS[i%5][:7]+"30"
              for i, n in enumerate(names)]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"${v:.4f}" if ho else f"${v:.2f}" for v in vals],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.4f}<extra></extra>" if ho else "%{x}: $%{y:.2f}<extra></extra>",
    ))
    be = pi.get("breakeven", 0)
    fig.add_hline(y=be + pi.get("margin", 0),
                  line=dict(color="rgba(255,61,90,.5)", width=1, dash="dash"),
                  annotation_text="Breakeven retail",
                  annotation_font=dict(color="#ff3d5a", size=9))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
        title=dict(text="Retail Price by Scenario", font=dict(size=10, color="#c8d8ec")),
        yaxis=dict(tickformat="$.4f" if ho else "$.2f"),
        showlegend=False, bargap=0.2,
    )
    return fig


def chart_ci_width(data, agent):
    """CI band width by horizon — shows uncertainty growth."""
    cib = data.get("ci_bands", {})
    ho  = agent == "ho"
    if not cib:
        return
    w95 = [round((cib.get(h,{}).get("ci95",[0,0])[1] - cib.get(h,{}).get("ci95",[0,0])[0]), 4) for h in HORIZONS]
    w80 = [round((cib.get(h,{}).get("ci80",[0,0])[1] - cib.get(h,{}).get("ci80",[0,0])[0]), 4) for h in HORIZONS]
    mids= [cib.get(h,{}).get("mid", 0) for h in HORIZONS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=HORIZONS, y=w95, name="95% CI Width",
                         marker_color="rgba(29,245,160,.65)", hovertemplate="%{x}: $%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(x=HORIZONS, y=w80, name="80% CI Width",
                         marker_color="rgba(0,212,255,.5)", hovertemplate="%{x}: $%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=HORIZONS, y=mids, name="Midpoint",
                             mode="lines+markers", line=dict(color="#ffd060", width=1.5, dash="dot"),
                             marker=dict(size=5)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
        title=dict(text="Forecast Uncertainty by Horizon (CI Width)", font=dict(size=10, color="#c8d8ec")),
        yaxis=dict(title="Width ($)", tickformat="$.4f" if ho else "$.2f"),
        barmode="overlay", bargap=0.2,
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


def chart_custom_bands(data):
    """HO custom risk bands heatmap."""
    cb = data.get("custom_bands", {})
    if not cb:
        return
    bands   = list(cb.get(HORIZONS[0], {}).keys())
    fig     = go.Figure()
    BCOLS   = ["#00d4ff", "#f5a623", "#ff3d5a", "#1df5a0", "#9d7aff"]
    for bi, band in enumerate(bands):
        vals = [round((cb.get(h,{}).get(band,0))*100, 2) for h in HORIZONS]
        fig.add_trace(go.Bar(
            x=HORIZONS, y=vals, name=band,
            marker_color=BCOLS[bi%5],
            hovertemplate=f"{band} — %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
        title=dict(text="HO Custom Risk Bands by Horizon", font=dict(size=10, color="#c8d8ec")),
        yaxis=dict(title="Probability (%)", ticksuffix="%"),
        barmode="group", bargap=0.15,
        legend=dict(font=dict(size=8)),
    )
    return fig


def render_prob_table(data, agent, sel_horizon, sel_bin):
    """Interactive sortable probability table."""
    rows = data.get("prob_table", {})
    if not rows:
        return

    all_bins = [r[0] for r in rows.get(HORIZONS[0], [])]
    table_data = {"Price Range": all_bins}
    for h in HORIZONS:
        table_data[h] = [f"{round(r[1]*100,2):.2f}%" for r in rows.get(h, [])]

    df = pd.DataFrame(table_data)
    if sel_bin:
        df["_sel"] = df["Price Range"] == sel_bin
        df = df.sort_values("_sel", ascending=False).drop(columns=["_sel"])

    # Style the dataframe
    def style_rows(row):
        styles = []
        for col in row.index:
            if row["Price Range"] == sel_bin:
                styles.append("background-color: rgba(0,212,255,.12); color: #00d4ff; font-weight: bold")
            elif col == sel_horizon:
                styles.append("background-color: rgba(26,37,64,.5); color: #c8d8ec")
            else:
                styles.append("color: #4a6080")
        return styles

    styled = df.style.apply(style_rows, axis=1)
    st.dataframe(styled, use_container_width=True, height=320, hide_index=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:16px 0 8px">
      <div style="font-size:18px;font-weight:800;color:#c8d8ec;font-family:'Syne',sans-serif">⚡ Energy Intel</div>
      <div style="font-size:9px;color:#4a6080;font-family:'JetBrains Mono',monospace;margin-top:3px;letter-spacing:.8px">COMMODITY PROBABILITY ENGINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    st.sidebar.markdown("**Run Agent**")
    col1, col2 = st.sidebar.columns(2)
    run_oil = col1.button("🛢️ Oil", use_container_width=True)
    run_ho  = col2.button("🔥 HO", use_container_width=True)

    if run_oil:
        with st.spinner("Fetching oil market data…"):
            try:
                result, log = run_oil_agent()
                st.session_state.result  = result
                st.session_state.agent   = "oil"
                st.session_state.log     = log
                st.session_state.sel_horizon  = "1M"
                st.session_state.sel_bin      = None
                st.session_state.sel_scenario = None
                st.session_state.sel_region   = None
                st.session_state.sel_driver   = None
            except Exception as e:
                st.error(f"Error: {e}")

    if run_ho:
        with st.spinner("Fetching heating oil data (1-year history)…"):
            try:
                result, log = run_ho_agent()
                st.session_state.result  = result
                st.session_state.agent   = "ho"
                st.session_state.log     = log
                st.session_state.sel_horizon  = "1M"
                st.session_state.sel_bin      = None
                st.session_state.sel_scenario = None
                st.session_state.sel_region   = None
                st.session_state.sel_driver   = None
            except Exception as e:
                st.error(f"Error: {e}")

    # Only show filters if data loaded
    result = st.session_state.result
    if result:
        st.sidebar.divider()
        st.sidebar.markdown("**Cross-Filters**")
        st.sidebar.caption("Selections here update all charts simultaneously")

        # Horizon
        h = st.sidebar.radio("Horizon", HORIZONS, index=HORIZONS.index(st.session_state.sel_horizon), horizontal=True)
        st.session_state.sel_horizon = h

        # Scenario
        sp = result.get("scenario_paths", {})
        scen_opts = ["(All)"] + list(sp.keys())
        scen_idx  = scen_opts.index(st.session_state.sel_scenario) if st.session_state.sel_scenario in scen_opts else 0
        sel_scen  = st.sidebar.selectbox("Scenario", scen_opts, index=scen_idx)
        st.session_state.sel_scenario = None if sel_scen == "(All)" else sel_scen

        # Region
        rp       = result.get("regional_prices", [])
        reg_opts = ["(All)"] + [r["region"] for r in rp]
        reg_idx  = reg_opts.index(st.session_state.sel_region) if st.session_state.sel_region in reg_opts else 0
        sel_reg  = st.sidebar.selectbox("Region", reg_opts, index=reg_idx)
        st.session_state.sel_region = None if sel_reg == "(All)" else sel_reg

        # Driver
        drivers  = result.get("drivers", [])
        drv_opts = ["(All)"] + [d["name"] for d in drivers]
        drv_idx  = drv_opts.index(st.session_state.sel_driver) if st.session_state.sel_driver in drv_opts else 0
        sel_drv  = st.sidebar.selectbox("Driver", drv_opts, index=drv_idx)
        st.session_state.sel_driver = None if sel_drv == "(All)" else sel_drv

        # Price bin
        rows   = result.get("prob_table", {}).get(h, [])
        bins   = ["(All)"] + [r[0] for r in rows]
        bin_idx = bins.index(st.session_state.sel_bin) if st.session_state.sel_bin in bins else 0
        sel_bin = st.sidebar.selectbox("Price Bin", bins, index=bin_idx)
        st.session_state.sel_bin = None if sel_bin == "(All)" else sel_bin

        if st.sidebar.button("🔄 Clear all filters", use_container_width=True):
            st.session_state.sel_horizon  = "1M"
            st.session_state.sel_bin      = None
            st.session_state.sel_scenario = None
            st.session_state.sel_region   = None
            st.session_state.sel_driver   = None
            st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="font-size:9px;color:#2a3850;font-family:'JetBrains Mono',monospace;line-height:1.8">
    📧 Contact for help:<br>
    <a href="mailto:lsaggioro@potonmail.com" style="color:#00d4ff;text-decoration:none">
    lsaggioro@potonmail.com</a>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN DASHBOARD ────────────────────────────────────────────────────────────
def render_dashboard():
    result   = st.session_state.result
    agent    = st.session_state.agent
    sel_h    = st.session_state.sel_horizon
    sel_bin  = st.session_state.sel_bin
    sel_scen = st.session_state.sel_scenario
    sel_reg  = st.session_state.sel_region
    sel_drv  = st.session_state.sel_driver

    if not result:
        st.markdown("""
        <div style="text-align:center;padding:80px 0">
          <div style="font-size:48px;margin-bottom:16px">⚡</div>
          <div style="font-size:22px;font-weight:800;color:#c8d8ec;font-family:'Syne',sans-serif;margin-bottom:8px">
            Energy Intelligence Dashboard
          </div>
          <div style="font-size:11px;color:#4a6080;font-family:'JetBrains Mono',monospace;letter-spacing:.8px">
            DETERMINISTIC COMMODITY PROBABILITY ENGINE
          </div>
          <div style="margin-top:32px;font-size:13px;color:#2a3850">
            ← Click <strong style="color:#f5a623">🛢️ Oil</strong> or <strong style="color:#00d4ff">🔥 HO</strong> in the sidebar to begin
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    ho = agent == "ho"
    md = result.get("market_data", {})
    f  = result.get("forecast", {})
    pi = result.get("profit_impact", {})
    ci = result.get("ci_bands", {})
    ci1m = ci.get("1M", {})

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    section("①", "KPI SNAPSHOT")
    if ho:
        cols = st.columns(6)
        metrics = [
            ("HO Price", f"${md.get('HO', 0):.4f}", "$/gal"),
            ("WTI", f"${md.get('WTI', 0):.2f}", "$/bbl"),
            ("Crack Spread", f"${md.get('crack_spread', 0):.2f}", "$/bbl"),
            ("VIX", f"{md.get('VIX', 0):.1f}", "volatility index"),
            ("Retail (est.)", f"${pi.get('retail', 0):.4f}", "incl. margin"),
            ("Margin %", f"{pi.get('margin_pct', 0):.1f}%", "distributor"),
        ]
    else:
        cols = st.columns(6)
        metrics = [
            ("WTI Live", f"${f.get('current_wti', 0):.2f}", "per barrel"),
            ("Brent", f"${result.get('brent', 0):.2f}", "per barrel"),
            ("1-Wk Forecast", f"${f.get('forecast_low',0)}–${f.get('forecast_high',0)}", "90% CI"),
            ("Ann. Vol", f"{f.get('annualised_vol', 0):.1f}%", "historical σ"),
            ("Retail equiv.", f"${pi.get('retail', 0):.2f}", "$/bbl"),
            ("Margin %", f"{pi.get('margin_pct', 0):.1f}%", "refinery"),
        ]
    for col, (label, val, sub) in zip(cols, metrics):
        col.metric(label, val, sub)

    # Δ from CI
    ci_width = round((ci1m.get("ci95",[0,0])[1] - ci1m.get("ci95",[0,0])[0]), 4 if ho else 2)
    st.caption(f"1M 95% CI range: **${ci_width}** · Regime: **{result.get('regime','—')}**" if ho
               else f"1M 95% CI range: **${ci_width}** · Direction: **{f.get('direction','—')}**")

    # ── ② PRICE + CI ──────────────────────────────────────────────────────────
    section("②", "PRICE HISTORY + CONFIDENCE INTERVAL FORECAST", "Scroll/pinch to zoom · Drag to pan")
    chart_price_ci(result, agent)

    # ── ③ PROBABILITY ─────────────────────────────────────────────────────────
    section("③", "PROBABILITY DISTRIBUTION", "Select horizon in sidebar · Click chart bars to cross-filter")
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_prob_distribution(result, agent, sel_h, sel_bin)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_2")
    with c2:
        fig = chart_cdf(result, sel_h)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_3")

    # Prob table
    st.markdown("**Probability Table** — all horizons")
    render_prob_table(result, agent, sel_h, sel_bin)

    # ── ④ VOL HEATMAP ─────────────────────────────────────────────────────────
    section("④", "VOLATILITY HEATMAP", "Color intensity = market instability")
    c1, c2 = st.columns([3, 1])
    with c1:
        fig = chart_vol_heatmap(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_4")
    with c2:
        fig = chart_vol_histogram(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_5")

    # ── ⑤ DRIVER ANALYSIS ────────────────────────────────────────────────────
    section("⑤", "DRIVER ANALYSIS / EXPLAINABILITY", "Select driver in sidebar to isolate · SHAP-style contribution")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = chart_drivers(result, agent, sel_drv)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_6")
    with c2:
        fig = chart_driver_donut(result, sel_drv)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_7")

    # ── ⑥ SCENARIO SIMULATION ────────────────────────────────────────────────
    section("⑥", "SCENARIO SIMULATION", "Select scenario in sidebar · All panels update together")
    fig = chart_scenarios(result, agent, sel_scen)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_8")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = chart_scenario_weights(result, sel_scen)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_9")
    with c2:
        fig = chart_scenario_final(result, agent, sel_scen)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_10")
    with c3:
        fig = chart_ci_width(result, agent)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_11")

    # ── ⑦ REGIONAL MAP ───────────────────────────────────────────────────────
    section("⑦", "REGIONAL PRICE MAP", "Select region in sidebar to cross-filter · Green=below avg · Red=above avg")
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_region_map(result, agent, sel_reg)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_12")
    with c2:
        fig = chart_region_bar(result, agent, sel_reg)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_13")

    # ── ⑧ PROFIT / COST ──────────────────────────────────────────────────────
    section("⑧", "PROFIT / COST IMPACT DASHBOARD", "Translates prices into business P&L · Select scenario to cross-filter")
    fig = chart_profit_timeline(result, agent)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_14")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = chart_scenario_cost(result, agent, sel_scen)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_15")
    with c2:
        if ho:
            fig = chart_custom_bands(result)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_16")
        else:
            fig = chart_ci_width(result, agent)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_17")
    with c3:
        # Margin % mini-chart from cost history
        pi_ch = pi.get("cost_history", [])
        if pi_ch:
            df_m = pd.DataFrame(pi_ch)
            fig_m = go.Figure(go.Scatter(
                x=df_m["date"], y=df_m["margin_pct"],
                mode="lines", line=dict(color="#1df5a0", width=1.5),
                fill="tozeroy", fillcolor="rgba(29,245,160,.07)",
                hovertemplate="Margin: %{y:.1f}%<extra></extra>",
            ))
            fig_m.update_layout(
                template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07090f",
        plot_bgcolor="#07090f", height=240,
                title=dict(text="Rolling Margin % History", font=dict(size=10, color="#c8d8ec")),
                yaxis=dict(ticksuffix="%"), showlegend=False)
            st.plotly_chart(fig_m, use_container_width=True, config={"displayModeBar": False, "displaylogo": False}, key="chart_18")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    section("📄", "MARKET SUMMARY")
    with st.expander("View full summary", expanded=False):
        st.code(result.get("summary", ""), language=None)

    # ── LOG ──────────────────────────────────────────────────────────────────
    with st.expander("🔧 Run log", expanded=False):
        st.markdown(
            '<div class="status-box">' + "\n".join(st.session_state.log or []) + "</div>",
            unsafe_allow_html=True
        )


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    render_sidebar()
    render_dashboard()

if __name__ == "__main__":
    main()
