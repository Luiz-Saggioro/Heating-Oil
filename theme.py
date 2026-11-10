"""
theme.py — Single source of truth for dashboard colours, fonts and the Plotly template.

Two palettes (light / dark) share the same semantic keys, so UI code never hard-codes a
colour: HTML uses CSS variables (``var(--ink)``), Plotly uses ``C["ink"]`` / ``rgba()``.
Text colours meet WCAG AA (>= 4.5:1) against their backgrounds in both palettes.

Usage (streamlit_app.py):
    C  = theme.palette(is_dark)
    st.markdown(theme.css(C), unsafe_allow_html=True)
    PT = theme.register_plotly(C)        # template name for fig.update_layout(template=PT)
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_UI = "'Syne',sans-serif"
FONT_MONO = "'JetBrains Mono',monospace"
FONT_PLOT = "JetBrains Mono, monospace"
FONT_PLOT_SIZE = 12       # base chart font (axis ticks, legend)
FONT_TITLE_SIZE = 14      # chart titles

_PALETTES: dict[str, dict[str, str]] = {
    "light": {
        # surfaces
        "app_bg": "#f0f4f8", "panel": "#ffffff", "plot_bg": "#f5f8fc",
        "th_bg": "#eaeff6", "sel_bg": "#e2edf8",
        # text (darkened vs v4.0 for contrast)
        "ink": "#1b2a3b", "muted": "#3f586f", "faint": "#5a7189",
        # lines
        "border": "#c4d0de", "grid": "#d8e2ee",
        # sidebar (always dark)
        "side_bg": "#1b2a3b", "side_border": "#26394d", "side_ink": "#e8f0f8",
        "side_label": "#b9cee2",
        # accents / series
        "blue": "#1758b0", "amber": "#b87010", "amber_dk": "#966010", "green": "#1a7a45",
        "red": "#b82828", "purple": "#5438a0", "gold": "#987010", "rust": "#b05828",
        "cyan": "#0089a8",
        # heatmap scales (low -> high)
        "heat0": "#eaf5ee", "heat1": "#b9dcc5", "heat2": "#e8c77f", "heat3": "#d98a4a",
        "seq0": "#f8fafc", "seq1": "#c8ddf4", "seq2": "#5888c8",
    },
    "dark": {
        "app_bg": "#0e1621", "panel": "#16212e", "plot_bg": "#121c28",
        "th_bg": "#1d2a39", "sel_bg": "#1f3550",
        "ink": "#e8eef5", "muted": "#b6c6d6", "faint": "#93a8bd",
        "border": "#2f4256", "grid": "#233244",
        "side_bg": "#0a111a", "side_border": "#1d2a39", "side_ink": "#e8f0f8",
        "side_label": "#b9cee2",
        "blue": "#5b9cf2", "amber": "#e3a03c", "amber_dk": "#c4852a", "green": "#3dbb78",
        "red": "#ec6a6a", "purple": "#a18cf0", "gold": "#d8b545", "rust": "#e88a58",
        "cyan": "#38c8e8",
        "heat0": "#15291f", "heat1": "#1f5a3a", "heat2": "#a8862e", "heat3": "#c0622e",
        "seq0": "#132033", "seq1": "#1f4a7a", "seq2": "#3f7fd0",
    },
}


def palette(dark: bool) -> dict[str, str]:
    """Return the colour palette for the requested mode (a copy, safe to mutate)."""
    return dict(_PALETTES["dark" if dark else "light"])


def rgba(c: dict[str, str], key: str, alpha: float) -> str:
    """Palette colour ``key`` as an ``rgba()`` string with the given alpha."""
    h = c[key].lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def register_plotly(c: dict[str, str]) -> str:
    """Register the Plotly template for palette ``c`` and return its name.

    Registered per mode (``energy_light`` / ``energy_dark``); the global default is NOT
    changed because pio.templates is shared by every user session on the server.
    """
    dark = c["app_bg"] == _PALETTES["dark"]["app_bg"]
    name = "energy_dark" if dark else "energy_light"
    axis = dict(gridcolor=c["grid"], zerolinecolor=c["border"], linecolor=c["border"],
                tickfont=dict(color=c["muted"], size=FONT_PLOT_SIZE),
                title=dict(font=dict(color=c["ink"], size=FONT_PLOT_SIZE + 1), standoff=8))
    pio.templates[name] = go.layout.Template(layout=go.Layout(
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(family=FONT_PLOT, color=c["ink"], size=FONT_PLOT_SIZE),
        title=dict(font=dict(color=c["ink"], size=FONT_TITLE_SIZE)),
        colorway=[c["blue"], c["amber"], c["green"], c["red"], c["purple"], c["gold"]],
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor=rgba(c, "panel", 0.96), bordercolor=c["border"], borderwidth=1,
                    font=dict(color=c["ink"], size=FONT_PLOT_SIZE)),
        hoverlabel=dict(bgcolor=c["panel"], bordercolor=c["border"],
                        font=dict(color=c["ink"], family=FONT_PLOT, size=FONT_PLOT_SIZE)),
        coloraxis=dict(colorbar=dict(tickfont=dict(color=c["muted"]))),
        geo=dict(bgcolor=c["plot_bg"], landcolor=c["th_bg"], oceancolor=c["panel"],
                 subunitcolor=c["border"], countrycolor=c["border"]),
        margin=dict(l=55, r=20, t=50, b=45),
    ))
    return f"plotly+{name}"


def css(c: dict[str, str]) -> str:
    """Global stylesheet. Exposes the palette as CSS variables and forces readable text
    colours on Streamlit's native widgets so they follow the app theme, not the browser's."""
    vars_ = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in c.items())
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@600;700;800&display=swap');
:root{{{vars_}}}
html,body,[class*="css"]{{font-family:{FONT_UI};}}
code,.stCode,pre{{font-family:{FONT_MONO}!important;}}
.stApp,header[data-testid="stHeader"]{{background:var(--app-bg)!important;color:var(--ink);}}
.stPlotlyChart,.stPlotlyChart>div,div[data-testid="stPlotlyChart"],div[data-testid="block-container"],
div[data-testid="stVerticalBlock"],div[data-testid="column"],div[data-testid="stHorizontalBlock"],
.element-container{{background:transparent!important;}}
/* main-area text (scoped to stMain so the dark sidebar keeps its own colours) */
[data-testid="stMain"] [data-testid="stMarkdownContainer"],[data-testid="stMain"] [data-testid="stMarkdownContainer"] *,
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,[data-testid="stMain"] li{{color:var(--ink);}}
[data-testid="stMain"] [data-testid="stCaptionContainer"],[data-testid="stMain"] [data-testid="stCaptionContainer"] *{{color:var(--muted)!important;font-size:13px;}}
h1,h2,h3,h4{{font-family:{FONT_UI}!important;color:var(--ink)!important;}}
/* sidebar (dark in both modes) */
section[data-testid="stSidebar"]{{background:var(--side-bg);border-right:1px solid var(--side-border);}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *{{color:var(--side-ink);}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{{color:var(--side-label)!important;}}
/* metrics */
div[data-testid="stMetric"]{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,0.06);}}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] label *{{color:var(--muted)!important;font-family:{FONT_MONO}!important;font-size:12px!important;text-transform:uppercase;letter-spacing:.8px;}}
div[data-testid="stMetricValue"],div[data-testid="stMetricValue"] *{{color:var(--ink)!important;font-family:{FONT_MONO}!important;font-size:22px!important;font-weight:700!important;}}
/* tabs: large, high-contrast category labels; smaller second row for measures */
[data-testid="stTabs"] [role="tablist"]{{gap:2px;flex-wrap:wrap;}}
[data-testid="stTab"]{{padding:8px 10px;}}
[data-testid="stTab"] p{{color:var(--muted)!important;font-weight:700!important;font-size:15px!important;}}
[data-testid="stTab"][aria-selected="true"] p{{color:var(--ink)!important;}}
[data-testid="stTab"] .react-aria-SelectionIndicator{{background:var(--amber)!important;height:3px;}}
[data-testid="stTabs"] [data-testid="stTabs"] [data-testid="stTab"] p{{font-size:14px!important;font-weight:600!important;}}
[data-testid="stTabs"] [data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator{{background:var(--blue)!important;}}
/* alerts: theme surface + coloured left rule, so text never sits on a mismatched background */
[data-testid="stAlertContainer"]{{background:var(--th-bg)!important;border-left:4px solid var(--blue);}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){{border-left-color:var(--red);}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){{border-left-color:var(--amber);}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){{border-left-color:var(--green);}}
[data-testid="stAlertContainer"] *{{color:var(--ink)!important;}}
/* inputs, dropdowns, expanders */
[data-baseweb="select"]>div,.stNumberInput input,.stTextInput input,.stDateInput input{{background:var(--panel)!important;border-color:var(--border)!important;color:var(--ink)!important;}}
[data-baseweb="select"] *{{color:var(--ink)!important;}}
div[data-baseweb="popover"] li,div[data-baseweb="popover"] ul{{background:var(--panel)!important;color:var(--ink)!important;}}
div[data-testid="stExpander"] details{{background:var(--panel);border-color:var(--border);}}
div[data-testid="stExpander"] summary *{{color:var(--ink)!important;font-weight:600;}}
.stButton button{{background:linear-gradient(90deg,var(--amber),var(--amber-dk))!important;border:none!important;border-radius:6px;font-family:{FONT_MONO};font-weight:700;letter-spacing:.5px;transition:opacity .15s;}}
.stButton button *{{color:#fff!important;}}
.stButton button:hover{{opacity:.85;}}
hr{{border-color:var(--border);}}
.status-box{{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px;font-family:{FONT_MONO};font-size:12px;color:var(--muted);line-height:1.8;white-space:pre;overflow-x:auto;}}
.js-plotly-plot{{border-radius:8px;}}
</style>"""
