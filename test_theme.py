"""Tests for theme.py — palettes stay in sync and helpers produce valid output."""
import re

import theme


def test_palettes_share_keys():
    assert theme.palette(False).keys() == theme.palette(True).keys()


def test_palette_values_are_hex():
    for dark in (False, True):
        assert all(re.fullmatch(r"#[0-9a-f]{6}", v) for v in theme.palette(dark).values())


def _lum(h):
    ch = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    ch = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def _contrast(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_text_contrast_wcag_aa():
    for dark in (False, True):
        c = theme.palette(dark)
        for fg in ("ink", "muted", "faint"):
            for bg in ("app_bg", "panel", "plot_bg"):
                assert _contrast(c[fg], c[bg]) >= 4.5, (dark, fg, bg)
        assert _contrast(c["side_ink"], c["side_bg"]) >= 4.5


def test_rgba():
    assert theme.rgba({"x": "#1758b0"}, "x", 0.2) == "rgba(23,88,176,0.2)"


def test_register_plotly_names():
    assert theme.register_plotly(theme.palette(False)) == "plotly+energy_light"
    assert theme.register_plotly(theme.palette(True)) == "plotly+energy_dark"


def test_css_exposes_every_key_as_variable():
    c = theme.palette(True)
    out = theme.css(c)
    assert all(f"--{k.replace('_', '-')}:" in out for k in c)
