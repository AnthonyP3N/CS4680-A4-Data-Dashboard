"""AskData's design tokens: the one place colours, type, spacing and shape are defined.

Everything visual is generated from the values below:

- `.streamlit/config.toml`  Streamlit's own light and dark themes (native widgets,
                             charts, fonts), written by `write_streamlit_config()`
- `docs/design-tokens.json` the same tokens in the W3C design-tokens format, which
                             Figma imports through the Tokens Studio plugin
- the stylesheet            CSS variables for AskData's own components, built for
                             the theme the viewer is using

Regenerate both files after changing a token:   python -m ui.theme
Tests fail if either file has drifted from the values here, and check that
every text colour pair meets WCAG AA contrast in both themes.

Identity: one signature, the highlighter stroke under the wordmark and key
numbers (`highlight`). Everything around it stays quiet: one indigo accent for
interaction, a calm grey ink scale, and colour otherwise reserved for status.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
TOKENS_PATH = ROOT / "docs" / "design-tokens.json"

# --------------------------------------------------------------------------- #
# Colour, per theme. Names describe the role, never the hue.
# --------------------------------------------------------------------------- #
PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "canvas": "#F7F8FA",          # page background
        "surface": "#FFFFFF",         # cards, notices, panels
        "surface-sunken": "#EDEFF4",  # sidebar, inputs, code
        "ink": "#1B2333",             # primary text
        "ink-muted": "#596273",       # secondary text
        "ink-faint": "#7A8294",       # separators, placeholders (decorative)
        "line": "#DCE0E8",            # borders and rules
        "chip": "#E9ECF4",            # step-trail pills
        "accent": "#3B4CCA",          # links, focus, primary buttons, "asking" notices
        "on-accent": "#FFFFFF",       # text on accent and on status dots
        "highlight": "#FFD84D",       # the highlighter stroke: AskData's signature
        "success": "#0B7A6E",
        "attention": "#9A5B00",       # self-corrected
        "danger": "#B93A0B",
        "danger-surface": "#FFF4EC",
        "bubble": "#1B2333",          # the person's question
        "on-bubble": "#FFFFFF",
    },
    "dark": {
        "canvas": "#10141C",
        "surface": "#171C26",
        "surface-sunken": "#1F2532",
        "ink": "#E7EAF1",
        "ink-muted": "#A6AEC0",
        "ink-faint": "#7C8598",
        "line": "#2E3545",
        "chip": "#252C3A",
        "accent": "#8E9CFF",
        "on-accent": "#10141C",
        "highlight": "#8A6A00",       # a darker stroke so light text on top stays readable
        "success": "#3CC8B4",
        "attention": "#E5A445",
        "danger": "#FF8A5B",
        "danger-surface": "#2B1C16",
        "bubble": "#E7EAF1",
        "on-bubble": "#10141C",
    },
}

# Chart series colours, in order. Each set is distinguishable on its own theme's canvas.
CHART_PALETTES: dict[str, list[str]] = {
    "light": ["#3B4CCA", "#0E9F8E", "#D98200", "#D64545", "#7C5CD6", "#2E8B57", "#C2418F", "#596273"],
    "dark": ["#8E9CFF", "#3CC8B4", "#F2B441", "#FF7B7B", "#B39BFF", "#5FD08A", "#F07CC0", "#A6AEC0"],
}

# --------------------------------------------------------------------------- #
# Type, space, shape and depth (shared by both themes)
# --------------------------------------------------------------------------- #
FONTS = {
    "sans": "Figtree",
    "mono": "JetBrains Mono",
}
FONT_URLS = {
    "sans": "https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap",
    "mono": "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap",
}
# A 1.2 (minor third) scale on a 1rem base, after the Elements of Typographic Style
TYPE_SCALE = {
    "xs": "0.833rem", "sm": "0.917rem", "md": "1rem", "lg": "1.2rem",
    "xl": "1.44rem", "2xl": "1.728rem", "3xl": "2.074rem", "display": "2.986rem",
}
LINE_HEIGHT = {"tight": "1.15", "body": "1.55"}
WEIGHT = {"regular": "400", "medium": "500", "semibold": "600", "bold": "700", "heavy": "800"}
MEASURE = "72ch"                      # longest comfortable line of running text
SPACE = {"1": "0.25rem", "2": "0.5rem", "3": "0.75rem", "4": "1rem", "5": "1.5rem", "6": "2rem", "7": "3rem"}
RADIUS = {"sm": "0.5rem", "md": "0.625rem", "lg": "1rem", "pill": "999px"}
ELEVATION = {
    "flat": "none",
    "raised": "0 1px 2px rgba(16, 20, 28, 0.06)",
    "overlay": "0 8px 24px rgba(16, 20, 28, 0.14)",
}

THEMES = tuple(PALETTES)


def font_stack(kind: str) -> str:
    fallback = "ui-monospace, monospace" if kind == "mono" else "system-ui, sans-serif"
    return f"'{FONTS[kind]}', {fallback}"


# --------------------------------------------------------------------------- #
# Contrast (WCAG 2.x), used by the tests
# --------------------------------------------------------------------------- #
def _luminance(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


# Text that must be readable (4.5:1) and marks that must be visible (3:1), as (foreground, background, minimum)
CONTRAST_PAIRS = [
    ("ink", "canvas", 4.5), ("ink", "surface", 4.5), ("ink", "surface-sunken", 4.5), ("ink", "chip", 4.5),
    ("ink-muted", "canvas", 4.5), ("ink-muted", "surface", 4.5), ("ink-muted", "surface-sunken", 4.5),
    ("accent", "canvas", 4.5), ("accent", "surface", 4.5), ("on-accent", "accent", 4.5),
    ("on-bubble", "bubble", 4.5), ("ink", "danger-surface", 4.5), ("danger", "danger-surface", 4.5),
    ("danger", "surface", 4.5), ("ink", "highlight", 3.0),
    ("success", "canvas", 3.0), ("attention", "canvas", 3.0), ("danger", "canvas", 3.0),
    ("ink-faint", "canvas", 3.0), ("on-accent", "success", 3.0), ("on-accent", "danger", 3.0),
]


# --------------------------------------------------------------------------- #
# Generated: Streamlit's theme
# --------------------------------------------------------------------------- #
def streamlit_theme() -> dict[str, dict[str, object]]:
    """The [theme...] sections of .streamlit/config.toml."""
    shared = {
        "font": f"{FONTS['sans']}:{FONT_URLS['sans']}, sans-serif",
        "codeFont": f"{FONTS['mono']}:{FONT_URLS['mono']}, monospace",
        "baseRadius": RADIUS["md"],
        "buttonRadius": RADIUS["sm"],
        "showWidgetBorder": True,
    }
    sections: dict[str, dict[str, object]] = {"theme": shared}
    for name, p in PALETTES.items():
        sections[f"theme.{name}"] = {
            "primaryColor": p["accent"], "backgroundColor": p["canvas"],
            "secondaryBackgroundColor": p["surface-sunken"], "textColor": p["ink"], "linkColor": p["accent"],
            "borderColor": p["line"], "dataframeBorderColor": p["line"],
            "codeBackgroundColor": p["surface-sunken"], "chartCategoricalColors": CHART_PALETTES[name],
        }
    return sections


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return json.dumps(str(value))


def streamlit_config_text() -> str:
    """The whole .streamlit/config.toml."""
    lines = ["# Generated by `python -m ui.theme` from ui/theme.py. Edit the tokens there, not this file.", ""]
    for section, values in streamlit_theme().items():
        lines.append(f"[{section}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        lines.append("")
    lines += ["[server]", "maxUploadSize = 20", "", "[browser]", "gatherUsageStats = false", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generated: design tokens for Figma (W3C Design Tokens format)
# --------------------------------------------------------------------------- #
def design_tokens() -> dict:
    """Tokens in the W3C draft format: {"group": {"name": {"$type": ..., "$value": ...}}}.

    Import into Figma with the Tokens Studio plugin (one set per theme plus "core"),
    then apply the "light" or "dark" set to switch a frame's theme.
    """
    def group(kind: str, values: dict[str, str]) -> dict:
        return {name: {"$type": kind, "$value": value} for name, value in values.items()}

    tokens = {
        "core": {
            "font-family": group("fontFamily", {k: font_stack(k) for k in FONTS}),
            "font-size": group("dimension", TYPE_SCALE),
            "font-weight": group("fontWeight", WEIGHT),
            "line-height": group("number", LINE_HEIGHT),
            "space": group("dimension", SPACE),
            "radius": group("dimension", RADIUS),
            "shadow": group("shadow", ELEVATION),
            "measure": {"$type": "dimension", "$value": MEASURE},
        },
    }
    for name, palette in PALETTES.items():
        tokens[name] = {
            "color": group("color", palette),
            "chart": group("color", {str(i + 1): c for i, c in enumerate(CHART_PALETTES[name])}),
        }
    return tokens


def design_tokens_text() -> str:
    return json.dumps(design_tokens(), indent=2) + "\n"


def write_generated_files() -> list[Path]:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(streamlit_config_text())
    TOKENS_PATH.write_text(design_tokens_text())
    return [CONFIG_PATH, TOKENS_PATH]


# --------------------------------------------------------------------------- #
# The stylesheet for AskData's own components
# --------------------------------------------------------------------------- #
_RULES = """
.block-container{ padding-top:2rem; max-width:1320px; }
code, pre { font-family:var(--font-mono) !important; }
/* Primary buttons: Streamlit uses white text on any primary colour; the dark theme's accent is light */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p,
[data-testid="stBaseButton-primary"] span { color:var(--on-accent) !important; }
/* Conversation list: left-aligned rows; the open one is marked, not greyed out */
[class*="st-key-conv_"] button{ justify-content:flex-start; text-align:left; }
.st-key-conv_open button:disabled{ background:var(--chip); color:var(--ink); border-color:var(--accent);
  font-weight:var(--weight-semibold); opacity:1; cursor:default; }

/* Wordmark: the highlighter stroke is the app's one signature */
.wordmark{ font-weight:var(--weight-heavy); font-size:var(--size-2xl); letter-spacing:-0.02em; color:var(--ink);
  line-height:var(--leading-tight); }
.wordmark span{ background:linear-gradient(transparent 58%, var(--highlight) 58%, var(--highlight) 92%, transparent 92%);
  padding:0 .1em; }
.tagline{ color:var(--ink-muted); font-size:var(--size-sm); margin:var(--space-2) 0 var(--space-4); max-width:34ch; }

/* Conversation header */
.conv-title{ font-weight:var(--weight-bold); font-size:var(--size-xl); letter-spacing:-0.01em; color:var(--ink);
  line-height:var(--leading-tight); }

/* The person's question */
.q{ display:flex; justify-content:flex-end; margin:var(--space-6) 0 var(--space-2); }
.q div{ background:var(--bubble); color:var(--on-bubble); padding:var(--space-2) var(--space-4);
  border-radius:var(--radius-lg) var(--radius-lg) 4px var(--radius-lg); max-width:min(70ch, 92%); font-weight:var(--weight-medium); }

/* Answers */
.answer-text{ font-size:var(--size-lg); line-height:var(--leading-body); color:var(--ink); max-width:var(--measure);
  margin:0 0 var(--space-2); }
.meta{ color:var(--ink-muted); font-size:var(--size-xs); margin:0 0 var(--space-3); }
.metric{ margin:var(--space-2) 0 var(--space-4); }
.metric .value{ font-size:var(--size-display); font-weight:var(--weight-heavy); letter-spacing:-0.03em; color:var(--ink);
  line-height:1; display:inline-block;
  background:linear-gradient(transparent 62%, var(--highlight) 62%, var(--highlight) 90%, transparent 90%); }
.metric .label{ color:var(--ink-muted); margin-top:var(--space-2); font-size:var(--size-sm); }
.source-line{ color:var(--ink-muted); font-size:var(--size-xs); margin:var(--space-2) 0 0; }

/* Step trail: what the agent did, in order */
.trail{ display:flex; flex-wrap:wrap; gap:var(--space-1) var(--space-2); align-items:center; margin:var(--space-1) 0 var(--space-3);
  font-size:var(--size-xs); color:var(--ink-muted); list-style:none; padding:0; }
.step{ display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .55rem; border-radius:var(--radius-pill);
  background:var(--chip); color:var(--ink); }
.step i{ font-style:normal; width:1.05rem; height:1.05rem; border-radius:50%; display:inline-grid; place-items:center;
  font-size:.68rem; font-weight:var(--weight-bold); background:var(--success); color:var(--on-accent); }
.step.fixed i{ background:var(--attention); }
.step.stop i{ background:var(--danger); }
.step.wait i{ background:var(--accent); }
.sep{ color:var(--ink-faint); }

/* Notices */
.notice{ border-left:3px solid var(--ink-muted); background:var(--surface); padding:var(--space-3) var(--space-4);
  border-radius:0 var(--radius-sm) var(--radius-sm) 0; margin:var(--space-1) 0 var(--space-3); max-width:80ch; color:var(--ink);
  box-shadow:var(--shadow-raised); }
.notice.ask{ border-color:var(--accent); }
.notice.err{ border-color:var(--danger); }
.notice.done{ border-color:var(--success); }
.confirm{ border:1.5px solid var(--danger); background:var(--danger-surface); border-radius:var(--radius-md);
  padding:var(--space-4) var(--space-4) var(--space-1); margin:var(--space-2) 0 var(--space-4); }
.confirm h4{ margin:0 0 var(--space-1); color:var(--danger); font-size:var(--size-md); }
.confirm p{ margin:.1rem 0 var(--space-2); color:var(--ink); }

/* Small print, empty states, details */
.small{ color:var(--ink-muted); font-size:var(--size-xs); }
.empty h2{ font-weight:var(--weight-bold); letter-spacing:-0.01em; margin-bottom:var(--space-1); color:var(--ink); }
.empty p{ color:var(--ink-muted); max-width:60ch; }
.details{ display:grid; grid-template-columns:max-content 1fr; gap:.15rem var(--space-4); margin:var(--space-2) 0;
  font-size:var(--size-xs); }
.details dt{ color:var(--ink-muted); }
.details dd{ margin:0; color:var(--ink); overflow-wrap:anywhere; }

/* Context panel */
.panel-heading{ font-size:var(--size-xs); font-weight:var(--weight-semibold); color:var(--ink-muted);
  margin:var(--space-4) 0 var(--space-1); }
.panel-heading:first-child{ margin-top:0; }
.panel-value{ font-size:var(--size-sm); color:var(--ink); margin:0 0 var(--space-1); overflow-wrap:anywhere; }
.status-dot{ display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.4rem; vertical-align:middle; }
.status-dot.ok{ background:var(--success); }
.status-dot.warn{ background:var(--attention); }
.status-dot.off{ background:var(--ink-faint); }

/* Sidebar buttons read as a list */
section[data-testid="stSidebar"] .stButton button{ text-align:left; justify-content:flex-start; }

/* Visible keyboard focus everywhere */
:focus-visible{ outline:2px solid var(--accent) !important; outline-offset:2px; }

/* Tablets, or a laptop with the sidebar open: the context panel moves below the chat rather than squeezing */
@media (max-width: 1180px){
  [data-testid="stHorizontalBlock"]:has(.st-key-context_panel){ flex-wrap:wrap; }
  [data-testid="stColumn"]:has(.st-key-context_panel){ flex:1 1 100% !important; min-width:100%; order:2; }
  [data-testid="stHorizontalBlock"]:has(.st-key-context_panel) > [data-testid="stColumn"]:first-child{
    flex:1 1 100% !important; min-width:100%; }
}
@media (max-width: 640px){
  .block-container{ padding-left:var(--space-4); padding-right:var(--space-4); }
  .answer-text{ font-size:var(--size-md); }
  .metric .value{ font-size:var(--size-3xl); }
}
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; scroll-behavior:auto !important; } }
"""


def css_variables(theme: str) -> str:
    palette = PALETTES[theme if theme in PALETTES else "light"]
    values = {**palette}
    values.update({f"size-{k}": v for k, v in TYPE_SCALE.items()})
    values.update({f"weight-{k}": v for k, v in WEIGHT.items()})
    values.update({f"leading-{k}": v for k, v in LINE_HEIGHT.items()})
    values.update({f"space-{k}": v for k, v in SPACE.items()})
    values.update({f"radius-{k}": v for k, v in RADIUS.items()})
    values.update({f"shadow-{k}": v for k, v in ELEVATION.items()})
    values.update({"measure": MEASURE, "font-sans": font_stack("sans"), "font-mono": font_stack("mono")})
    return " ".join(f"--{name}:{value};" for name, value in values.items())


def stylesheet(theme: str = "light") -> str:
    """The complete <style> block for a theme ("light" or "dark")."""
    css = f":root{{ {css_variables(theme)} }}\n{_RULES}"
    # Streamlit's markdown parser ends an HTML block at a blank line and reads
    # /* */ as emphasis, so comments and blank lines are removed before injecting.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return "<style>\n" + "\n".join(line for line in css.splitlines() if line.strip()) + "\n</style>"


def current_theme() -> str:
    """The theme the viewer is using ("light" or "dark"), as reported by their browser."""
    import streamlit as st
    try:
        kind = (st.context.theme or {}).get("type")
    except Exception:  # older Streamlit, or no browser (tests)
        kind = None
    return kind if kind in PALETTES else "light"


def inject() -> str:
    """Add the stylesheet for the viewer's theme to the page. Returns the theme used."""
    import streamlit as st
    theme = current_theme()
    st.markdown(stylesheet(theme), unsafe_allow_html=True)
    return theme


if __name__ == "__main__":
    for path in write_generated_files():
        print(f"Wrote {path.relative_to(ROOT)}")
