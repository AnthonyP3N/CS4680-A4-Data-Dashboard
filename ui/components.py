"""Small reusable building blocks. Views compose these and never write raw HTML themselves.

Every helper that takes text escapes it, so model output and user data can't
inject markup into the page.
"""

from __future__ import annotations

import html
from datetime import date, datetime
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st


def visitor_timezone() -> ZoneInfo | None:
    """The time zone of the person's browser, when Streamlit can tell us. None means "use the server's"."""
    try:
        name = st.context.timezone
        return ZoneInfo(name) if name else None
    except Exception:  # older Streamlit, no browser context (tests), or an unknown zone name
        return None


def local_day(timestamp: str, zone: ZoneInfo | None = None) -> date | None:
    """The calendar day of a stored UTC timestamp, as the person would see it."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return None
    return moment.astimezone(zone).date() if zone else moment.astimezone().date()


MAIN_TABS = "main_tab"


def go_to_tab(label: str) -> None:
    """Switch the main tabs. Call from a button's on_click, before the tabs are drawn."""
    st.session_state[MAIN_TABS] = label


def esc(text: object) -> str:
    return html.escape("" if text is None else str(text))


def format_value(value: object) -> str:
    """Human-friendly number for the big-metric display."""
    if isinstance(value, bool):
        return esc(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"
    if isinstance(value, int):
        return f"{value:,}"
    return esc(value)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# -- text blocks -------------------------------------------------------------- #
def wordmark(tagline: str) -> None:
    _html(f'<div class="wordmark" role="heading" aria-level="1"><span>AskData</span></div>'
          f'<div class="tagline">{esc(tagline)}</div>')


def question_bubble(text: str) -> None:
    _html(f'<div class="q" role="note" aria-label="Your question"><div>{esc(text)}</div></div>')


def answer_text(text: str) -> None:
    _html(f'<div class="answer-text">{esc(text)}</div>')


def meta(text: str) -> None:
    _html(f'<p class="meta">{esc(text)}</p>')


def source_line(text: str) -> None:
    _html(f'<p class="source-line">{esc(text)}</p>')


def local_time(timestamp: str, zone: ZoneInfo | None = None) -> str:
    """ "Nov 10, 6:41 PM" in the reader's time zone, or "" if the timestamp can't be read."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return ""
    moment = moment.astimezone(zone) if zone else moment.astimezone()
    hour = moment.strftime("%I").lstrip("0") or "12"
    return f"{moment.strftime('%b')} {moment.day}, {hour}:{moment.strftime('%M %p')}"


def conversation_title(title: str, subtitle: str = "") -> None:
    _html(f'<div class="conv-title" role="heading" aria-level="2">{esc(title)}</div>'
          + (f'<p class="small">{esc(subtitle)}</p>' if subtitle else ""))


def detail_list(pairs: list[tuple[str, str]]) -> None:
    """A compact two-column list of facts about a turn."""
    rows = "".join(f"<dt>{esc(label)}</dt><dd>{esc(value)}</dd>" for label, value in pairs)
    _html(f'<dl class="details">{rows}</dl>')


def small(text: str) -> None:
    _html(f'<p class="small">{esc(text)}</p>')


def notice(text: str, kind: str = "", lead: str = "", trusted_html: str = "") -> None:
    """A bordered message. kind: "" (neutral), "ask", "err" or "done".

    `trusted_html` is appended unescaped and must only ever be a literal written
    in this codebase, never model output or user data.
    """
    lead_html = f"<b>{esc(lead)}</b> " if lead else ""
    role = "alert" if kind == "err" else "status"
    _html(f'<div class="notice {esc(kind)}" role="{role}">{lead_html}{esc(text)}{trusted_html}</div>')


def empty_state(title: str, body: str) -> None:
    _html(f'<div class="empty"><h2>{esc(title)}</h2><p>{esc(body)}</p></div>')


def metric(value: object, label: str) -> None:
    _html(f'<div class="metric" role="group" aria-label="{esc(label)}: {format_value(value)}">'
          f'<div class="value" aria-hidden="true">{format_value(value)}</div>'
          f'<div class="label" aria-hidden="true">{esc(label)}</div></div>')


def confirm_panel(title: str, paragraphs: Iterable[str]) -> None:
    """Warning-coloured panel for a pending data change. Paragraphs may contain <b> written by the caller."""
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    _html(f'<div class="confirm"><h4>{esc(title)}</h4>{body}</div>')


def step_trail(steps: list[tuple[str, str, str]], seconds: float | None = None) -> None:
    """A row of pills showing what the agent did. Each step is (style, icon, label)."""
    parts = []
    for i, (style, icon, label) in enumerate(steps):
        if i:
            parts.append('<span class="sep" aria-hidden="true">›</span>')
        parts.append(f'<span class="step {esc(style)}" role="listitem"><i aria-hidden="true">{esc(icon)}</i>'
                     f'{esc(label)}</span>')
    timing = f'<span class="sep" aria-hidden="true"> </span><span>{seconds:.1f}s</span>' if seconds else ""
    _html(f'<div class="trail" role="list" aria-label="What AskData did">{"".join(parts)}{timing}</div>')


def confirm_button(key: str, label: str, warning: str, confirm_label: str) -> bool:
    """A two-step button for anything that can't be undone.

    The first click only shows `warning` with a confirm and a keep button.
    Returns True on the run in which the user confirms.
    """
    flag = f"_confirm_{key}"
    if not st.session_state.get(flag):
        if st.button(label, key=f"ask_{key}"):
            st.session_state[flag] = True
            st.rerun()
        return False
    st.warning(warning)
    if st.button("Keep", key=f"no_{key}"):
        st.session_state[flag] = False
        st.rerun()
    if st.button(confirm_label, key=f"yes_{key}", type="primary"):
        st.session_state[flag] = False
        return True
    return False


# -- data --------------------------------------------------------------------- #
def dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs: dict = {"hide_index": True}
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:  # older Streamlit
        st.dataframe(df, use_container_width=True, **kwargs)


def table_height(rows: int, cap: int) -> int:
    return min(38 * (rows + 1), cap)


def plotly(fig, key: str) -> None:
    """Draw a chart. `key` must be unique on the page: the same chart can appear in two places."""
    try:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=key)
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, key=key)


def panel_heading(text: str) -> None:
    _html(f'<div class="panel-heading" role="heading" aria-level="3">{esc(text)}</div>')


def panel_value(text: str, status: str = "") -> None:
    """A line in the context panel. status: "ok", "warn" or "off" adds a coloured dot (with its meaning in text)."""
    dot = f'<span class="status-dot {esc(status)}" aria-hidden="true"></span>' if status else ""
    _html(f'<p class="panel-value">{dot}{esc(text)}</p>')
