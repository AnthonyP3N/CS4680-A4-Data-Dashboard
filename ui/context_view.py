"""The context panel beside the chat: what the answer is based on, at a glance.

Shows the active data source and how fresh its structure is, the safety status,
the latest answer's tables, filters and timings, the business rules and metrics
the AI was given, any remembered context it used, and related conversations.
On narrow screens Streamlit stacks it below the chat.
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from agent import citations
from agent.conversation import ConversationError
from agent.explain import describe_query
from services import AppServices

from . import components as c


def time_ago(moment: datetime | None, now: datetime | None = None) -> str:
    if moment is None:
        return "not read yet"
    now = now or datetime.now(timezone.utc)
    seconds = max(0, int((now - moment).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = seconds // 86400
    return f"{days} day{'s' if days != 1 else ''} ago"


def _open_related(services: AppServices, conversation_id: str) -> None:
    try:
        services.open_conversation(conversation_id)
    except ConversationError as error:
        st.session_state._context_error = str(error)


def render(services: AppServices, allow_writes: bool) -> None:
    source = services.source
    info, semantics = source.info, source.semantics
    conversation = services.conversation
    latest = next((r for r in reversed(conversation.history) if r.sql), None) if conversation else None

    with st.container(border=True, key="context_panel"):
        error = st.session_state.pop("_context_error", None)
        if error:
            st.error(error)

        c.panel_heading("Data source")
        c.panel_value(info.name)
        try:
            tables = len(source.schema().tables)
        except Exception:  # a broken source must not take the panel down
            tables = 0
        c.small(f"{tables} table{'s' if tables != 1 else ''} · {info.dialect} · structure read "
                f"{time_ago(info.last_schema_refresh)}")
        if st.button("Refresh structure", key="refresh_structure",
                     help="Read the tables and columns again, for example after the data changed."):
            source.schema(refresh=True)
            services.audit.log("refresh_schema", source=info.name)
            st.rerun()

        c.panel_heading("Safety")
        if allow_writes and info.writable:
            c.panel_value("Changes allowed. Each one is previewed and needs your confirmation.", "warn")
        else:
            c.panel_value("Read-only. Questions can't change the data.", "ok")
        c.small(f"Each query: up to {info.limits.max_rows:,} rows and {info.limits.query_timeout_s:g} seconds.")
        if semantics.sensitive_columns:
            count = len(semantics.sensitive_columns)
            shown = {"partial": "partly hidden in results", "full": "hidden in results",
                     "off": "shown in results"}.get(semantics.masking, "hidden in results")
            c.small(f"{count} sensitive column{'s' if count != 1 else ''}: kept from the AI, {shown}.")

        if latest is not None:
            c.panel_heading("Latest query")
            facts = describe_query(latest.sql, info.dialect) if latest.source_id == info.id else None
            c.panel_value("Tables: " + (", ".join(latest.tables) or "none"))
            if facts is not None:
                c.panel_value("Filters: " + ("; ".join(facts.filters) if facts.filters else "none"))
            timing = []
            if latest.row_count is not None:
                timing.append(f"{latest.row_count:,} rows")
            if latest.execution_ms is not None:
                timing.append(f"{latest.execution_ms:,.0f} ms")
            if timing:
                c.small(" · ".join(timing))

        rules = list(semantics.default_filters) + list(semantics.ownership_rules)
        c.panel_heading("Business rules")
        if rules:
            for rule in rules:
                c.panel_value(rule)
        else:
            c.small("None set. Add default filters in the Data tab.")

        if semantics.metrics:
            c.panel_heading("Metrics")
            for metric in semantics.metrics:
                c.panel_value(metric.name)

        if latest is not None and latest.memories:
            c.panel_heading("Remembered for this answer")
            zone = c.visitor_timezone()
            for remembered in latest.memories:
                c.panel_value(str(remembered.get("content", "")))
                c.small(citations.describe(citations.Citation.from_dict(remembered.get("citation") or {}), zone))

        related = services.related_conversations(limit=3)
        if related:
            c.panel_heading("Related conversations")
            for record in related:
                st.button(record.title, key=f"related_{record.id}", on_click=_open_related,
                          args=(services, record.id))
