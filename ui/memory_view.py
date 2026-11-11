"""The Memory tab: see, add, edit and remove what AskData remembers for this workspace."""

from __future__ import annotations

import streamlit as st

from agent import citations
from agent.memory import MEMORY_TYPES, MemoryActionError
from services import AppServices
from storage.models import MemoryItem

from . import components as c

ALL_DATA = "All data in this workspace"
AUTO = "Choose for me"


def _scope_options(services: AppServices) -> dict[str, str | None]:
    """Label -> data source id (None for workspace-wide)."""
    return {ALL_DATA: None, **{f"Only {s.info.name}": s.info.id for s in services.sources}}


def _describe(services: AppServices, item: MemoryItem) -> str:
    names = {s.info.id: s.info.name for s in services.sources}
    scope = names.get(item.data_source_id, "data that is no longer available") if item.data_source_id else "all data"
    parts = [MEMORY_TYPES.get(item.memory_type, item.memory_type), f"applies to {scope}",
             citations.describe(citations.memory_citation(item), c.visitor_timezone()).replace("Based on", "From", 1)]
    if item.pinned:
        parts.append("pinned")
    if item.last_used_at:
        parts.append(f"last used {citations.format_day(item.last_used_at, c.visitor_timezone())}")
    return " · ".join(parts)


def render(services: AppServices) -> None:
    memory = services.memory
    if memory is None:
        return
    enabled = memory.enabled
    wanted = st.toggle("Use long-term memory", value=enabled, key=f"memory_enabled_{services.workspace.id}",
                       help="When off, nothing below is used in answers and nothing new is remembered or suggested. "
                            "What is already here is kept.")
    if wanted != enabled:
        memory.set_enabled(wanted)
        st.rerun()
    c.small("AskData remembers preferences and definitions you state, so you don't have to repeat them. It only "
            "remembers your own words, never query results. Say \"remember that ...\" in a conversation, add one "
            "here, or approve a suggestion. Relevant ones are sent with your questions about the data they apply to.")
    if not enabled:
        c.notice("Long-term memory is off for this workspace. Recent turns in a conversation still carry over.")

    _render_suggestions(services)
    _render_remembered(services)
    _render_add(services)
    st.divider()
    _render_clear(services)


def _render_suggestions(services: AppServices) -> None:
    suggested = services.memory.list(status="suggested")
    if not suggested:
        return
    st.markdown("**Suggested**")
    c.small("Noticed in your questions. Nothing here is used until you approve it.")
    for item in suggested:
        text_column, yes_column, no_column = st.columns([6, 1, 1])
        with text_column:
            c.answer_text(item.content)
            c.small(_describe(services, item))
        if yes_column.button("Approve", key=f"memory_approve_{item.id}", type="primary"):
            services.memory.approve(item.id)
            st.rerun()
        if no_column.button("Reject", key=f"memory_reject_{item.id}"):
            services.memory.reject(item.id)
            st.rerun()


def _render_remembered(services: AppServices) -> None:
    items = services.memory.list()
    st.markdown("**Remembered**")
    if not items:
        c.small("Nothing yet.")
        return
    scopes = _scope_options(services)
    scope_labels = list(scopes)
    types = list(MEMORY_TYPES)
    for item in items:
        text_column, edit_column, delete_column = st.columns([6, 1, 1])
        with text_column:
            c.answer_text(("📌 " if item.pinned else "") + item.content)
            c.small(_describe(services, item))
        with edit_column.popover("Edit"):
            content = st.text_area("What to remember", value=item.content, max_chars=500, key=f"memory_text_{item.id}")
            kind = st.selectbox("Kind", types, index=types.index(item.memory_type) if item.memory_type in types else 0,
                                format_func=lambda value: MEMORY_TYPES[value], key=f"memory_type_{item.id}")
            current_scope = next((label for label, source_id in scopes.items() if source_id == item.data_source_id),
                                 ALL_DATA)
            scope = st.selectbox("Applies to", scope_labels, index=scope_labels.index(current_scope),
                                 key=f"memory_scope_{item.id}")
            pinned = st.checkbox("Pinned: always include it", value=item.pinned, key=f"memory_pin_{item.id}")
            if st.button("Save changes", key=f"memory_save_{item.id}", type="primary"):
                try:
                    services.memory.update(item.id, content=content, memory_type=kind, pinned=pinned,
                                           data_source_id=scopes[scope])
                except MemoryActionError as error:
                    st.error(str(error))
                else:
                    st.rerun()
        with delete_column:
            if c.confirm_button(f"memory_delete_{item.id}", "Delete", "Delete this memory? This can't be undone.",
                                "Delete it"):
                services.memory.delete(item.id)
                st.rerun()


def _render_add(services: AppServices) -> None:
    scopes = _scope_options(services)
    current = next((label for label, source_id in scopes.items() if source_id == services.source.info.id), ALL_DATA)
    with st.form(f"memory_add_{services.workspace.id}", clear_on_submit=True):
        st.markdown("**Add a memory**")
        content = st.text_area("What should AskData remember?", max_chars=500,
                               placeholder="For example: Revenue always excludes cancelled orders")
        left, right = st.columns(2)
        kind = left.selectbox("Kind", [AUTO] + list(MEMORY_TYPES),
                              format_func=lambda value: MEMORY_TYPES.get(value, value))
        scope = right.selectbox("Applies to", list(scopes), index=list(scopes).index(current))
        if st.form_submit_button("Remember this"):
            try:
                services.memory.add(content, None if kind == AUTO else kind, data_source_id=scopes[scope])
            except MemoryActionError as error:
                st.error(str(error))
            else:
                st.rerun()


def _render_clear(services: AppServices) -> None:
    conversation = services.conversation
    record = conversation.conversation if conversation else None
    if record is not None:
        if c.confirm_button(
                f"memory_clear_conversation_{record.id}", "Forget this conversation's memory",
                f"This deletes what was remembered or suggested from \"{record.title}\" and its summary. "
                "The conversation's questions and answers stay. This can't be undone.", "Forget it"):
            removed = conversation.clear_memory()
            st.session_state._memory_notice = f"Forgot {removed} memor{'y' if removed == 1 else 'ies'} from that conversation."
            st.rerun()
    total = len(services.memory.list(status=None))
    if c.confirm_button(
            f"memory_clear_all_{services.workspace.id}", "Forget everything in this workspace",
            f"This permanently deletes all {total} memories and suggestions in this workspace. "
            "Conversations are not affected. This can't be undone.", "Forget everything"):
        removed = services.memory.clear_all()
        st.session_state._memory_notice = f"Forgot {removed} memor{'y' if removed == 1 else 'ies'}."
        st.rerun()
    notice = st.session_state.pop("_memory_notice", None)
    if notice:
        st.success(notice)

    st.markdown("**Your conversation history**")
    if st.session_state.get("_export_ready"):
        name, media_type, content = services.export_all()
        st.download_button("Download all conversations (JSON)", content, file_name=name, mime=media_type,
                           key="export_all_download")
    elif st.button("Prepare a download of all conversations"):
        st.session_state._export_ready = True
        st.rerun()
