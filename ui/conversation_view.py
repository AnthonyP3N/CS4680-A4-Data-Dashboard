"""The sidebar (new conversation, data source, search and the conversation list) and the Saved tab."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import streamlit as st

from agent.conversation import ConversationError
from services import AppServices

from . import components as c
from . import data_view
from .chat_view import queue_question

TAGLINE = "Ask questions about your data in plain English. Get the answer, the chart, and the SQL behind it."
ANY = "Any"
PERIODS = {"Any time": None, "Today": 0, "Last 7 days": 7, "Last 30 days": 30}
KIND_LABELS = {"result": "Saved result", "chart": "Chart", "query": "Query"}


SHORTCUT_NEW = "Alt+N"
SHOW_CONTEXT = "show_context_panel"


def render_sidebar(services: AppServices) -> bool:
    """Draw the sidebar and return whether data changes are switched on."""
    with st.sidebar:
        c.wordmark(TAGLINE)
        if st.button("New conversation", type="primary", disabled=not services.ready, shortcut=SHORTCUT_NEW,
                     width="stretch", help="Start fresh. The current conversation stays saved."):
            services.new_conversation()
            st.rerun()
        _workspace_selector(services)
        _source_selector(services)
        _conversation_list(services)
        source = services.source
        info, semantics = source.info, source.semantics

        if semantics.example_questions:
            chatting = bool(services.conversation and services.conversation.history)
            with st.expander("Try asking", expanded=not chatting):
                for i, question in enumerate(semantics.example_questions):
                    st.button(question, key=f"ex_{info.id}_{i}", on_click=queue_question, args=(question,),
                              disabled=not services.ready)

        st.divider()
        allow_writes = st.toggle(
            "Allow data changes", value=False, disabled=not info.writable,
            help="When on, you can ask to update, add or delete records. "
                 "Every change is previewed and needs your confirmation.")
        if allow_writes:
            c.small("Changes only affect your private copy of the data.")
            for i, question in enumerate(semantics.example_changes):
                st.button(question, key=f"wex_{info.id}_{i}", on_click=queue_question, args=(question,),
                          disabled=not services.ready)
        st.toggle("Show context panel", value=True, key=SHOW_CONTEXT,
                  help="The panel beside the chat: data source, safety, the latest query and what it was based on.")
        if services.ready:
            c.small(f"Model: {services.model_label}")
    return bool(allow_writes) and info.writable


def _workspace_selector(services: AppServices) -> None:
    """Switch workspace. Shown once there is more than one (create them in Settings)."""
    workspaces = services.workspaces()
    if len(workspaces) < 2:
        return
    ids = [w.id for w in workspaces]
    names = {w.id: w.name for w in workspaces}
    chosen = st.selectbox("Workspace", ids, index=ids.index(services.workspace.id),
                          format_func=lambda workspace_id: names[workspace_id], key="sidebar_workspace")
    if chosen != services.workspace.id:
        services.switch_workspace(chosen)
        st.session_state.pop("_last_upload", None)
        st.rerun()


def _source_selector(services: AppServices) -> None:
    """Choose which data source to ask about. Shown once there is more than one."""
    sources = services.sources
    if len(sources) < 2:
        return
    ids = [s.info.id for s in sources]
    names = {s.info.id: s.info.name for s in sources}
    chosen = st.selectbox("Data source", ids, index=ids.index(services.source.info.id),
                          format_func=lambda source_id: names[source_id],
                          help="Each data source has its own conversations and business context.")
    if chosen != services.source.info.id:
        services.select_source(chosen)
        st.rerun()


# --------------------------------------------------------------------------- #
# Conversation list with search and filters
# --------------------------------------------------------------------------- #
def _filters(services: AppServices) -> dict:
    """The filter controls. Returns keyword arguments for services.find_conversations."""
    sources = {s.info.id: s.info.name for s in services.sources}
    tags = {t.id: t.name for t in services.tags()}
    with st.popover("Filters"):
        category = st.selectbox("Category", [ANY] + services.categories(), key="filter_category")
        source = st.selectbox("About", [ANY] + list(sources), key="filter_source",
                              format_func=lambda value: sources.get(value, ANY))
        tag = st.selectbox("Tag", [ANY] + list(tags), key="filter_tag", format_func=lambda value: tags.get(value, ANY))
        period = st.selectbox("Last active", list(PERIODS), key="filter_period")
        archived = st.toggle("Show archived conversations", key="filter_archived")
    days = PERIODS[period]
    zone = c.visitor_timezone()
    today = datetime.now(zone).date() if zone else date.today()
    return {
        "category": None if category == ANY else category,
        "data_source_id": None if source == ANY else source,
        "tag_id": None if tag == ANY else tag,
        "since": None if days is None else today - timedelta(days=days),
        "tz": zone,
        "archived": bool(archived),
    }


def _conversation_list(services: AppServices) -> None:
    """Saved conversations in this workspace. Search, filter, and click one to reopen it."""
    if not services.ready:
        return
    text = st.text_input("Search conversations", key="conversation_search", placeholder="Search conversations",
                         label_visibility="collapsed")
    filters = _filters(services)
    found = services.find_conversations(text=text or "", limit=15, **filters)
    filtering = bool(text) or any(value for name, value in filters.items() if name != "tz")

    if not found:
        st.markdown("**Archived**" if filters["archived"] else ("**Results**" if filtering else "**Recent**"))
        c.small("No conversations match." if filtering else "Your conversations will appear here.")
        return
    current = services.conversation.conversation if services.conversation else None
    names = {s.info.id: s.info.name for s in services.sources}
    if filters["archived"]:
        groups = [("Archived", found)]
    elif filtering:
        groups = [("Results", found)]
    else:
        groups = [("Pinned", [f for f in found if f[0].pinned]), ("Recent", [f for f in found if not f[0].pinned])]
    for heading, members in groups:
        if not members:
            continue
        st.markdown(f"**{heading}**")
        _conversation_buttons(services, members, current, names, text)


def _conversation_buttons(services: AppServices, found, current, names, text: str) -> None:
    for record, snippet in found:
        is_open = current is not None and current.id == record.id
        label = ("📌 " if record.pinned else "") + record.title
        about = names.get(record.data_source_id, "data that is no longer available")
        # The open conversation is wrapped so the stylesheet can mark it as the current one
        with st.container(key="conv_open") if is_open else st.container():
            if st.button(label, key=f"conv_{record.id}", disabled=is_open, width="stretch",
                         help=("Open now · " if is_open else "") + f"{record.category or 'General'} · {about}"):
                _open(services, record.id)
        if text and snippet and snippet.strip().lower() != record.title.strip().lower():
            c.small(snippet.strip())


def _open(services: AppServices, conversation_id: str) -> None:
    try:
        services.open_conversation(conversation_id)
    except ConversationError as error:
        st.error(str(error))
    else:
        st.rerun()


# --------------------------------------------------------------------------- #
# Saved tab
# --------------------------------------------------------------------------- #
def _open_from_saved(services: AppServices, conversation_id: str) -> None:
    """Button callback: open the conversation and switch to the Ask tab to show it."""
    try:
        services.open_conversation(conversation_id)
    except ConversationError as error:
        st.session_state._saved_error = str(error)
    else:
        c.go_to_tab("Ask")


def _preview(services: AppServices, item) -> None:
    """The saved chart, result or query, as it was stored."""
    result = services.saved_item_result(item)
    if result is None:
        c.small("The answer this points to is no longer available.")
        return
    if item.kind in ("chart", "result"):
        if result.df is None:
            c.small("The rows weren't saved with this answer (it reads a column marked sensitive). "
                    "Open the conversation and use Run again to see it.")
        elif item.kind == "chart" and result.figure is not None:
            c.plotly(result.figure, key=f"saved_chart_{item.id}")
        elif result.chart and result.chart.type == "metric" and result.chart.y in result.df.columns:
            c.metric(result.df[result.chart.y].iloc[0], result.chart.title or result.chart.y)
        else:
            c.dataframe(result.df, height=c.table_height(len(result.df), 320))
    if result.explanation:
        c.small(result.explanation)
    st.code(result.sql or "-- no query", language="sql")


def render_saved(services: AppServices) -> None:
    """Results the user saved and charts or queries they bookmarked."""
    error = st.session_state.pop("_saved_error", None)
    if error:
        st.error(error)
    items = services.saved_items()
    if not items:
        c.empty_state("Nothing saved yet",
                      "Use Save under an answer to keep a result, or to bookmark its chart or query, and find it here.")
        return
    titles = {record.id: record.title for record in
              services.storage.conversations.list(services.workspace.id, include_archived=True, limit=1000)}
    for item in items:
        left, open_column, remove_column = st.columns([6, 1.4, 1])
        origin = titles.get(item.conversation_id, "a conversation")
        details = f'{KIND_LABELS.get(item.kind, item.kind)} · from "{origin}" · {item.created_at[:10]}'
        if item.note:
            details += f" · {item.note}"
        with left:
            c.answer_text(item.title)
            c.small(details)
        open_column.button("Open conversation", key=f"saved_open_{item.id}", disabled=not services.ready,
                           on_click=_open_from_saved, args=(services, item.conversation_id),
                           help="Go to the Ask tab with the conversation this came from.")
        if remove_column.button("Remove", key=f"saved_remove_{item.id}"):
            services.delete_saved_item(item.id)
            st.rerun()
        with st.expander(f"Show the saved {item.kind}"):
            _preview(services, item)
