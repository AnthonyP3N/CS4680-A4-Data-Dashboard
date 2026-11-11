"""AskData: ask questions about your data in plain English.

Run locally:   streamlit run app.py

This file only wires things together: it works out who the visitor is, builds
the services once per browser session and hands them to the views in ui/. All
logic lives in agent/, data/, storage/ and services.py, none of which depend
on Streamlit.
"""

from __future__ import annotations

import os

import streamlit as st

from services import AppServices, build_services
from storage import identity
from ui import (chat_view, components, context_view, conversation_view, data_view, memory_view, settings_view,
                theme)

st.set_page_config(page_title="AskData", page_icon="📊", layout="wide", initial_sidebar_state="auto")   # collapsed on phones


def get_secret(name: str, default=None):
    """Read a setting from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets.toml at all
        pass
    return os.environ.get(name, default)


def _host() -> str | None:
    """The address this page was opened at, used only to pick a safe default identity mode."""
    try:
        url = st.context.url
        if url:
            return str(url)
    except Exception:
        pass
    try:
        return st.context.headers.get("Host")
    except Exception:
        return None


def _client_address() -> str:
    """The visitor's network address, used only (hashed) for per-client question limits."""
    try:
        return str(st.context.ip_address or "")
    except Exception:
        return ""


def resolve_owner() -> str:
    """Whose conversations to show. See storage/identity.py for the three modes."""
    mode = identity.resolve_mode(get_secret("ASKDATA_AUTH"), _host())
    st.session_state.auth_mode = mode
    if mode == "local":
        return identity.LOCAL_OWNER
    if mode == "link":
        token = st.query_params.get("u")
        if not identity.valid_token(token):
            token = identity.new_token()
            st.query_params["u"] = token          # the address now carries this browser's private link
        return identity.owner_from_token(token)

    # mode == "login": Streamlit's built-in sign-in
    if not getattr(st.user, "is_logged_in", False):
        components.wordmark(conversation_view.TAGLINE)
        st.write("Sign in to see your conversations.")
        st.button("Sign in", type="primary", on_click=st.login)
        st.stop()
    return identity.owner_from_login(str(st.user.get("sub") or st.user.get("email") or ""))


def get_services(owner_id: str) -> AppServices:
    """One set of services per browser session, rebuilt if the owner changes."""
    if st.session_state.get("owner_id") != owner_id or "services" not in st.session_state:
        with st.spinner("Preparing your data..."):
            st.session_state.services = build_services(get_secret, owner_id=owner_id,
                                                       client_address=_client_address())
        st.session_state.owner_id = owner_id
        st.session_state.queued = None
    return st.session_state.services


theme.inject()
services = get_services(resolve_owner())
allow_writes = conversation_view.render_sidebar(services)

# A key lets other screens switch tabs (for example "Open" on a saved item jumps to Ask)
tab_ask, tab_saved, tab_memory, tab_data, tab_log, tab_settings = st.tabs(
    ["Ask", "Saved", "Memory", "Data", "Activity log", "Settings"], key=components.MAIN_TABS, on_change="rerun")
with tab_ask:
    if st.session_state.get(conversation_view.SHOW_CONTEXT, True):
        chat_column, context_column = st.columns([2.7, 1], gap="large")
        with context_column:
            context_view.render(services, allow_writes)
        with chat_column:
            chat_view.render(services, allow_writes)
    else:
        chat_view.render(services, allow_writes)
with tab_saved:
    conversation_view.render_saved(services)
with tab_memory:
    memory_view.render(services)
with tab_data:
    data_view.render(services)
with tab_log:
    settings_view.render_activity_log(services)
with tab_settings:
    settings_view.render_settings(services)
