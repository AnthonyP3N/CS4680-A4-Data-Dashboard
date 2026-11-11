"""Activity log and settings: where conversations are saved, who they belong to, and workspaces."""

from __future__ import annotations

import io

import streamlit as st

from agent.llm import CUSTOM_PROVIDER, PROVIDERS, LLMError
from data.adapters import DataSourceError
from services import AppServices
from storage.migrations import LATEST_VERSION

from . import components as c

LOG_COLUMNS = ["ts", "request", "event", "status", "question", "intent", "sql", "error",
               "rows", "affected_rows", "latency_ms", "model"]


def render_activity_log(services: AppServices) -> None:
    st.markdown("Every step the agent takes is recorded: questions, model responses, safety checks, "
                "queries run and changes confirmed or cancelled.")
    log = services.audit.read()
    if log.empty:
        c.small("Nothing yet. Ask a question to start the log.")
        return
    c.dataframe(log[[column for column in LOG_COLUMNS if column in log.columns]], height=460)
    buffer = io.StringIO()
    log.to_json(buffer, orient="records", lines=True)
    st.download_button("Download log (JSONL)", buffer.getvalue().encode(),
                       file_name=f"audit_{services.session_id}.jsonl", mime="application/json")


IDENTITY_NOTES = {
    "local": "This app treats everyone using it as one person. That suits an app on your own computer. "
             "On a shared or public address, set ASKDATA_AUTH to \"link\" or \"login\".",
    "link": "Your conversations are tied to the private address of this page. Bookmark it to come back to "
            "them. Anyone who has this address can read them, so share it only if you mean to.",
    "login": "Your conversations are tied to the account you signed in with.",
}


SHORTCUTS = [("Alt+N (Option+N on a Mac)", "Start a new conversation"),
             ("Alt+G (Option+G on a Mac)", "Regenerate the latest answer"),
             ("Enter", "Send the question in the question box"),
             ("Shift+Enter", "New line in the question box"),
             ("Esc", "Close an open menu")]


def render_settings(services: AppServices) -> None:
    render_model_settings(services)
    st.markdown("**Keyboard shortcuts**")
    c.detail_list(SHORTCUTS)
    c.small("Switch between light and dark in the ⋮ menu at the top right, under Settings.")
    st.markdown("**Saving**")
    if services.persistent:
        c.small(f"Conversations and business context are saved in {services.storage.path.name} "
                f"(format version {services.storage.version} of {LATEST_VERSION}). "
                "Uploaded files are kept in this workspace's own folder.")
        if not services.store_results:
            c.small("Result rows are not saved (ASKDATA_STORE_RESULTS is off); answers can be run again.")
        else:
            c.small("A preview of each result (up to 200 rows) is saved so charts reappear, except for "
                    "results that include a column marked sensitive.")
    else:
        c.notice(services.storage.warning or "Saving is turned off for this session.", "err",
                 lead="Nothing is being saved.")

    st.markdown("**Who these conversations belong to**")
    c.small(IDENTITY_NOTES.get(st.session_state.get("auth_mode", "local"), ""))

    st.markdown("**Workspace**")
    c.small("A workspace has its own data, business context and conversations. Nothing is shared between them.")
    workspaces = services.workspaces()
    ids = [w.id for w in workspaces]
    names = {w.id: w.name for w in workspaces}
    if len(ids) > 1:
        chosen = st.selectbox("Current workspace", ids, index=ids.index(services.workspace.id),
                              format_func=lambda workspace_id: names[workspace_id])
        if chosen != services.workspace.id:
            _switch(services, chosen)
    else:
        c.small(f"Current workspace: {services.workspace.name}")
    with st.form("new_workspace", clear_on_submit=True):
        name = st.text_input("New workspace name", max_chars=80)
        if st.form_submit_button("Create workspace") and name.strip():
            services.create_workspace(name)
            st.session_state.pop("_last_upload", None)
            st.rerun()


def _switch(services: AppServices, workspace_id: str) -> None:
    try:
        services.switch_workspace(workspace_id)
    except DataSourceError as error:
        st.error(str(error))
    else:
        st.session_state.pop("_last_upload", None)
        st.rerun()


KEY_PRIVACY = ("Your key stays in this browser session only. It is not saved, not written to the activity log, and "
               "is forgotten when you close the page. It is sent to the provider you choose, which needs it to answer.")


OWN_PROVIDER_LABEL = "Enter your own"
KEY_HELP = {
    "groq": "Free key: console.groq.com/keys",
    "gemini": "Free key: aistudio.google.com/apikey",
    "openai": "Paid key: platform.openai.com/api-keys",
    "openrouter": "Free models available: openrouter.ai/keys",
}


def render_key_form(services: AppServices, key: str) -> None:
    """Let a visitor answer questions with their own API key.

    The provider is a fixed choice (no typing). "Enter your own" takes any
    OpenAI-compatible API: its address, model and a display name.
    """
    # Free options first
    listed = [p for p in ("groq", "gemini", "openrouter", "openai") if p in PROVIDERS]
    options = listed + ([CUSTOM_PROVIDER] if services.allow_custom_providers else [])
    labels = {**{p: PROVIDERS[p]["name"] for p in PROVIDERS}, CUSTOM_PROVIDER: OWN_PROVIDER_LABEL}
    # Outside the form, so choosing "Enter your own" shows its extra fields straight away
    provider = st.radio("Provider", options, format_func=lambda p: labels[p], horizontal=True,
                        key=f"own_key_provider_{key}")
    custom = provider == CUSTOM_PROVIDER
    if custom:
        c.small("Any provider with an OpenAI-compatible API, such as Together, Mistral, DeepSeek or Fireworks. "
                "The address must be https and on the public internet.")
    with st.form(f"own_key_{key}", clear_on_submit=True):
        name = base_url = ""
        if custom:
            name = st.text_input("Provider name (optional)", key=f"own_key_name_{key}", max_chars=40,
                                 placeholder="For example: Together AI")
            base_url = st.text_input("API address", key=f"own_key_url_{key}",
                                     placeholder="https://api.together.xyz/v1")
        api_key = st.text_input("API key", type="password", key=f"own_key_value_{key}",
                                help=KEY_HELP.get(provider, "The key from your provider's website."))
        model = st.text_input("Model" if custom else "Model (optional)", key=f"own_key_model_{key}",
                              help=("The exact model name your provider uses." if custom else
                                    "Leave empty to let AskData choose and fall back automatically."))
        c.small(KEY_PRIVACY)
        submitted = st.form_submit_button("Use this key", type="primary")
    if submitted:
        try:
            label = services.use_own_key(provider, api_key, model, base_url=base_url, name=name)
        except LLMError as error:
            st.error(str(error))
        else:
            st.session_state._key_notice = f"Using your key: {label}."
            st.rerun()


def render_model_settings(services: AppServices) -> None:
    st.markdown("**AI model**")
    notice = st.session_state.pop("_key_notice", None)
    if notice:
        st.success(notice)
    if services.key_source == "yours":
        c.small(f"Answering with your own key ({services.model_label}). "
                + KEY_PRIVACY)
        if st.button("Stop using my key"):
            services.stop_using_own_key()
            st.rerun()
        return
    if services.key_source == "server":
        c.small(f"Answering with this app's key ({services.model_label}).")
    if not services.allow_user_keys:
        if services.key_source != "server":
            c.small("No API key is configured. The person running this app needs to add one to its secrets.")
        return
    with st.expander("Use your own API key" if services.key_source == "server" else "Add an API key",
                     expanded=services.key_source != "server"):
        render_key_form(services, "settings")
