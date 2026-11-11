"""Composition root: builds and wires the application's services.

Nothing in here (or below it) imports Streamlit. The UI asks `build_services`
for an `AppServices` object once per session and calls its methods; tests can
build the same object with fake parts.

Layout on disk (everything under the data directory, ASKDATA_DATA_DIR):

    askdata_meta.db                 metadata: workspaces, conversations, messages, business context
    workspaces/<workspace id>/      that workspace's data files (sample copy, uploads)
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import IO, Callable

from agent import export
from agent import llm as llm_module
from agent.audit import AuditLog
from agent.context import DEFAULT_CONTEXT_TOKENS, ContextBudget, ContextBuilder
from agent.conversation import ConversationError, ConversationService
from agent.llm import LLMClient, LLMError
from agent.memory import STOPWORDS, MemoryService
from agent.pipeline import AskDataAgent
from agent.ratelimit import Decision, RequestLimiter, client_key, shared_limiter
from data.adapters import DataSourceAdapter, DataSourceError, DataSourceLimits
from data.csv_adapter import CsvImport, import_csv
from data.metadata import SemanticConfig
from data.registry import DataSourceRegistry
from storage import Storage
from storage.identity import LOCAL_OWNER
from storage.models import Conversation, Message, SavedItem, Tag, Workspace

GetSecret = Callable[..., object]
KIND_NAMES = {"result": "result", "chart": "chart", "query": "query"}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "var"
META_FILE = "askdata_meta.db"
OFF_VALUES = ("off", "0", "false", "no")


@dataclass
class AppServices:
    session_id: str
    audit: AuditLog
    storage: Storage
    owner_id: str
    workspace: Workspace
    registry: DataSourceRegistry
    data_dir: Path | None = None          # None when nothing is being saved
    llm: LLMClient | None = None          # the client in use: the visitor's own key if they gave one
    setup_error: str = ""
    server_llm: LLMClient | None = None   # the client built from the app's own secrets, if any
    server_setup_error: str = ""
    allow_user_keys: bool = True          # whether visitors may paste their own API key
    allow_custom_providers: bool = True   # whether that key may be for an unlisted OpenAI-compatible provider
    key_source: str = ""                  # "server", "yours" or "" (none)
    user_provider: str = ""
    store_results: bool = True
    context_builder: ContextBuilder = field(default_factory=ContextBuilder)
    memory: MemoryService | None = None   # long-term memory for the current workspace
    limiter: RequestLimiter | None = None # question limits shared by every session (agent/ratelimit.py)
    client_id: str = ""                   # a hash of the visitor's network address, when known
    _open: dict[str, ConversationService] = field(default_factory=dict)   # data source id -> open conversation

    def __post_init__(self) -> None:
        self._use_workspace_memory()

    def _use_workspace_memory(self) -> None:
        """Point long-term memory, and the prompt builder that reads it, at the current workspace."""
        self.memory = MemoryService(self.storage, self.workspace.id, self.owner_id, self.audit)
        self.context_builder.long_term = self.memory

    # -- state ------------------------------------------------------------- #
    @property
    def ready(self) -> bool:
        return self.llm is not None

    @property
    def persistent(self) -> bool:
        return self.storage.persistent

    @property
    def source(self) -> DataSourceAdapter:
        return self.registry.active

    @property
    def sources(self) -> list[DataSourceAdapter]:
        return self.registry.list()

    @property
    def model_label(self) -> str:
        return self.llm.model if self.llm else ""

    # -- API keys ---------------------------------------------------------- #
    def use_own_key(self, provider: str, api_key: str, model: str = "", *, base_url: str = "",
                    name: str = "") -> str:
        """Answer this session's questions with the visitor's own API key. Returns the model label.

        The key is held only in this session's memory. It is never written to the
        metadata database, the activity log, exports or the page, and it is gone
        when the session ends. The provider sees it, as it must to answer.
        """
        if not self.allow_user_keys:
            raise LLMError("This app doesn't accept personal API keys.")
        provider = (provider or "").strip().lower()
        if provider != llm_module.CUSTOM_PROVIDER and provider not in llm_module.PROVIDERS:
            raise LLMError(f"Unknown provider '{provider}'.")
        api_key = (api_key or "").strip()
        if len(api_key) < 8 or any(ch.isspace() for ch in api_key):
            raise LLMError("That doesn't look like an API key. Copy it again from the provider's website.")
        if provider == llm_module.CUSTOM_PROVIDER:
            if not self.allow_custom_providers:
                raise LLMError("This app only accepts keys for the listed providers.")
            client = llm_module.build_custom_client(api_key, base_url, model, name)
        else:
            settings = {"LLM_PROVIDER": provider, llm_module.PROVIDERS[provider]["key"]: api_key,
                        f"{provider.upper()}_MODEL": model.strip() or None}
            client = llm_module.build_client(lambda key, default=None: settings.get(key) or default)
        self._switch_llm(client, "yours", provider)
        self.audit.log("llm_config", model=client.model, key_source="yours")   # never the key itself
        return client.model

    def stop_using_own_key(self) -> None:
        """Forget the visitor's key and go back to the app's own key, if it has one."""
        self._switch_llm(self.server_llm, "server" if self.server_llm else "", "")
        self.setup_error = self.server_setup_error
        self.audit.log("llm_config", model=self.model_label or None, key_source=self.key_source or "none")

    def _switch_llm(self, client: LLMClient | None, source: str, provider: str) -> None:
        self.llm, self.key_source, self.user_provider = client, source, provider
        self.setup_error = "" if client else self.setup_error
        for service in self._open.values():           # open conversations carry on with the new client
            service.agent.llm = client

    @property
    def needs_key(self) -> bool:
        """True when nothing can be answered until the visitor provides a key."""
        return self.llm is None and self.allow_user_keys

    def schema_text(self) -> str:
        """The schema exactly as it is sent to the model (sensitive values hidden)."""
        return self.context_builder.schema_text(self.source)

    # -- conversations ----------------------------------------------------- #
    @property
    def conversation(self) -> ConversationService | None:
        """The open conversation for the active data source.

        Each data source has its own, so context never crosses datasets. The
        first time a source is used in a session, its most recent saved
        conversation is reopened.
        """
        if self.llm is None:
            return None
        source = self.source
        if source.info.id not in self._open:
            latest = self.storage.conversations.list(self.workspace.id, data_source_id=source.info.id, limit=1)
            self._open[source.info.id] = self._service(source, latest[0] if latest else None)
        return self._open[source.info.id]

    def _service(self, source: DataSourceAdapter, record: Conversation | None) -> ConversationService:
        agent = AskDataAgent(source, self.llm, self.audit, context_builder=self.context_builder,
                             request_gate=self._request_gate)
        return ConversationService(agent, self.audit, self.storage, self.workspace.id, record,
                                   store_results=self.store_results, memory=self.memory)

    def _request_gate(self) -> Decision:
        if self.limiter is None:
            return Decision(True)
        return self.limiter.check(self.workspace.id, self.client_id, server_key=self.key_source == "server")

    def list_conversations(self, *, include_archived: bool = False, only_archived: bool = False,
                           limit: int = 30) -> list[Conversation]:
        """This workspace's saved conversations whose data is still available, most recent first."""
        available = set(self.registry.sources)
        found = self.storage.conversations.list(self.workspace.id, include_archived=include_archived,
                                                only_archived=only_archived, limit=limit)
        return [c for c in found if c.data_source_id in available]

    def new_conversation(self) -> None:
        """Start a fresh conversation about the active data source. The current one stays saved."""
        if self.conversation is not None:
            self.conversation.clear()

    def open_conversation(self, conversation_id: str) -> ConversationService:
        """Reopen a saved conversation and switch to its data source."""
        if self.llm is None:
            raise ConversationError("Add an API key before opening conversations.")
        record = self.storage.conversations.get(self.workspace.id, conversation_id)
        if record is None:
            raise ConversationError("That conversation doesn't exist in this workspace.")
        if record.data_source_id not in self.registry.sources:
            raise ConversationError("The data this conversation was about is no longer available.")
        current = self._open.get(record.data_source_id)
        if current is not None and current.conversation is not None and current.conversation.id == record.id:
            self.registry.select(record.data_source_id)
            return current
        if current is not None:
            current.clear()                # cancels any change still waiting in the conversation being left
        source = self.registry.select(record.data_source_id)
        self._open[source.info.id] = self._service(source, record)
        self.audit.log("open_conversation", conversation=record.id)
        return self._open[source.info.id]

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        changed = self.storage.conversations.update(self.workspace.id, conversation_id, title=title)
        self._refresh_open(conversation_id)
        return changed

    def archive_conversation(self, conversation_id: str, archived: bool = True) -> bool:
        changed = self.storage.conversations.update(self.workspace.id, conversation_id, archived=archived)
        if changed and archived:
            self._close_if_open(conversation_id)
        return changed

    def pin_conversation(self, conversation_id: str, pinned: bool = True) -> bool:
        return self.storage.conversations.update(self.workspace.id, conversation_id, pinned=pinned)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Permanently remove a conversation and its messages."""
        self._close_if_open(conversation_id)
        deleted = self.storage.conversations.delete(self.workspace.id, conversation_id)
        if deleted:
            self.audit.log("delete_conversation", conversation=conversation_id)
        return deleted

    def find_conversations(self, *, text: str = "", category: str | None = None, data_source_id: str | None = None,
                           tag_id: str | None = None, since: str | date | None = None,
                           until: str | date | None = None, tz: tzinfo | None = None,
                           archived: bool = False, limit: int = 30) -> list[tuple[Conversation, str]]:
        """Search and filter this workspace's conversations. Returns (conversation, matching text) pairs.

        since / until: calendar days (inclusive) in time zone `tz`, the computer's own zone by default.
        """
        available = set(self.registry.sources)
        since_at, until_at = day_bounds(since, until, tz)
        found = self.storage.conversations.find(
            self.workspace.id, text=text, category=category, data_source_id=data_source_id, tag_id=tag_id,
            since=since_at, until=until_at, archived=archived, limit=limit)
        return [(c, snippet) for c, snippet in found if c.data_source_id in available]

    def related_conversations(self, limit: int = 3) -> list[Conversation]:
        """Other conversations about the same data that mention the words of the latest question."""
        current = self.conversation
        if current is None or not current.history:
            return []
        words = [w for w in current.history[-1].question.lower().replace("?", " ").split()
                 if w not in STOPWORDS and len(w) > 2]
        if not words:
            return []
        found = self.storage.conversations.find(
            self.workspace.id, text=" ".join(words), match_any=True, data_source_id=self.source.info.id,
            exclude_id=current.conversation.id if current.conversation else None, limit=limit)
        return [c for c, _ in found]

    def duplicate_conversation(self, conversation_id: str) -> Conversation:
        """Copy a conversation's visible turns into a new one. A change still waiting for
        confirmation is copied as cancelled, so a copy can never apply anything."""
        original = self.storage.conversations.get(self.workspace.id, conversation_id)
        if original is None:
            raise ConversationError("That conversation doesn't exist in this workspace.")
        copy = self.storage.conversations.create(self.workspace.id, f"Copy of {original.title}",
                                                 data_source_id=original.data_source_id, category=original.category)
        new_ids: dict[str, str] = {}
        for message in self.storage.messages.list(self.workspace.id, conversation_id):
            status, metadata = message.status, dict(message.result_metadata)
            if status == "needs_confirmation":
                status = "cancelled"
                metadata["error"] = "This change was waiting for confirmation in the original conversation and was not copied as pending."
            clone = Message(conversation_id=copy.id, role=message.role, content=message.content,
                            parent_message_id=new_ids.get(message.parent_message_id or ""), status=status,
                            intent=message.intent, sql=message.sql, explanation=message.explanation,
                            reasoning_summary=message.reasoning_summary, chart_spec=dict(message.chart_spec),
                            result_metadata=metadata, request_id=message.request_id, model=message.model)
            self.storage.messages.add(self.workspace.id, clone)
            new_ids[message.id] = clone.id
        for tag in self.storage.tags.for_conversation(self.workspace.id, conversation_id):
            self.storage.tags.assign(self.workspace.id, copy.id, tag.id)
        self.audit.log("duplicate_conversation", conversation=conversation_id, copy=copy.id)
        return self.storage.conversations.get(self.workspace.id, copy.id)

    # -- categories and tags ------------------------------------------------ #
    def categories(self) -> list[str]:
        return self.storage.conversations.categories(self.workspace.id)

    def set_category(self, conversation_id: str, category: str) -> bool:
        changed = self.storage.conversations.update(self.workspace.id, conversation_id, category=category.strip()[:40])
        self._refresh_open(conversation_id)
        return changed

    def tags(self) -> list[Tag]:
        return self.storage.tags.list(self.workspace.id)

    def tags_for(self, conversation_id: str) -> list[Tag]:
        return self.storage.tags.for_conversation(self.workspace.id, conversation_id)

    def set_tags(self, conversation_id: str, names: list[str]) -> list[Tag]:
        """Make the conversation's tags exactly `names`, creating tags that don't exist yet."""
        if self.storage.conversations.get(self.workspace.id, conversation_id) is None:
            raise ConversationError("That conversation doesn't exist in this workspace.")
        wanted = {n.strip()[:40] for n in names if n and n.strip()}
        current = {t.name: t for t in self.tags_for(conversation_id)}
        for name in wanted - set(current):
            self.storage.tags.assign(self.workspace.id, conversation_id,
                                     self.storage.tags.create(self.workspace.id, name).id)
        for name in set(current) - wanted:
            self.storage.tags.unassign(self.workspace.id, conversation_id, current[name].id)
        return self.tags_for(conversation_id)

    # -- saved results and bookmarks ----------------------------------------- #
    def save_item(self, request_id: str, kind: str, title: str = "", note: str = "") -> SavedItem:
        """Save a result, or bookmark a chart or query, from the open conversation."""
        current = self.conversation
        turn = next((r for r in current.history if r.request_id == request_id), None) if current else None
        if turn is None or not turn.message_id:
            raise ConversationError("That answer hasn't been saved, so it can't be bookmarked.")
        if kind == "chart" and (turn.chart is None or turn.chart.type in ("table", "metric")):
            raise ConversationError("That answer has no chart to bookmark.")
        if kind in ("query", "chart", "result") and not turn.sql:
            raise ConversationError("That answer has no query to save.")
        if any(i.message_id == turn.message_id for i in self.saved_items(kind)):
            raise ConversationError(f"This {KIND_NAMES.get(kind, kind)} is already in the Saved tab.")
        item = self.storage.saved_items.add(self.workspace.id, turn.message_id, kind, title or turn.question, note)
        self.audit.log("save_item", request_id=request_id, kind=kind)
        if self.memory is not None and self.memory.enabled and not turn.sensitive_columns:
            # Saving is a clear signal that this analysis matters, so later questions can refer back to it.
            # Only the name and the question are remembered, never the result.
            self.memory.add(f'Saved analysis "{item.title}": it answered the question "{turn.question}"',
                            "saved_analysis", data_source_id=turn.source_id or None,
                            conversation_id=item.conversation_id, source_message_id=turn.message_id,
                            importance=0.6)
        return item

    def saved_item_result(self, item: SavedItem):
        """The stored answer a saved item points to, rebuilt for display (None if it is gone)."""
        from agent.conversation import to_result
        answer = self.storage.messages.get(self.workspace.id, item.message_id)
        question = self.storage.messages.get(self.workspace.id, answer.parent_message_id) if answer else None
        return to_result(question, answer) if answer and question else None

    def saved_items(self, kind: str | None = None) -> list[SavedItem]:
        return self.storage.saved_items.list(self.workspace.id, kind)

    def delete_saved_item(self, item_id: str) -> bool:
        return self.storage.saved_items.delete(self.workspace.id, item_id)

    # -- export ---------------------------------------------------------------- #
    def _document(self, conversation: Conversation) -> dict:
        names = {s.info.id: s.info.name for s in self.sources}
        return export.conversation_document(
            conversation, self.storage.messages.list(self.workspace.id, conversation.id),
            source_name=names.get(conversation.data_source_id or "", ""),
            tags=[t.name for t in self.tags_for(conversation.id)])

    def export_conversation(self, conversation_id: str, fmt: str, *, record: bool = True) -> tuple[str, str, bytes]:
        """(file name, media type, content) for one conversation as "json", "markdown" or "html".

        `record=False` skips the activity-log entry, for callers that prepare the
        file before the user decides to download it and log the download themselves.
        """
        if fmt not in export.EXPORTERS:
            raise ConversationError(f"Unknown export format '{fmt}'.")
        conversation = self.storage.conversations.get(self.workspace.id, conversation_id)
        if conversation is None:
            raise ConversationError("That conversation doesn't exist in this workspace.")
        extension, media_type, render = export.EXPORTERS[fmt]
        if record:
            self.audit.log("export_conversation", conversation=conversation_id, format=fmt)
        return (export.safe_filename(conversation.title, extension), media_type,
                render(self._document(conversation)).encode("utf-8"))

    def export_all(self) -> tuple[str, str, bytes]:
        """Every conversation in this workspace, archived ones included, as one JSON file."""
        conversations = self.storage.conversations.list(self.workspace.id, include_archived=True, limit=10_000)
        document = export.workspace_document(self.workspace.name, [self._document(c) for c in conversations])
        self.audit.log("export_all", conversations=len(conversations))
        return (export.safe_filename(self.workspace.name + " conversations", "json"), "application/json",
                export.to_json(document).encode("utf-8"))

    def _close_if_open(self, conversation_id: str) -> None:
        for service in self._open.values():
            if service.conversation is not None and service.conversation.id == conversation_id:
                service.clear()

    def _refresh_open(self, conversation_id: str) -> None:
        for service in self._open.values():
            if service.conversation is not None and service.conversation.id == conversation_id:
                service.conversation = self.storage.conversations.get(self.workspace.id, conversation_id)

    # -- data sources ------------------------------------------------------ #
    def select_source(self, source_id: str) -> DataSourceAdapter:
        source = self.registry.select(source_id)
        self.audit.log("select_source", source=source.info.name)
        return source

    def import_csv(self, file: IO, filename: str) -> CsvImport:
        """Add a CSV as a table in the "Uploaded files" data source and switch to it."""
        uploads = self.registry.ensure_uploads()
        imported = import_csv(uploads, file, filename)
        self.registry.persist(uploads)
        self.registry.select(uploads.info.id)
        self.audit.log("upload_csv", table=imported.table, rows=imported.rows, columns=imported.columns,
                       date_columns=imported.date_columns or None, truncated=imported.truncated or None)
        return imported

    def update_semantics(self, config: SemanticConfig) -> list[str]:
        """Replace the active data source's business context. Returns warnings about unknown references."""
        source = self.source
        source.semantics = config
        self.registry.persist(source)
        # The content itself isn't logged: a glossary can contain confidential business rules
        self.audit.log("semantics_updated", source=source.info.name,
                       glossary_terms=len(config.glossary), metrics=len(config.metrics),
                       sensitive_columns=len(config.sensitive_columns))
        return config.problems(source.schema())

    def restore_sample(self) -> None:
        """Undo every edit to the sample data and its business context. Conversations are kept."""
        sample = self.registry.sample
        if sample is None:
            raise DataSourceError("There is no sample data in this workspace.")
        open_service = self._open.get(sample.info.id)
        if open_service is not None and open_service.pending is not None:
            open_service.cancel(open_service.pending.request_id)
        self.registry.reset_sample(sample)
        self.audit.log("restore_sample")

    def uploads_summary(self) -> tuple[int, int]:
        """(tables, conversations) that removing the uploaded files would delete."""
        uploads = self.registry.uploads
        if uploads is None:
            return 0, 0
        return len(uploads.schema().tables), self.storage.conversations.count(self.workspace.id, uploads.info.id)

    def remove_uploads(self) -> None:
        """Delete the uploaded files and the conversations about them."""
        uploads = self.registry.uploads
        if uploads is None:
            return
        self._open.pop(uploads.info.id, None)
        self.registry.remove(uploads.info.id)
        self.audit.log("remove_uploads")

    def reset(self) -> None:
        """Back to the starting data: sample restored, uploads removed, and a fresh conversation."""
        self.remove_uploads()
        if self.registry.sample is not None:
            self.restore_sample()
            self.registry.select(self.registry.sample.info.id)
        self.new_conversation()
        self.audit.log("reset_data")

    # -- workspaces -------------------------------------------------------- #
    def workspaces(self) -> list[Workspace]:
        return self.storage.workspaces.list(self.owner_id)

    def create_workspace(self, name: str) -> Workspace:
        workspace = self.storage.workspaces.create(self.owner_id, name)
        self.audit.log("create_workspace", workspace_name=workspace.name)
        return self.switch_workspace(workspace.id)

    def switch_workspace(self, workspace_id: str) -> Workspace:
        """Move to another of this owner's workspaces. Someone else's workspace id is refused."""
        workspace = self.storage.workspaces.get(self.owner_id, workspace_id)
        if workspace is None:
            raise DataSourceError("That workspace doesn't exist.")
        for service in self._open.values():
            if service.pending is not None:
                service.cancel(service.pending.request_id)
        self._open.clear()
        self.workspace = workspace
        self.registry = _registry_for(self.storage, workspace, self.data_dir, self.registry.limits)
        self.audit.workspace_id = workspace.id
        self._use_workspace_memory()
        self.audit.log("switch_workspace")
        return workspace

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete another of this owner's workspaces with its conversations and data files."""
        if workspace_id == self.workspace.id:
            raise DataSourceError("Switch to another workspace before deleting this one.")
        deleted = self.storage.workspaces.delete(self.owner_id, workspace_id)
        if deleted and self.data_dir is not None:
            folder = self.data_dir / "workspaces" / workspace_id
            if folder.parent == self.data_dir / "workspaces":
                shutil.rmtree(folder, ignore_errors=True)
        return deleted


def day_bounds(since: str | date | None, until: str | date | None,
               tz: tzinfo | None = None) -> tuple[str | None, str | None]:
    """Turn calendar days into the UTC instants that bound them.

    `since` and `until` are days as the person sees them (inclusive), in time zone
    `tz` (the computer's own zone when omitted). Timestamps are stored in UTC, and
    a local day does not line up with a UTC day: 6 pm on the 10th in Los Angeles is
    already the 11th in UTC. Returns (start of `since`, start of the day after
    `until`) as UTC text in the stored timestamp format.
    """
    zone = tz or datetime.now().astimezone().tzinfo

    def instant(day: str | date, extra_days: int) -> str:
        day = date.fromisoformat(day[:10]) if isinstance(day, str) else day
        local_midnight = datetime.combine(day + timedelta(days=extra_days), time.min, tzinfo=zone)
        return local_midnight.astimezone(timezone.utc).isoformat(timespec="milliseconds")

    return (instant(since, 0) if since else None, instant(until, 1) if until else None)


def query_limits(get_secret: GetSecret) -> DataSourceLimits:
    """Per-query limits from settings, kept within safe bounds.

    ASKDATA_MAX_ROWS        most rows one answer may return (10 to 10,000; default 1,000)
    ASKDATA_QUERY_TIMEOUT_S seconds before a query is stopped (0.5 to 60; default 8)
    """
    defaults = DataSourceLimits()

    def number(name: str, default: float, low: float, high: float) -> float:
        try:
            return min(high, max(low, float(get_secret(name, default))))
        except (TypeError, ValueError):
            return default
    return DataSourceLimits(query_timeout_s=number("ASKDATA_QUERY_TIMEOUT_S", defaults.query_timeout_s, 0.5, 60),
                            max_rows=int(number("ASKDATA_MAX_ROWS", defaults.max_rows, 10, 10_000)))


def _registry_for(storage: Storage, workspace: Workspace, data_dir: Path | None,
                  limits: DataSourceLimits | None = None) -> DataSourceRegistry:
    """The workspace's data sources: reloaded from its own folder when saving is on, temporary otherwise."""
    work_dir = data_dir / "workspaces" / workspace.id if (storage.persistent and data_dir is not None) else None
    registry = DataSourceRegistry(work_dir=work_dir, storage=storage, workspace_id=workspace.id)
    registry.limits = limits
    registry.load()
    return registry


def build_services(get_secret: GetSecret, *, owner_id: str = LOCAL_OWNER, llm: LLMClient | None = None,
                   limiter: RequestLimiter | None = None, client_address: str = "",
                   registry: DataSourceRegistry | None = None, audit: AuditLog | None = None,
                   storage: Storage | None = None, data_dir: str | Path | None = None) -> AppServices:
    """Create the services for one user session.

    get_secret(name, default=None) supplies configuration.
    owner_id   who the session belongs to (see storage/identity.py)
    storage    where conversations are kept; opened from the data directory when omitted
    llm, registry, audit   substitutes for tests
    """
    session_id = uuid.uuid4().hex[:10]
    configured_dir = data_dir or get_secret("ASKDATA_DATA_DIR", None)
    data_path = Path(str(configured_dir)) if configured_dir else DEFAULT_DATA_DIR
    if storage is None:
        storage = Storage.open(data_path / META_FILE)
    if not storage.persistent:
        data_path = None

    workspace = storage.workspaces.ensure_default(owner_id)
    if audit is None:
        log_path = get_secret("ASKDATA_LOG_PATH", None)
        audit = AuditLog(session_id, Path(str(log_path))) if log_path else AuditLog(session_id)
    audit.workspace_id = workspace.id
    audit.log("session_start", saving=storage.persistent)
    if storage.warning:
        audit.log("storage_unavailable", status="failed", error=storage.warning)

    limits = query_limits(get_secret)
    if registry is None:
        registry = _registry_for(storage, workspace, data_path, limits)
    else:
        if not registry.sources:
            registry.create_sample()
        registry.attach(storage, workspace.id)

    keep_results = str(get_secret("ASKDATA_STORE_RESULTS", "on") or "on").strip().lower() not in OFF_VALUES
    try:
        context_tokens = max(200, int(get_secret("ASKDATA_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS)))
    except (TypeError, ValueError):
        context_tokens = DEFAULT_CONTEXT_TOKENS
    services = AppServices(session_id=session_id, audit=audit, storage=storage, owner_id=owner_id,
                           workspace=workspace, registry=registry, data_dir=data_path, store_results=keep_results,
                           context_builder=ContextBuilder(budget=ContextBudget(total_tokens=context_tokens)))
    services.limiter = limiter or shared_limiter(get_secret)
    services.client_id = client_key(client_address)
    services.allow_user_keys = str(get_secret("ASKDATA_ALLOW_USER_KEYS", "on") or "on").strip().lower() not in OFF_VALUES
    services.allow_custom_providers = (
        str(get_secret("ASKDATA_ALLOW_CUSTOM_PROVIDERS", "on") or "on").strip().lower() not in OFF_VALUES)
    try:
        services.server_llm = llm or llm_module.build_client(get_secret)
        services.llm, services.key_source = services.server_llm, "server"
        audit.log("llm_config", model=services.llm.model, key_source="server")
    except LLMError as e:
        services.setup_error = services.server_setup_error = str(e)

    # Reopen where the owner left off: the data source of their most recent conversation
    recent = services.list_conversations(limit=1)
    if recent and recent[0].data_source_id in registry.sources:
        registry.select(recent[0].data_source_id)
    return services
