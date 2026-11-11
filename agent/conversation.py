"""The conversation service: the one object the user interface talks to for chat.

It holds one open conversation and enforces the rules around it, so those rules
hold no matter which screen (or test) is driving:

- A data change can only be applied while it is the pending, most recent turn.
- Asking something new abandons an unconfirmed change (it is cancelled, never applied).
- Applying a change requires data changes to still be switched on.

Every turn is written through to storage (two messages per turn, plus a record
of each query run), and `ConversationService(..., conversation=record)` loads
one back. What is and isn't saved:

- Saved: the question, explanation, SQL, chart choice, timings and outcome.
- Saved, capped: a preview of the result (first rows only) so charts reappear
  after a reload.
- Never saved: results of queries that read a column marked sensitive, write
  previews, and anything shaped like a password or API key.

Nothing loaded from storage is trusted to run: saved SQL is validated again
before it is re-run, and a change that was waiting for confirmation is put
through a fresh dry run before it can be applied.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Callable

import pandas as pd

from data.adapters import DataSourceError, redact
from data.registry import to_record
from storage import Storage
from storage.models import Conversation, Message, QueryRun

from .audit import AuditLog
from .charts import ChartSpec, apply_view, resolve_spec
from .executor import ExecutionError
from .memory import REMEMBER_COMMAND, MemoryActionError, MemoryService, summarize_turns
from .pipeline import AgentResult, AskDataAgent, Attempt
from .validator import READ, ValidationError, validate_sql

PREVIEW_ROWS = 200        # most rows of a result kept with a conversation
# Notes added when a chart had to be changed; replaced whenever the view changes
VIEW_NOTE_PREFIXES = ("Switched to", "Showing a table", "Adjusted the chart", "This result can't be shown", "Unknown view")
PREVIEW_COLUMNS = 30
PREVIEW_CELL = 300        # longest text value kept
TITLE_LENGTH = 60
WRITE_STATUSES = ("needs_confirmation", "committed", "cancelled")


# Categories are worked out from the first question with plain keyword rules:
# no model call, the same answer every time, and easy to read and change.
# The first rule that matches wins; the user can pick another category later.
CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Trends over time", ("over time", "trend", "monthly", "weekly", "daily", "yearly", "each month", "by month",
                          "per month", "by week", "by year", "growth", "last 12 months", "year over year")),
    ("Rankings", ("top ", "highest", "lowest", "best", "worst", "most ", "least ", "rank", "biggest", "smallest")),
    ("Breakdowns", ("break down", "breakdown", "share of", "split", "distribution", "for each", " by ", " per ")),
    ("Totals and counts", ("how many", "how much", "count", "total", "sum of", "average", "number of")),
    ("Lookups", ("show ", "list ", "which ", "who ", "find ", "what is", "what was", "details")),
)
DATA_CHANGE_CATEGORY = "Data changes"
DEFAULT_CATEGORY = "General"


def categorize(question: str, intent: str = "") -> str:
    """The category for a conversation that starts with this question."""
    if intent == "modify":
        return DATA_CHANGE_CATEGORY
    if intent == "remember":
        return DEFAULT_CATEGORY
    text = f" {' '.join(question.lower().split())} "
    for category, keywords in CATEGORY_RULES:
        if any(keyword in text for keyword in keywords):
            return category
    return DEFAULT_CATEGORY


class ConversationError(Exception):
    """The requested conversation action isn't allowed right now."""


# --------------------------------------------------------------------------- #
# Turning results into stored messages and back
# --------------------------------------------------------------------------- #
def title_from(question: str) -> str:
    """A conversation title made from its first question."""
    text = " ".join(str(redact(question)).split())
    if len(text) <= TITLE_LENGTH:
        return text or "New conversation"
    return text[:TITLE_LENGTH].rsplit(" ", 1)[0].rstrip(",.;:") + "..."


def snapshot(df: pd.DataFrame | None) -> dict[str, Any] | None:
    """A small, JSON-safe copy of a result: the first rows and columns, long text clipped."""
    if df is None or df.empty:
        return None
    part = df.iloc[:PREVIEW_ROWS, :PREVIEW_COLUMNS]
    data = json.loads(part.to_json(orient="split", date_format="iso"))
    rows = [[(v[:PREVIEW_CELL] + "...") if isinstance(v, str) and len(v) > PREVIEW_CELL else v for v in row]
            for row in data["data"]]
    return {"columns": [str(c) for c in data["columns"]], "rows": rows,
            "truncated": len(df) > PREVIEW_ROWS or df.shape[1] > PREVIEW_COLUMNS}


def frame_from(saved: dict[str, Any] | None) -> pd.DataFrame | None:
    if not saved or not saved.get("columns"):
        return None
    try:
        return pd.DataFrame(saved.get("rows") or [], columns=saved["columns"])
    except (ValueError, TypeError):
        return None


def _chart_to_dict(chart: ChartSpec | None) -> dict[str, Any]:
    if chart is None:
        return {}
    return {"type": chart.type, "x": chart.x, "y": chart.y, "color": chart.color, "title": chart.title}


def _chart_from_dict(saved: dict[str, Any]) -> ChartSpec | None:
    if not saved.get("type"):
        return None
    return ChartSpec(type=str(saved["type"]), x=str(saved.get("x", "")), y=str(saved.get("y", "")),
                     color=str(saved.get("color", "")), title=str(saved.get("title", "")))


def to_messages(result: AgentResult, conversation_id: str, *, store_results: bool) -> tuple[Message, Message]:
    """The user message and the assistant message for one turn."""
    user = Message(conversation_id=conversation_id, role="user", content=str(redact(result.question)),
                   request_id=result.request_id)
    keep_rows = store_results and result.status == "answer" and not result.sensitive_columns
    metadata = {
        "error": result.error, "notes": list(result.notes), "follow_ups": list(result.follow_ups),
        "failed_attempts": [{"sql": a.sql, "stage": a.stage, "error": a.error} for a in result.failed_attempts],
        "statement": result.statement, "target_table": result.target_table,
        "source_id": result.source_id, "source_name": result.source_name,
        "llm_calls": result.llm_calls, "elapsed_ms": round(result.elapsed_ms, 1),
        "execution_ms": result.execution_ms,
        "row_count": result.row_count if result.row_count is not None else (len(result.df) if result.df is not None else None),
        "columns": [str(c) for c in result.df.columns] if result.df is not None else [],
        "sensitive_columns": list(result.sensitive_columns),
        "masked_columns": list(result.masked_columns),
        "tables": list(result.tables),
        "memories": list(result.memories),          # which long-term memories were in the prompt, with citations
        "summary_used": result.summary_used,
        "preview": snapshot(result.df) if keep_rows else None,
    }
    assistant = Message(
        conversation_id=conversation_id, role="assistant", parent_message_id=user.id,
        content=str(redact(result.explanation or result.error)), status=result.status, intent=result.intent,
        sql=str(redact(result.sql)), explanation=str(redact(result.explanation)),
        reasoning_summary=str(redact(result.reasoning)), chart_spec=_chart_to_dict(result.chart),
        result_metadata=redact(metadata), request_id=result.request_id, model=result.model)
    return user, assistant


def to_result(user: Message, assistant: Message) -> AgentResult:
    """Rebuild a turn from its stored messages."""
    meta = assistant.result_metadata
    df = frame_from(meta.get("preview"))
    result = AgentResult(
        request_id=assistant.request_id, question=user.content, status=assistant.status,
        explanation=assistant.explanation, reasoning=assistant.reasoning_summary, sql=assistant.sql,
        df=df, chart=_chart_from_dict(assistant.chart_spec), follow_ups=[str(f) for f in meta.get("follow_ups") or []],
        notes=[str(n) for n in meta.get("notes") or []],
        failed_attempts=[Attempt(str(a.get("sql", "")), str(a.get("stage", "")), str(a.get("error", "")))
                         for a in meta.get("failed_attempts") or [] if isinstance(a, dict)],
        error=str(meta.get("error") or ""), model=assistant.model, llm_calls=int(meta.get("llm_calls") or 0),
        elapsed_ms=float(meta.get("elapsed_ms") or 0), statement=str(meta.get("statement") or ""),
        target_table=str(meta.get("target_table") or ""), source_id=str(meta.get("source_id") or ""),
        source_name=str(meta.get("source_name") or ""), intent=assistant.intent,
        row_count=meta.get("row_count"), sensitive_columns=[str(c) for c in meta.get("sensitive_columns") or []],
        masked_columns=[str(c) for c in meta.get("masked_columns") or []],
        message_id=assistant.id, user_message_id=user.id, created_at=assistant.created_at, restored=True,
        execution_ms=meta.get("execution_ms"), tables=[str(t) for t in meta.get("tables") or []],
        memories=[m for m in meta.get("memories") or [] if isinstance(m, dict)],
        summary_used=bool(meta.get("summary_used")),
        preview_truncated=bool((meta.get("preview") or {}).get("truncated")))
    return result


def _query_runs(result: AgentResult, message_id: str) -> list[QueryRun]:
    """One record per statement the agent tried for this turn."""
    runs = [QueryRun(message_id=message_id, sql=str(redact(a.sql)),
                     validation_status="failed" if a.stage == "validation" else "passed",
                     execution_status="not_run" if a.stage == "validation" else "failed",
                     error_message=a.error) for a in result.failed_attempts if a.sql]
    if result.sql and result.status in ("answer", "blocked", *WRITE_STATUSES):
        execution = {"answer": "ok", "blocked": "not_run", "needs_confirmation": "dry_run",
                     "committed": "committed", "cancelled": "cancelled"}[result.status]
        rows = result.write_preview.affected_rows if result.write_preview else result.row_count
        runs.append(QueryRun(message_id=message_id, sql=str(redact(result.sql)),
                             validation_status="blocked" if result.status == "blocked" else "passed",
                             execution_status=execution, row_count=rows, execution_time=round(result.elapsed_ms, 1),
                             error_message=result.error))
    return runs


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #
class ConversationService:
    """One open conversation about one data source.

    Without `storage` it keeps to a private in-memory store, which is what
    scripts and tests want. Pass `conversation` to reopen a saved one.
    """

    def __init__(self, agent: AskDataAgent, audit: AuditLog, storage: Storage | None = None,
                 workspace_id: str | None = None, conversation: Conversation | None = None, *,
                 store_results: bool = True, memory: MemoryService | None = None):
        self.agent = agent
        self.audit = audit
        self.storage = storage or Storage.in_memory()
        self.workspace_id = workspace_id or self.storage.workspaces.ensure_default("local").id
        self.store_results = store_results
        self.memory = memory             # long-term memory for the workspace; None means none is kept
        self.conversation = conversation
        self.storage_error = ""          # set when a turn could not be saved
        self.history: list[AgentResult] = self._load() if conversation is not None else []
        self._recheck_pending()

    # -- reading ----------------------------------------------------------- #
    @property
    def pending(self) -> AgentResult | None:
        """The data change waiting for the user's decision, if any."""
        if self.history and self.history[-1].status == "needs_confirmation":
            return self.history[-1]
        return None

    @property
    def title(self) -> str:
        return self.conversation.title if self.conversation else "New conversation"

    def is_latest(self, result: AgentResult) -> bool:
        return bool(self.history) and self.history[-1] is result

    # -- actions ----------------------------------------------------------- #
    def ask(self, question: str, allow_writes: bool = False, *,
            progress: Callable[[str], None] | None = None) -> AgentResult:
        waiting = self.pending
        if waiting is not None:
            self._cancel(waiting)
        command = REMEMBER_COMMAND.match(question or "")
        if command:
            result = self._remember(question, command.group("body"))
        else:
            result = self._ask_agent(question, self.history, allow_writes, progress=progress)
        self._save_turn(result)
        self.history.append(result)
        self._after_turn(result)
        return result

    def _ask_agent(self, question: str, history: list[AgentResult], allow_writes: bool, *,
                   progress: Callable[[str], None] | None = None) -> AgentResult:
        result = self.agent.ask(question, history=history, allow_writes=allow_writes,
                                conversation_summary=self.summary_of(history),
                                conversation_id=self.conversation.id if self.conversation else None,
                                progress=progress)
        if result.sensitive_columns and result.status == "answer" and self.store_results:
            result.notes.append("This result includes a column marked sensitive, so its rows aren't saved "
                                "with the conversation.")
        return result

    # -- memory ------------------------------------------------------------ #
    def summary_of(self, history: list[AgentResult]) -> str:
        """A summary of the turns that have dropped out of the short-term window ("" while everything fits)."""
        older = self.agent.context_builder.memory.older(history)
        return summarize_turns(older, dialect=self.agent.source.info.dialect) if older else ""

    @property
    def summary(self) -> str:
        return self.conversation.summary if self.conversation else ""

    def _remember(self, question: str, body: str) -> AgentResult:
        """Handle "remember that ...": store it as a long-term memory. No model call is made."""
        info = self.agent.source.info
        result = AgentResult(request_id=uuid.uuid4().hex[:8], question=question, status="noted", intent="remember",
                             source_id=info.id, source_name=info.name)
        self.audit.log("question", request_id=result.request_id, question=question, source=info.name)
        if self.memory is None or not self.memory.enabled:
            result.status = "unsupported"
            result.explanation = ("Long-term memory is turned off for this workspace, so that wasn't saved. "
                                  "You can turn it on in the Memory tab.")
            return result
        try:
            self._ensure_conversation(question, "remember")
            item = self.memory.add(body.strip(), data_source_id=info.id,
                                   conversation_id=self.conversation.id if self.conversation else None)
        except (MemoryActionError, sqlite3.Error) as error:
            result.status, result.explanation = "unsupported", f"That couldn't be remembered: {error}"
            return result
        result.explanation = f"I'll remember this for questions about {info.name}: “{item.content}”"
        result.notes.append("You can edit, pin or remove it in the Memory tab.")
        return result

    def _after_turn(self, result: AgentResult) -> None:
        """Keep the rolling summary current, and offer to remember standing instructions in the question."""
        self._refresh_summary()
        if self.memory is None or result.status in ("error", "blocked", "noted") or result.intent == "remember":
            return
        try:
            offered = self.memory.suggest_from_question(
                result.question, data_source_id=self.agent.source.info.id,
                conversation_id=self.conversation.id if self.conversation else None,
                source_message_id=result.user_message_id or None)
        except sqlite3.Error as error:
            self._storage_failed(error)
            return
        result.suggestions = [{"id": m.id, "type": m.memory_type, "content": m.content} for m in offered]

    def _refresh_summary(self) -> None:
        if self.conversation is None:
            return
        summary = self.summary_of(self.history)
        if summary != self.conversation.summary:
            self._update_conversation(summary=summary)

    def pending_suggestions(self) -> list:
        """Memories suggested from this conversation that are still waiting for a yes or no."""
        if self.memory is None or self.conversation is None or not self.memory.enabled:
            return []
        return [m for m in self.memory.list(status="suggested") if m.conversation_id == self.conversation.id]

    def clear_memory(self) -> int:
        """Forget what was remembered from this conversation (its turns stay).

        Removes the long-term memories and suggestions that came from it and its
        stored summary. If the conversation continues, a summary is built again
        from the turns still in it.
        """
        if self.conversation is None:
            return 0
        if self.memory is not None:
            removed = self.memory.clear_conversation(self.conversation.id)
        else:
            removed = 0
            self.storage.conversations.update(self.workspace_id, self.conversation.id, summary="")
        self.conversation = self.storage.conversations.get(self.workspace_id, self.conversation.id)
        return removed

    def set_view(self, request_id: str, view: str) -> AgentResult:
        """Show an answer as a different chart, a table, a single number or no chart, and remember the choice.

        A view that doesn't fit the data falls back to a table, with a note.
        """
        turn = self._turn(request_id)
        if turn.status != "answer":
            raise ConversationError("Only answers with results can change how they're shown.")
        if turn.df is None:
            raise ConversationError("This result's rows weren't saved. Run it again first.")
        spec = apply_view(turn.df, turn.chart, view)
        turn.chart = spec
        turn.notes = [n for n in turn.notes if not n.startswith(VIEW_NOTE_PREFIXES)]
        if spec.adjusted:
            turn.notes.append(spec.adjusted)
        if turn.message_id:
            try:
                self.storage.messages.update(self.workspace_id, turn.message_id, chart_spec=_chart_to_dict(spec))
            except sqlite3.Error as error:
                self._storage_failed(error)
        self.audit.log("change_view", request_id=request_id, view=spec.type)
        return turn

    def regenerate(self, request_id: str, allow_writes: bool = False, *,
                   progress: Callable[[str], None] | None = None) -> AgentResult:
        """Ask the latest question again and replace its answer (also how a failed turn is retried).

        The earlier answer is kept in storage but hidden. A change that was already
        applied can't be regenerated: that would hide the record of it.
        """
        turn = self._turn(request_id)
        if not self.is_latest(turn):
            raise ConversationError("Only the latest answer can be regenerated. Edit the question instead.")
        if turn.status == "committed":
            raise ConversationError("That change was already applied, so its record can't be replaced.")
        if turn.intent == "remember":
            raise ConversationError("That was a request to remember something. Edit or remove it in the Memory tab.")
        if turn.status == "needs_confirmation":
            self._cancel(turn)                       # never leave a pending change behind a new answer
        result = self._ask_agent(turn.question, self.history[:-1], allow_writes, progress=progress)
        self._hide([turn.message_id])
        self._save_turn(result, user_message_id=turn.user_message_id)
        self.history[-1] = result
        self._refresh_summary()
        self.audit.log("regenerate", request_id=result.request_id, replaces=turn.request_id)
        return result

    def edit_and_resubmit(self, request_id: str, question: str, allow_writes: bool = False) -> AgentResult:
        """Replace an earlier question. That turn and everything after it leave the conversation
        (kept in storage, hidden), then the new question is asked.

        Refused if any of those turns applied a data change, so the record of a change is never hidden.
        """
        if not question or not question.strip():
            raise ConversationError("Type the new question first.")
        turn = self._turn(request_id)
        index = self.history.index(turn)
        removed = self.history[index:]
        if any(r.status == "committed" for r in removed):
            raise ConversationError("A data change was applied at or after that point, so it can't be edited. "
                                    "Start a new conversation instead.")
        waiting = self.pending
        if waiting is not None:
            self._cancel(waiting)
        self._hide([i for r in removed for i in (r.user_message_id, r.message_id)])
        self.history = self.history[:index]
        self.audit.log("edit_question", request_id=request_id, removed_turns=len(removed))
        result = self.ask(question, allow_writes=allow_writes)
        if index == 0 and self.conversation is not None and self.conversation.title == title_from(turn.question):
            self._update_conversation(title=title_from(question))     # the title followed the old first question
        return result

    def confirm(self, request_id: str, allow_writes: bool) -> AgentResult:
        waiting = self._require_pending(request_id)
        if not allow_writes:
            self.audit.log("commit", request_id=request_id, status="blocked", sql=waiting.sql,
                           error="data changes are switched off")
            raise ConversationError("Data changes are switched off. Turn on 'Allow data changes' to apply this.")
        if waiting.write_preview is None:
            raise ConversationError("This change needs to be checked again before it can be applied. Ask again.")
        result = self.agent.confirm(waiting)
        self._save_outcome(result)
        return result

    def cancel(self, request_id: str) -> AgentResult:
        return self._cancel(self._require_pending(request_id))

    def clear(self) -> None:
        """Start a new conversation. The previous one stays saved."""
        waiting = self.pending
        if waiting is not None:
            self._cancel(waiting)
        self.conversation = None
        self.history = []

    def rerun(self, request_id: str) -> AgentResult:
        """Run a saved question's query again against the current data.

        The stored SQL is validated from scratch, and only read-only queries can be re-run.
        """
        result = next((r for r in self.history if r.request_id == request_id), None)
        if result is None or not result.sql or result.status != "answer":
            raise ConversationError("There is no saved query to run again for that turn.")
        source = self.agent.source
        try:
            validated = validate_sql(result.sql, source.schema(), allow_writes=False,
                                     max_rows=source.info.limits.max_rows, dialect=source.info.dialect,
                                     sensitive=source.semantics.sensitive_set())
            if validated.kind != READ:
                raise ValidationError("Only read-only queries can be run again.", policy=True)
            outcome = source.run_read(validated)
        except (ValidationError, ExecutionError, DataSourceError) as error:
            self.audit.log("rerun", request_id=request_id, status="failed", sql=result.sql, error=str(error))
            raise ConversationError(f"Couldn't run that again: {error}") from None
        chart = result.chart
        guarded = self.agent.guard(outcome.df, validated)
        result.df, result.row_count, result.preview_truncated = guarded.df, len(guarded.df), False
        result.sensitive_columns = validated.sensitive_columns
        result.masked_columns = guarded.masked_columns
        result.chart = resolve_spec(outcome.df, {
            "chart_type": chart.type if chart else "table", "x": chart.x if chart else "",
            "y": chart.y if chart else "", "color": chart.color if chart else "",
            "chart_title": chart.title if chart else ""})
        self.audit.log("rerun", request_id=request_id, sql=validated.sql, rows=len(outcome.df))
        return result

    # -- storage ----------------------------------------------------------- #
    def _load(self) -> list[AgentResult]:
        messages = self.storage.messages.list(self.workspace_id, self.conversation.id)
        users = {m.id: m for m in messages if m.role == "user"}
        return [to_result(users[m.parent_message_id], m) for m in messages
                if m.role == "assistant" and m.parent_message_id in users]

    def _save_turn(self, result: AgentResult, user_message_id: str = "") -> None:
        """Store a turn. With `user_message_id`, the answer is attached to that existing question."""
        try:
            self._ensure_conversation(result.question, result.intent)
            user, assistant = to_messages(result, self.conversation.id, store_results=self.store_results)
            if user_message_id:
                assistant.parent_message_id = user_message_id
            else:
                self.storage.messages.add(self.workspace_id, user)
                user_message_id = user.id
            self.storage.messages.add(self.workspace_id, assistant)
            for run in _query_runs(result, assistant.id):
                self.storage.query_runs.add(self.workspace_id, run)
            result.message_id, result.created_at = assistant.id, assistant.created_at
            result.user_message_id = user_message_id
            self.storage_error = ""
        except (sqlite3.Error, LookupError) as error:
            self._storage_failed(error)

    def _ensure_conversation(self, first_question: str, intent: str) -> None:
        """Create the conversation record (and its data-source record) the first time something is stored."""
        source = self.agent.source
        if self.storage.data_sources.get(self.workspace_id, source.info.id) is None:
            self.storage.data_sources.save(to_record(source, self.workspace_id))
        if self.conversation is None:
            self.conversation = self.storage.conversations.create(
                self.workspace_id, title_from(first_question), data_source_id=source.info.id,
                category=categorize(first_question, intent))

    def _hide(self, message_ids: list[str]) -> None:
        ids = [i for i in message_ids if i]
        if not ids:
            return
        try:
            self.storage.messages.deactivate(self.workspace_id, ids)
        except sqlite3.Error as error:
            self._storage_failed(error)

    def _update_conversation(self, **changes: Any) -> None:
        if self.conversation is None:
            return
        try:
            self.storage.conversations.update(self.workspace_id, self.conversation.id, **changes)
            self.conversation = self.storage.conversations.get(self.workspace_id, self.conversation.id)
        except sqlite3.Error as error:
            self._storage_failed(error)

    def _save_outcome(self, result: AgentResult) -> None:
        """Record what happened to a pending change (applied, failed or cancelled)."""
        if not result.message_id:
            return
        try:
            stored = self.storage.messages.get(self.workspace_id, result.message_id)
            if stored is None:
                return
            metadata = dict(stored.result_metadata)
            metadata.update({"error": result.error, "notes": list(result.notes)})
            self.storage.messages.update(self.workspace_id, result.message_id, status=result.status,
                                         content=str(redact(result.explanation or result.error)),
                                         result_metadata=redact(metadata))
            execution = {"committed": "committed", "cancelled": "cancelled"}.get(result.status, "failed")
            self.storage.query_runs.add(self.workspace_id, QueryRun(
                message_id=result.message_id, sql=str(redact(result.sql)), validation_status="passed",
                execution_status=execution, error_message=result.error,
                row_count=result.write_preview.affected_rows if result.write_preview else None))
        except (sqlite3.Error, LookupError) as error:
            self._storage_failed(error)

    def _storage_failed(self, error: Exception) -> None:
        self.storage_error = "This conversation couldn't be saved, so it will be lost when you leave."
        self.audit.log("storage_error", status="failed", error=str(error))

    # -- helpers ----------------------------------------------------------- #
    def _cancel(self, waiting: AgentResult) -> AgentResult:
        result = self.agent.cancel(waiting)
        self._save_outcome(result)
        return result

    def _turn(self, request_id: str) -> AgentResult:
        turn = next((r for r in self.history if r.request_id == request_id), None)
        if turn is None:
            raise ConversationError("That turn isn't part of this conversation any more.")
        return turn

    def _require_pending(self, request_id: str) -> AgentResult:
        waiting = self.pending
        if waiting is None or waiting.request_id != request_id:
            raise ConversationError("That change is no longer waiting for confirmation. Ask again to redo it.")
        return waiting

    def _recheck_pending(self) -> None:
        """A change loaded from storage must pass validation and a fresh dry run before it can be applied.

        The saved preview is never reused: the data may have changed since it was made.
        If the change no longer checks out, it is cancelled.
        """
        waiting = self.pending
        if waiting is None or waiting.write_preview is not None:
            return
        source = self.agent.source
        try:
            if not source.info.writable:
                raise DataSourceError(f"'{source.info.name}' is read-only.")
            validated = validate_sql(waiting.sql, source.schema(), allow_writes=True,
                                     max_rows=source.info.limits.max_rows, dialect=source.info.dialect)
            if validated.kind == READ:
                raise ValidationError("The saved statement is not a data change.", policy=True)
            waiting.write_preview = source.dry_run_write(validated)
            waiting.sql, waiting.statement = validated.sql, validated.statement
            waiting.target_table = validated.target_table or ""
            self.audit.log("dry_run", request_id=waiting.request_id, sql=validated.sql,
                           affected_rows=waiting.write_preview.affected_rows, reason="reloaded")
        except (ValidationError, ExecutionError, DataSourceError) as error:
            waiting.status = "cancelled"
            waiting.error = f"This change could not be re-checked after reloading, so it was cancelled: {error}"
            self.audit.log("commit", request_id=waiting.request_id, status="cancelled", sql=waiting.sql,
                           error=str(error), reason="failed re-check after reload")
            self._save_outcome(waiting)
