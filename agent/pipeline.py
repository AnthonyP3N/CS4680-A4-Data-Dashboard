"""The agent loop: question -> LLM -> validate -> execute (or confirm) -> chart.

    user question
         |
         v
    [LLM] structured JSON (intent, sql, chart spec)
         |
         +-- clarify / unsupported --> reply to user
         |
         v
    [Validator] parse, allow-list, safety rules ----- policy block --> refuse
         |                \
         |                 +-- fixable error --> feed back to LLM (self-correction)
         v
    read:  [Executor] read-only run ---- DB error --> feed back to LLM
    write: [Executor] dry run + preview --> user confirms --> commit
         |
         v
    [Charts] table + Plotly figure, audit log at every step
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path

import pandas as pd

from . import prompts
from data import sample_store
from data.adapters import DataSourceAdapter, DataSourceError
from data.sqlite_adapter import SQLiteAdapter

from . import masking
from .audit import AuditLog
from .charts import ChartSpec, build_figure, resolve_spec
from .context import ContextBuilder
from .executor import ExecutionError, WritePreview
from .llm import LLMClient, LLMError
from .memory import DEFAULT_TURNS
from .ratelimit import Decision
from .schema import SchemaInfo
from .validator import READ, ValidationError, validate_sql

RATE_LIMITED_NOTE = "Question limit reached: nothing was sent to the AI."
MAX_REPAIRS = 2          # extra LLM calls allowed to fix a broken query
HISTORY_TURNS = DEFAULT_TURNS   # previous Q/A pairs sent for follow-up questions (see agent/memory.py)


@dataclass
class Attempt:
    sql: str
    stage: str           # "validation" | "execution"
    error: str


@dataclass
class AgentResult:
    request_id: str
    question: str
    status: str          # answer | clarify | unsupported | needs_confirmation | blocked | error | committed | cancelled
    explanation: str = ""
    reasoning: str = ""
    sql: str = ""
    df: pd.DataFrame | None = None
    chart: ChartSpec | None = None
    follow_ups: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    failed_attempts: list[Attempt] = field(default_factory=list)
    error: str = ""
    model: str = ""
    llm_calls: int = 0
    elapsed_ms: float = 0.0
    write_preview: WritePreview | None = None
    statement: str = ""
    target_table: str = ""
    source_id: str = ""
    source_name: str = ""
    intent: str = ""                 # what the model said it was doing: query, modify, clarify, unsupported
    row_count: int | None = None     # rows in the full result (the frame may hold only a saved preview)
    sensitive_columns: list[str] = field(default_factory=list)   # sensitive columns the SQL may read
    masked_columns: list[str] = field(default_factory=list)      # result columns whose values were hidden
    # Set by the conversation service when a turn is saved or loaded
    message_id: str = ""
    user_message_id: str = ""        # the stored question this answer belongs to
    created_at: str = ""
    execution_ms: float | None = None   # time the database spent on the final query (elapsed_ms is the whole turn)
    tables: list[str] = field(default_factory=list)     # tables the final query read or changed
    memories: list[dict] = field(default_factory=list)  # long-term memories put in the prompt (with citations)
    summary_used: bool = False                           # whether a summary of older turns was in the prompt
    suggestions: list[dict] = field(default_factory=list)   # memories offered for approval after this turn
    restored: bool = False           # loaded from storage rather than produced in this session
    preview_truncated: bool = False  # the saved preview holds only the first rows

    @property
    def figure(self):
        return self.figure_for(None)

    def figure_for(self, palette: list[str] | None):
        """The chart in the given series colours (the viewer's theme), or None when there is no chart."""
        if self.df is None or self.chart is None:
            return None
        return build_figure(self.df, self.chart, palette)

    def history_json(self) -> str:
        """Compact version of this turn for conversation memory."""
        return json.dumps({"intent": {"answer": "query"}.get(self.status, self.status),
                           "sql": self.sql, "explanation": self.explanation})


class AskDataAgent:
    """Runs one question through the loop above against one data source.

    `source` is a DataSourceAdapter. A path to a SQLite file is also accepted
    and wrapped automatically, which keeps simple scripts and tests short.
    """

    def __init__(self, source: DataSourceAdapter | str | Path, llm: LLMClient, audit: AuditLog, *,
                 sample_db: bool = True, context_builder: ContextBuilder | None = None,
                 request_gate: Callable[[], "Decision"] | None = None):
        if not isinstance(source, DataSourceAdapter):
            source = SQLiteAdapter(source, is_sample=sample_db,
                                   semantics=sample_store.config() if sample_db else None)
        self.source: DataSourceAdapter = source
        self.llm = llm
        self.audit = audit
        self.context_builder = context_builder or ContextBuilder()
        # Asked before each question reaches the model; refuses it when a rate limit is reached
        self.request_gate = request_gate
        self.source.schema()   # fail early if the data source can't be read

    # ------------------------------------------------------------------ #
    @property
    def schema(self) -> SchemaInfo:
        return self.source.schema()

    def refresh_schema(self) -> SchemaInfo:
        return self.source.schema(refresh=True)

    # ------------------------------------------------------------------ #
    def ask(self, question: str, history: list[AgentResult] | None = None,
            allow_writes: bool = False, *, conversation_summary: str = "",
            conversation_id: str | None = None, progress: Callable[[str], None] | None = None) -> AgentResult:
        """Answer one question. `progress`, if given, is called with a short description of each stage."""
        def stage(text: str) -> None:
            if progress is not None:
                try:
                    progress(text)
                except Exception:  # a display problem must never break the answer
                    pass

        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        info = self.source.info
        result = AgentResult(request_id=rid, question=question, status="error",
                             source_id=info.id, source_name=info.name)
        self.audit.log("question", request_id=rid, question=question, allow_writes=allow_writes,
                       source=info.name)

        question = question.strip()
        if not question:
            result.error = "Please type a question."
            return result
        if len(question) > 1000:
            result.error = "That question is too long (1,000 character limit)."
            self.audit.log("rejected_input", request_id=rid, status="blocked", reason="too long")
            return result

        if self.request_gate is not None:
            decision = self.request_gate()
            if not decision.allowed:
                result.error = decision.message
                result.notes.append(RATE_LIMITED_NOTE)
                self.audit.log("rate_limited", request_id=rid, status="blocked", limit=decision.limit,
                               retry_after_s=round(decision.retry_after_s))
                result.elapsed_ms = (time.perf_counter() - t0) * 1000
                return result

        context = self.context_builder.build(question, history or [], self.source,
                                             summary=conversation_summary, conversation_id=conversation_id)
        system, turns = context.system, context.turns
        result.memories = [m.to_dict() for m in context.memories]
        result.summary_used = context.summary_used
        self.audit.log("context", request_id=rid, recent_turns=context.remembered,
                       memories=[m.item.id for m in context.memories] or None,
                       summary=context.summary_used or None, tokens=context.tokens)

        for attempt in range(MAX_REPAIRS + 1):
            # ---- 1. LLM ------------------------------------------------- #
            stage("Reading your question" if attempt == 0 else "Fixing a problem with the query")
            try:
                resp = self.llm.generate(system, turns)
            except LLMError as e:
                result.error = str(e)
                self.audit.log("llm_error", request_id=rid, status="error", error=str(e))
                break
            result.llm_calls += resp.attempts
            result.model = resp.model
            data = resp.data
            self.audit.log("llm_response", request_id=rid, model=resp.model, intent=data["intent"],
                           sql=data["sql"], latency_ms=round(resp.latency_ms), usage=resp.usage,
                           repair_round=attempt)

            result.intent = data["intent"]
            result.explanation = data["explanation"]
            result.reasoning = data["reasoning"]
            result.follow_ups = data["follow_ups"]
            result.sql = data["sql"]

            if data["intent"] in ("clarify", "unsupported"):
                result.status = data["intent"]
                result.sql = ""
                if not result.follow_ups:
                    # Always leave the user with something this data can answer
                    result.follow_ups = list(self.source.semantics.example_questions[:2])
                break

            # ---- 2. Validate --------------------------------------------- #
            stage("Checking the query is safe")
            try:
                v = validate_sql(data["sql"], self.schema, allow_writes=allow_writes,
                                 max_rows=info.limits.max_rows, dialect=info.dialect,
                                 sensitive=self.source.semantics.sensitive_set())
            except ValidationError as e:
                self.audit.log("validation", request_id=rid, status="blocked" if e.policy else "failed",
                               sql=data["sql"], error=str(e))
                if e.policy:
                    result.status, result.error = "blocked", str(e)
                    break
                result.failed_attempts.append(Attempt(data["sql"], "validation", str(e)))
                turns += [("model", resp.raw_text),
                          ("user", prompts.repair_prompt(question, data["sql"], "validation", str(e), info.dialect))]
                result.error = str(e)
                continue

            if v.kind == READ and data["intent"] == "modify":
                v.notes.append("The model labelled this as a change, but it only reads data.")
            self.audit.log("validation", request_id=rid, sql=v.sql, kind=v.kind, tables=v.tables, notes=v.notes)
            result.sql, result.notes, result.statement = v.sql, v.notes, v.statement
            result.sensitive_columns = v.sensitive_columns
            result.tables = list(v.tables)

            # ---- 3a. Write: dry run, then wait for confirmation ----------- #
            if v.kind != READ:
                stage("Trying the change without saving it")
            if v.kind != READ:
                if not self.source.info.writable:
                    msg = f"'{self.source.info.name}' is read-only, so data changes can't be made here."
                    self.audit.log("validation", request_id=rid, status="blocked", sql=v.sql, error=msg)
                    result.status, result.error = "blocked", msg
                    break
                try:
                    preview = self.source.dry_run_write(v)
                except (ExecutionError, DataSourceError) as e:
                    self.audit.log("dry_run", request_id=rid, status="failed", sql=v.sql, error=str(e))
                    result.failed_attempts.append(Attempt(v.sql, "execution", str(e)))
                    turns += [("model", resp.raw_text),
                              ("user", prompts.repair_prompt(question, v.sql, "execution", str(e), info.dialect))]
                    result.error = str(e)
                    continue
                self.audit.log("dry_run", request_id=rid, sql=v.sql, affected_rows=preview.affected_rows)
                if preview.preview is not None and v.target_table:
                    semantics = self.source.semantics
                    preview.preview = masking.protect_table(preview.preview, v.target_table,
                                                            semantics.sensitive_set(), semantics.masking)
                result.status = "needs_confirmation"
                result.write_preview = preview
                result.target_table = v.target_table or ""
                result.error = ""
                break

            # ---- 3b. Read: execute ---------------------------------------- #
            stage("Running the query")
            try:
                qr = self.source.run_read(v)
            except (ExecutionError, DataSourceError) as e:
                self.audit.log("execute", request_id=rid, status="failed", sql=v.sql, error=str(e))
                result.failed_attempts.append(Attempt(v.sql, "execution", str(e)))
                turns += [("model", resp.raw_text),
                          ("user", prompts.repair_prompt(question, v.sql, "execution", str(e), info.dialect))]
                result.error = str(e)
                continue

            self.audit.log("execute", request_id=rid, rows=len(qr.df), elapsed_ms=round(qr.elapsed_ms, 1))
            guarded = self.guard(qr.df, v)
            result.masked_columns = guarded.masked_columns
            result.notes += guarded.notes(self.source.semantics.masking)
            if guarded.masked_columns:
                self.audit.log("masked", request_id=rid, columns=guarded.masked_columns,
                               traced=guarded.traced, mode=self.source.semantics.masking)
            qr.df = guarded.df
            result.df = qr.df
            result.row_count = len(qr.df)
            result.execution_ms = round(qr.elapsed_ms, 1)
            result.chart = resolve_spec(qr.df, data)
            if result.chart.adjusted:
                result.notes.append(result.chart.adjusted)
            if qr.truncated:
                result.notes.append("Results were truncated; refine the question to see fewer rows.")
            result.status, result.error = "answer", ""
            break
        else:
            result.status = "error"
            result.error = f"Couldn't produce a working query after {MAX_REPAIRS + 1} tries. Last problem: {result.error}"

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        self.audit.log("result", request_id=rid, status=result.status, error=result.error or None,
                       llm_calls=result.llm_calls, repairs=len(result.failed_attempts),
                       elapsed_ms=round(result.elapsed_ms))
        return result

    # ------------------------------------------------------------------ #
    def guard(self, df: pd.DataFrame, v) -> masking.GuardResult:
        """Hide sensitive values in a result and cut over-long cells (see agent/masking.py)."""
        semantics, info = self.source.semantics, self.source.info
        return masking.protect(df, v.sql, self.schema, semantics.sensitive_set(), semantics.masking,
                               info.dialect, flagged=bool(v.sensitive_columns))

    def confirm(self, pending: AgentResult) -> AgentResult:
        exp_rows = pending.write_preview.affected_rows if pending.write_preview else None
        # Validated again just before it runs: the SQL that reaches the database is always
        # the SQL that passed the checks, against the structure as it is now.
        try:
            checked = validate_sql(pending.sql, self.schema, allow_writes=True,
                                   max_rows=self.source.info.limits.max_rows, dialect=self.source.info.dialect)
            if checked.kind == READ:
                raise ValidationError("This is not a change.", policy=True)
        except ValidationError as e:
            self.audit.log("commit", request_id=pending.request_id, status="blocked", sql=pending.sql, error=str(e))
            pending.status, pending.error = "blocked", f"The change was checked again and refused: {e}"
            return pending
        try:
            n = self.source.commit_write(pending.sql, expected_rows=exp_rows)
        except (ExecutionError, DataSourceError) as e:
            self.audit.log("commit", request_id=pending.request_id, status="failed", sql=pending.sql, error=str(e))
            pending.status, pending.error = "error", str(e)
            return pending
        self.audit.log("commit", request_id=pending.request_id, sql=pending.sql, affected_rows=n,
                       confirmed_by="user")
        pending.status = "committed"
        pending.notes.append(f"{n} row{'s' if n != 1 else ''} changed in '{pending.target_table}'.")
        self.refresh_schema()
        return pending

    def cancel(self, pending: AgentResult) -> AgentResult:
        self.audit.log("commit", request_id=pending.request_id, status="cancelled", sql=pending.sql,
                       confirmed_by="user")
        pending.status = "cancelled"
        return pending
