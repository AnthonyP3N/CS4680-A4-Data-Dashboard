"""Records kept in the metadata database. Plain data, no behaviour.

Ids are random text, and timestamps are ISO-8601 UTC text, so the same shapes
work unchanged on PostgreSQL later.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Workspace:
    name: str
    owner_id: str
    description: str = ""
    settings: dict[str, Any] = field(default_factory=dict)   # e.g. {"long_term_memory": False}
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)


@dataclass
class DataSourceRecord:
    workspace_id: str
    name: str
    source_type: str                 # "sqlite", "csv", ...
    dialect: str
    description: str = ""
    status: str = "active"
    connection: dict[str, Any] = field(default_factory=dict)        # non-secret details only
    semantic_config: dict[str, Any] = field(default_factory=dict)   # the business context, as JSON
    writable: bool = False
    is_sample: bool = False
    last_schema_refresh: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)


@dataclass
class Conversation:
    workspace_id: str
    title: str
    data_source_id: str | None = None
    summary: str = ""
    category: str = ""
    archived: bool = False
    pinned: bool = False
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    last_message_at: str | None = None


@dataclass
class Message:
    conversation_id: str
    role: str                        # "user" or "assistant"
    content: str = ""
    parent_message_id: str | None = None
    status: str = ""                 # assistant only: answer, clarify, needs_confirmation, committed, ...
    intent: str = ""                 # assistant only: query, modify, clarify, unsupported
    sql: str = ""
    explanation: str = ""
    reasoning_summary: str = ""
    chart_spec: dict[str, Any] = field(default_factory=dict)
    result_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    model: str = ""
    position: int = 0                # order within the conversation, assigned on insert
    active: bool = True              # False for versions replaced by a regenerate or an edit
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)


@dataclass
class QueryRun:
    message_id: str
    sql: str
    validation_status: str           # passed, failed, blocked
    execution_status: str            # ok, failed, dry_run, committed, cancelled, not_run
    row_count: int | None = None
    execution_time: float | None = None    # milliseconds
    error_message: str = ""
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)


@dataclass
class Tag:
    workspace_id: str
    name: str
    color: str = ""
    id: str = field(default_factory=new_id)


@dataclass
class MemoryItem:
    """One thing remembered for a workspace: a preference, definition, default filter and so on."""
    workspace_id: str
    memory_type: str
    content: str
    conversation_id: str | None = None      # the conversation it came from, if any
    source_message_id: str | None = None    # the question it came from, if any
    data_source_id: str | None = None       # None = applies to every data source in the workspace
    importance: float = 0.5
    confidence: float = 0.5                 # how sure the suggester was (1.0 when the user wrote it)
    user_approved: bool = False
    status: str = "active"                  # "suggested", "active" or "rejected"
    pinned: bool = False
    last_used_at: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)


@dataclass
class SavedItem:
    """A result, chart or query the user chose to keep a pointer to."""
    workspace_id: str
    conversation_id: str
    message_id: str
    kind: str                        # "result", "chart" or "query"
    title: str
    note: str = ""
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=now)
