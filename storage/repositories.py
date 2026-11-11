"""Reading and writing records. One repository per kind of record.

Isolation rule: every method that touches workspace data takes the
`workspace_id` (or, for workspaces themselves, the `owner_id`) and includes it
in the SQL. A record that belongs to someone else is indistinguishable from one
that doesn't exist: lookups return None and updates change nothing. Callers
cannot reach across workspaces by guessing an id.

All values are passed as parameters, never formatted into SQL text.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .database import MetadataDB
from .models import (Conversation, DataSourceRecord, MemoryItem, Message, QueryRun, SavedItem, Tag, Workspace,
                     now)

DEFAULT_WORKSPACE_NAME = "My workspace"


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _load(text: str | None) -> dict:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------- #
# Workspaces
# --------------------------------------------------------------------------- #
class WorkspaceRepository:
    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Workspace:
        return Workspace(id=r["id"], name=r["name"], description=r["description"], owner_id=r["owner_id"],
                         settings=_load(r["settings"]), created_at=r["created_at"], updated_at=r["updated_at"])

    def create(self, owner_id: str, name: str, description: str = "") -> Workspace:
        workspace = Workspace(name=name.strip()[:80] or DEFAULT_WORKSPACE_NAME, owner_id=owner_id,
                              description=description.strip()[:500])
        with self.db.connect() as conn:
            conn.execute("INSERT INTO workspaces (id, name, description, owner_id, created_at, updated_at) "
                         "VALUES (?, ?, ?, ?, ?, ?)",
                         (workspace.id, workspace.name, workspace.description, owner_id,
                          workspace.created_at, workspace.updated_at))
        return workspace

    def list(self, owner_id: str) -> list[Workspace]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM workspaces WHERE owner_id = ? ORDER BY created_at", (owner_id,))
            return [self._row(r) for r in rows]

    def get(self, owner_id: str, workspace_id: str) -> Workspace | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE id = ? AND owner_id = ?",
                               (workspace_id, owner_id)).fetchone()
        return self._row(row) if row else None

    def ensure_default(self, owner_id: str) -> Workspace:
        """The owner's first workspace, created on their first visit."""
        existing = self.list(owner_id)
        return existing[0] if existing else self.create(owner_id, DEFAULT_WORKSPACE_NAME)

    def rename(self, owner_id: str, workspace_id: str, name: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("UPDATE workspaces SET name = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
                               (name.strip()[:80], now(), workspace_id, owner_id))
            return cur.rowcount == 1

    def update_settings(self, owner_id: str, workspace_id: str, **changes: Any) -> dict[str, Any]:
        """Merge `changes` into the workspace's settings and return the result."""
        with self.db.transaction() as conn:
            row = conn.execute("SELECT settings FROM workspaces WHERE id = ? AND owner_id = ?",
                               (workspace_id, owner_id)).fetchone()
            if row is None:
                raise LookupError("That workspace doesn't exist.")
            settings = {**_load(row["settings"]), **changes}
            conn.execute("UPDATE workspaces SET settings = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
                         (_json(settings), now(), workspace_id, owner_id))
        return settings

    def delete(self, owner_id: str, workspace_id: str) -> bool:
        """Remove a workspace and everything in it (data-source records, conversations, messages, memory)."""
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM workspaces WHERE id = ? AND owner_id = ?", (workspace_id, owner_id))
            return cur.rowcount == 1


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
class DataSourceRepository:
    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> DataSourceRecord:
        return DataSourceRecord(
            id=r["id"], workspace_id=r["workspace_id"], name=r["name"], source_type=r["source_type"],
            dialect=r["dialect"], description=r["description"], status=r["status"],
            connection=_load(r["connection"]), semantic_config=_load(r["semantic_config"]),
            writable=bool(r["writable"]), is_sample=bool(r["is_sample"]),
            last_schema_refresh=r["last_schema_refresh"], created_at=r["created_at"], updated_at=r["updated_at"])

    def save(self, record: DataSourceRecord) -> DataSourceRecord:
        """Insert, or update the existing row with the same id in the same workspace."""
        record.updated_at = now()
        with self.db.connect() as conn:
            conn.execute(
                """INSERT INTO data_sources (id, workspace_id, name, source_type, dialect, description, status,
                       connection, semantic_config, writable, is_sample, last_schema_refresh, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (id) DO UPDATE SET
                       name = excluded.name, description = excluded.description, status = excluded.status,
                       connection = excluded.connection, semantic_config = excluded.semantic_config,
                       writable = excluded.writable, last_schema_refresh = excluded.last_schema_refresh,
                       updated_at = excluded.updated_at
                   WHERE data_sources.workspace_id = excluded.workspace_id""",
                (record.id, record.workspace_id, record.name, record.source_type, record.dialect,
                 record.description, record.status, _json(record.connection), _json(record.semantic_config),
                 int(record.writable), int(record.is_sample), record.last_schema_refresh,
                 record.created_at, record.updated_at))
        return record

    def list(self, workspace_id: str) -> list[DataSourceRecord]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM data_sources WHERE workspace_id = ? ORDER BY created_at",
                                (workspace_id,))
            return [self._row(r) for r in rows]

    def get(self, workspace_id: str, source_id: str) -> DataSourceRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM data_sources WHERE id = ? AND workspace_id = ?",
                               (source_id, workspace_id)).fetchone()
        return self._row(row) if row else None

    def save_semantics(self, workspace_id: str, source_id: str, config: dict) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("UPDATE data_sources SET semantic_config = ?, updated_at = ? "
                               "WHERE id = ? AND workspace_id = ?", (_json(config), now(), source_id, workspace_id))
            return cur.rowcount == 1

    def delete(self, workspace_id: str, source_id: str) -> bool:
        """Remove the record. Conversations about this data source are removed with it."""
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM data_sources WHERE id = ? AND workspace_id = ?",
                               (source_id, workspace_id))
            return cur.rowcount == 1


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
class ConversationRepository:
    EDITABLE = {"title", "summary", "category", "archived", "pinned"}

    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Conversation:
        return Conversation(
            id=r["id"], workspace_id=r["workspace_id"], data_source_id=r["data_source_id"], title=r["title"],
            summary=r["summary"], category=r["category"], archived=bool(r["archived"]), pinned=bool(r["pinned"]),
            created_at=r["created_at"], updated_at=r["updated_at"], last_message_at=r["last_message_at"])

    def create(self, workspace_id: str, title: str, data_source_id: str | None = None,
               category: str = "") -> Conversation:
        conversation = Conversation(workspace_id=workspace_id, title=title.strip()[:120] or "New conversation",
                                    data_source_id=data_source_id, category=category.strip()[:40])
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, workspace_id, data_source_id, title, category, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation.id, workspace_id, data_source_id, conversation.title, conversation.category,
                 conversation.created_at, conversation.updated_at))
        return conversation

    # -- search and filters ------------------------------------------------- #
    def has_search_index(self) -> bool:
        """Whether the FTS5 index exists (it is optional; see migration 2)."""
        with self.db.connect() as conn:
            return conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'message_search'").fetchone() is not None

    def _text_hits(self, conn: sqlite3.Connection, workspace_id: str, words: list[str],
                   match_any: bool) -> dict[str, tuple[float, str]]:
        """conversation id -> (rank, snippet) for conversations whose visible messages mention the words.

        Uses the full-text index when it exists, otherwise plain LIKE. The words are
        passed as parameters; in the full-text query each one is quoted, so user
        text can never act as search syntax.
        """
        hits: dict[str, tuple[float, str]] = {}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'message_search'").fetchone():
            query = (" OR " if match_any else " ").join('"' + w.replace('"', '""') + '"*' for w in words)
            rows = conn.execute(
                "SELECT message_search.conversation_id AS cid, bm25(message_search) AS score, "
                "snippet(message_search, 0, '', '', ' ... ', 14) AS snip "
                "FROM message_search JOIN messages m ON m.id = message_search.message_id AND m.active = 1 "
                "WHERE message_search MATCH ? AND message_search.workspace_id = ? ORDER BY score LIMIT 300",
                (query, workspace_id))
            for r in rows:
                hits.setdefault(r["cid"], (r["score"], r["snip"]))
        else:
            joiner = " OR " if match_any else " AND "
            clause = joiner.join("(m.content LIKE ? ESCAPE '\\' OR m.sql LIKE ? ESCAPE '\\')" for _ in words)
            args: list[Any] = []
            for w in words:
                pattern = "%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                args += [pattern, pattern]
            rows = conn.execute(
                "SELECT m.conversation_id AS cid, m.content AS snip FROM messages m "
                "JOIN conversations c ON c.id = m.conversation_id "
                f"WHERE c.workspace_id = ? AND m.active = 1 AND ({clause}) ORDER BY m.position LIMIT 300",
                [workspace_id] + args)
            for r in rows:
                hits.setdefault(r["cid"], (0.0, r["snip"][:120]))
        return hits

    def find(self, workspace_id: str, *, text: str = "", match_any: bool = False, category: str | None = None,
             data_source_id: str | None = None, tag_id: str | None = None, since: str | None = None,
             until: str | None = None, archived: bool = False, exclude_id: str | None = None,
             limit: int = 50) -> list[tuple[Conversation, str]]:
        """Search and filter this workspace's conversations. Returns (conversation, matching text) pairs.

        text:          words to look for in titles, questions, answers and SQL
        match_any:     True ranks conversations that mention any of the words (used for "related");
                       False requires all of them (used for the search box)
        since / until: UTC timestamps in the stored format (see models.now); last activity must be
                       at or after `since` and before `until`. Callers convert calendar days to
                       instants first (services.day_bounds), because a local day is not a UTC day.
        archived:      True lists only archived conversations, False only current ones
        """
        words = re.findall(r"[^\W_]+", text or "", flags=re.UNICODE)[:12]
        sql = "SELECT * FROM conversations c WHERE c.workspace_id = ? AND c.archived = ?"
        args: list[Any] = [workspace_id, int(archived)]
        if category is not None:
            sql += " AND c.category = ?"
            args.append(category)
        if data_source_id is not None:
            sql += " AND c.data_source_id = ?"
            args.append(data_source_id)
        if tag_id is not None:
            sql += " AND EXISTS (SELECT 1 FROM conversation_tags ct WHERE ct.conversation_id = c.id AND ct.tag_id = ?)"
            args.append(tag_id)
        if since:
            sql += " AND COALESCE(c.last_message_at, c.created_at) >= ?"
            args.append(since)
        if until:
            sql += " AND COALESCE(c.last_message_at, c.created_at) < ?"
            args.append(until)
        if exclude_id is not None:
            sql += " AND c.id != ?"
            args.append(exclude_id)
        sql += " ORDER BY c.pinned DESC, COALESCE(c.last_message_at, c.created_at) DESC"

        with self.db.connect() as conn:
            candidates = [self._row(r) for r in conn.execute(sql, args)]
            if not words:
                return [(c, "") for c in candidates[:limit]]
            hits = self._text_hits(conn, workspace_id, words, match_any)

        results: list[tuple[float, Conversation, str]] = []
        for c in candidates:
            title = c.title.lower()
            in_title = [w for w in words if w.lower() in title]
            title_match = bool(in_title) if match_any else len(in_title) == len(words)
            if c.id in hits:
                score, snippet = hits[c.id]
                results.append((score - (1.0 if title_match else 0.0), c, snippet))
            elif title_match:
                results.append((-0.5, c, c.title))
        results.sort(key=lambda item: item[0])              # bm25: lower is a better match
        return [(c, snippet) for _, c, snippet in results[:limit]]

    def categories(self, workspace_id: str) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT DISTINCT category FROM conversations WHERE workspace_id = ? "
                                "AND category != '' ORDER BY category", (workspace_id,))
            return [r[0] for r in rows]

    def get(self, workspace_id: str, conversation_id: str) -> Conversation | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM conversations WHERE id = ? AND workspace_id = ?",
                               (conversation_id, workspace_id)).fetchone()
        return self._row(row) if row else None

    def list(self, workspace_id: str, *, data_source_id: str | None = None, include_archived: bool = False,
             only_archived: bool = False, limit: int = 50) -> list[Conversation]:
        """Pinned first, then most recently active."""
        sql = "SELECT * FROM conversations WHERE workspace_id = ?"
        args: list[Any] = [workspace_id]
        if data_source_id is not None:
            sql += " AND data_source_id = ?"
            args.append(data_source_id)
        if only_archived:
            sql += " AND archived = 1"
        elif not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY pinned DESC, COALESCE(last_message_at, created_at) DESC LIMIT ?"
        args.append(int(limit))
        with self.db.connect() as conn:
            return [self._row(r) for r in conn.execute(sql, args)]

    def update(self, workspace_id: str, conversation_id: str, **changes: Any) -> bool:
        """Change title, summary, category, archived or pinned. Other fields are rejected."""
        unknown = set(changes) - self.EDITABLE
        if unknown:
            raise ValueError(f"These conversation fields can't be edited: {sorted(unknown)}")
        if not changes:
            return False
        if "title" in changes:
            changes["title"] = str(changes["title"]).strip()[:120] or "Untitled"
        for flag in ("archived", "pinned"):
            if flag in changes:
                changes[flag] = int(bool(changes[flag]))
        columns = sorted(changes)              # names come from EDITABLE, never from the caller's text
        assignments = ", ".join(f"{c} = ?" for c in columns) + ", updated_at = ?"
        with self.db.connect() as conn:
            cur = conn.execute(f"UPDATE conversations SET {assignments} WHERE id = ? AND workspace_id = ?",
                               [changes[c] for c in columns] + [now(), conversation_id, workspace_id])
            return cur.rowcount == 1

    def delete(self, workspace_id: str, conversation_id: str) -> bool:
        """Permanently remove a conversation with its messages and query runs."""
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ? AND workspace_id = ?",
                               (conversation_id, workspace_id))
            return cur.rowcount == 1

    def count(self, workspace_id: str, data_source_id: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) FROM conversations WHERE workspace_id = ?", [workspace_id]
        if data_source_id is not None:
            sql += " AND data_source_id = ?"
            args.append(data_source_id)
        with self.db.connect() as conn:
            return int(conn.execute(sql, args).fetchone()[0])


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
class MessageRepository:
    EDITABLE = {"content", "status", "explanation", "chart_spec", "result_metadata", "active"}
    # Every statement checks, through the conversation, that the message is in the caller's workspace
    _OWNED = "conversation_id IN (SELECT id FROM conversations WHERE workspace_id = ?)"

    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Message:
        return Message(
            id=r["id"], conversation_id=r["conversation_id"], parent_message_id=r["parent_message_id"],
            role=r["role"], content=r["content"], status=r["status"], intent=r["intent"], sql=r["sql"],
            explanation=r["explanation"], reasoning_summary=r["reasoning_summary"],
            chart_spec=_load(r["chart_spec"]), result_metadata=_load(r["result_metadata"]),
            request_id=r["request_id"], model=r["model"], position=r["position"], created_at=r["created_at"],
            active=bool(r["active"]))

    def add(self, workspace_id: str, message: Message) -> Message:
        """Append to the end of its conversation. Raises LookupError if that conversation isn't in the workspace."""
        with self.db.transaction() as conn:
            owned = conn.execute("SELECT 1 FROM conversations WHERE id = ? AND workspace_id = ?",
                                 (message.conversation_id, workspace_id)).fetchone()
            if not owned:
                raise LookupError("That conversation doesn't exist in this workspace.")
            message.position = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM messages WHERE conversation_id = ?",
                (message.conversation_id,)).fetchone()[0]
            conn.execute(
                """INSERT INTO messages (id, conversation_id, parent_message_id, role, content, status, intent, sql,
                       explanation, reasoning_summary, chart_spec, result_metadata, request_id, model, position,
                       created_at, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message.id, message.conversation_id, message.parent_message_id, message.role, message.content,
                 message.status, message.intent, message.sql, message.explanation, message.reasoning_summary,
                 _json(message.chart_spec), _json(message.result_metadata), message.request_id, message.model,
                 message.position, message.created_at, int(message.active)))
            conn.execute("UPDATE conversations SET last_message_at = ?, updated_at = ? WHERE id = ?",
                         (message.created_at, message.created_at, message.conversation_id))
        return message

    def list(self, workspace_id: str, conversation_id: str, *, include_inactive: bool = False) -> list[Message]:
        """Messages in order. Versions replaced by a regenerate or an edit are left out unless asked for."""
        visible = "" if include_inactive else "AND active = 1 "
        with self.db.connect() as conn:
            rows = conn.execute(f"SELECT * FROM messages WHERE conversation_id = ? AND {self._OWNED} "
                                f"{visible}ORDER BY position", (conversation_id, workspace_id))
            return [self._row(r) for r in rows]

    def deactivate(self, workspace_id: str, message_ids: list[str]) -> int:
        """Hide messages (they stay stored). Returns how many were hidden."""
        hidden = 0
        with self.db.transaction() as conn:
            for message_id in message_ids:
                cur = conn.execute(f"UPDATE messages SET active = 0 WHERE id = ? AND active = 1 AND {self._OWNED}",
                                   (message_id, workspace_id))
                hidden += cur.rowcount
        return hidden

    def get(self, workspace_id: str, message_id: str) -> Message | None:
        with self.db.connect() as conn:
            row = conn.execute(f"SELECT * FROM messages WHERE id = ? AND {self._OWNED}",
                               (message_id, workspace_id)).fetchone()
        return self._row(row) if row else None

    def update(self, workspace_id: str, message_id: str, **changes: Any) -> bool:
        unknown = set(changes) - self.EDITABLE
        if unknown:
            raise ValueError(f"These message fields can't be edited: {sorted(unknown)}")
        if not changes:
            return False
        for key in ("chart_spec", "result_metadata"):
            if key in changes:
                changes[key] = _json(changes[key])
        if "active" in changes:
            changes["active"] = int(bool(changes["active"]))
        columns = sorted(changes)
        assignments = ", ".join(f"{c} = ?" for c in columns)
        with self.db.connect() as conn:
            cur = conn.execute(f"UPDATE messages SET {assignments} WHERE id = ? AND {self._OWNED}",
                               [changes[c] for c in columns] + [message_id, workspace_id])
            return cur.rowcount == 1


# --------------------------------------------------------------------------- #
# Query runs
# --------------------------------------------------------------------------- #
class QueryRunRepository:
    _OWNED = ("message_id IN (SELECT m.id FROM messages m JOIN conversations c ON c.id = m.conversation_id "
              "WHERE c.workspace_id = ?)")

    def __init__(self, db: MetadataDB):
        self.db = db

    def add(self, workspace_id: str, run: QueryRun) -> QueryRun:
        with self.db.transaction() as conn:
            owned = conn.execute(
                "SELECT 1 FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ? AND c.workspace_id = ?", (run.message_id, workspace_id)).fetchone()
            if not owned:
                raise LookupError("That message doesn't exist in this workspace.")
            conn.execute(
                "INSERT INTO query_runs (id, message_id, sql, validation_status, execution_status, row_count, "
                "execution_time, error_message, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run.id, run.message_id, run.sql, run.validation_status, run.execution_status, run.row_count,
                 run.execution_time, run.error_message[:1000], run.created_at))
        return run

    def list(self, workspace_id: str, message_id: str) -> list[QueryRun]:
        with self.db.connect() as conn:
            rows = conn.execute(f"SELECT * FROM query_runs WHERE message_id = ? AND {self._OWNED} "
                                "ORDER BY created_at, rowid", (message_id, workspace_id))
            return [QueryRun(id=r["id"], message_id=r["message_id"], sql=r["sql"],
                             validation_status=r["validation_status"], execution_status=r["execution_status"],
                             row_count=r["row_count"], execution_time=r["execution_time"],
                             error_message=r["error_message"], created_at=r["created_at"]) for r in rows]


# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #
class TagRepository:
    def __init__(self, db: MetadataDB):
        self.db = db

    def create(self, workspace_id: str, name: str, color: str = "") -> Tag:
        """Return the workspace's tag with this name, creating it if needed."""
        name = name.strip()[:40]
        if not name:
            raise ValueError("A tag needs a name.")
        tag = Tag(workspace_id=workspace_id, name=name, color=color.strip()[:20])
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id, color FROM tags WHERE workspace_id = ? AND name = ?",
                               (workspace_id, name)).fetchone()
            if row:
                return Tag(workspace_id=workspace_id, name=name, color=row["color"], id=row["id"])
            conn.execute("INSERT INTO tags (id, workspace_id, name, color) VALUES (?, ?, ?, ?)",
                         (tag.id, workspace_id, tag.name, tag.color))
        return tag

    def list(self, workspace_id: str) -> list[Tag]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM tags WHERE workspace_id = ? ORDER BY name", (workspace_id,))
            return [Tag(id=r["id"], workspace_id=r["workspace_id"], name=r["name"], color=r["color"]) for r in rows]

    def assign(self, workspace_id: str, conversation_id: str, tag_id: str) -> bool:
        """Tag a conversation. Both must belong to the workspace."""
        with self.db.connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO conversation_tags (conversation_id, tag_id) "
                "SELECT c.id, t.id FROM conversations c, tags t "
                "WHERE c.id = ? AND t.id = ? AND c.workspace_id = ? AND t.workspace_id = ?",
                (conversation_id, tag_id, workspace_id, workspace_id))
            return cur.rowcount == 1

    def unassign(self, workspace_id: str, conversation_id: str, tag_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversation_tags WHERE conversation_id = ? AND tag_id = ? AND conversation_id IN "
                "(SELECT id FROM conversations WHERE workspace_id = ?)", (conversation_id, tag_id, workspace_id))
            return cur.rowcount == 1

    def for_conversation(self, workspace_id: str, conversation_id: str) -> list[Tag]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT t.* FROM tags t JOIN conversation_tags ct ON ct.tag_id = t.id "
                "WHERE ct.conversation_id = ? AND t.workspace_id = ? ORDER BY t.name", (conversation_id, workspace_id))
            return [Tag(id=r["id"], workspace_id=r["workspace_id"], name=r["name"], color=r["color"]) for r in rows]


# --------------------------------------------------------------------------- #
# Saved items (saved results, bookmarked charts and queries)
# --------------------------------------------------------------------------- #
class SavedItemRepository:
    KINDS = ("result", "chart", "query")

    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> SavedItem:
        return SavedItem(id=r["id"], workspace_id=r["workspace_id"], conversation_id=r["conversation_id"],
                         message_id=r["message_id"], kind=r["kind"], title=r["title"], note=r["note"],
                         created_at=r["created_at"])

    def add(self, workspace_id: str, message_id: str, kind: str, title: str, note: str = "") -> SavedItem:
        """Keep a pointer to an answer. The message must belong to the workspace."""
        if kind not in self.KINDS:
            raise ValueError(f"A saved item must be one of: {', '.join(self.KINDS)}.")
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT m.conversation_id FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ? AND c.workspace_id = ?", (message_id, workspace_id)).fetchone()
            if not row:
                raise LookupError("That answer doesn't exist in this workspace.")
            item = SavedItem(workspace_id=workspace_id, conversation_id=row["conversation_id"],
                             message_id=message_id, kind=kind, title=title.strip()[:120] or "Saved item",
                             note=note.strip()[:500])
            conn.execute("INSERT INTO saved_items (id, workspace_id, conversation_id, message_id, kind, title, "
                         "note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (item.id, workspace_id, item.conversation_id, message_id, kind, item.title, item.note,
                          item.created_at))
        return item

    def list(self, workspace_id: str, kind: str | None = None) -> list[SavedItem]:
        sql, args = "SELECT * FROM saved_items WHERE workspace_id = ?", [workspace_id]
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        with self.db.connect() as conn:
            return [self._row(r) for r in conn.execute(sql + " ORDER BY created_at DESC, rowid DESC", args)]

    def delete(self, workspace_id: str, item_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM saved_items WHERE id = ? AND workspace_id = ?", (item_id, workspace_id))
            return cur.rowcount == 1


# --------------------------------------------------------------------------- #
# Long-term memory
# --------------------------------------------------------------------------- #
class MemoryRepository:
    EDITABLE = {"content", "memory_type", "importance", "confidence", "user_approved", "status", "pinned",
                "data_source_id", "last_used_at"}
    STATUSES = ("suggested", "active", "rejected")

    def __init__(self, db: MetadataDB):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"], workspace_id=r["workspace_id"], conversation_id=r["conversation_id"],
            memory_type=r["memory_type"], content=r["content"], source_message_id=r["source_message_id"],
            data_source_id=r["data_source_id"], importance=r["importance"], confidence=r["confidence"],
            user_approved=bool(r["user_approved"]), status=r["status"], pinned=bool(r["pinned"]),
            last_used_at=r["last_used_at"], created_at=r["created_at"], updated_at=r["updated_at"])

    @staticmethod
    def _check_scope(conn: sqlite3.Connection, workspace_id: str, item: MemoryItem) -> None:
        """Everything a memory points at must be in the same workspace."""
        if item.data_source_id and not conn.execute(
                "SELECT 1 FROM data_sources WHERE id = ? AND workspace_id = ?",
                (item.data_source_id, workspace_id)).fetchone():
            raise LookupError("That data source doesn't exist in this workspace.")
        if item.conversation_id and not conn.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND workspace_id = ?",
                (item.conversation_id, workspace_id)).fetchone():
            raise LookupError("That conversation doesn't exist in this workspace.")
        if item.source_message_id and not conn.execute(
                "SELECT 1 FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ? AND c.workspace_id = ?", (item.source_message_id, workspace_id)).fetchone():
            raise LookupError("That message doesn't exist in this workspace.")

    def add(self, workspace_id: str, item: MemoryItem) -> MemoryItem:
        if item.status not in self.STATUSES:
            raise ValueError(f"A memory's status must be one of: {', '.join(self.STATUSES)}.")
        item.workspace_id = workspace_id
        with self.db.transaction() as conn:
            self._check_scope(conn, workspace_id, item)
            conn.execute(
                """INSERT INTO memory_items (id, workspace_id, conversation_id, memory_type, content,
                       source_message_id, importance, confidence, user_approved, created_at, updated_at,
                       data_source_id, status, pinned, last_used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (item.id, workspace_id, item.conversation_id, item.memory_type, item.content,
                 item.source_message_id, item.importance, item.confidence, int(item.user_approved),
                 item.created_at, item.updated_at, item.data_source_id, item.status, int(item.pinned),
                 item.last_used_at))
        return item

    def get(self, workspace_id: str, memory_id: str) -> MemoryItem | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM memory_items WHERE id = ? AND workspace_id = ?",
                               (memory_id, workspace_id)).fetchone()
        return self._row(row) if row else None

    def list(self, workspace_id: str, *, status: str | None = None, conversation_id: str | None = None,
             for_data_source: str | None = None) -> list[MemoryItem]:
        """This workspace's memories, pinned first, then newest.

        for_data_source: keep memories that apply to that data source, meaning those
        scoped to it and those scoped to the whole workspace.
        """
        sql, args = "SELECT * FROM memory_items WHERE workspace_id = ?", [workspace_id]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            args.append(conversation_id)
        if for_data_source is not None:
            sql += " AND (data_source_id IS NULL OR data_source_id = ?)"
            args.append(for_data_source)
        sql += " ORDER BY pinned DESC, updated_at DESC, rowid DESC"
        with self.db.connect() as conn:
            return [self._row(r) for r in conn.execute(sql, args)]

    def update(self, workspace_id: str, memory_id: str, **changes: Any) -> bool:
        unknown = set(changes) - self.EDITABLE
        if unknown:
            raise ValueError(f"These memory fields can't be edited: {sorted(unknown)}")
        if not changes:
            return False
        if "status" in changes and changes["status"] not in self.STATUSES:
            raise ValueError(f"A memory's status must be one of: {', '.join(self.STATUSES)}.")
        for flag in ("user_approved", "pinned"):
            if flag in changes:
                changes[flag] = int(bool(changes[flag]))
        touch_only = set(changes) == {"last_used_at"}
        columns = sorted(changes)
        assignments = ", ".join(f"{c} = ?" for c in columns) + ("" if touch_only else ", updated_at = ?")
        values = [changes[c] for c in columns] + ([] if touch_only else [now()])
        with self.db.transaction() as conn:
            if changes.get("data_source_id"):
                if not conn.execute("SELECT 1 FROM data_sources WHERE id = ? AND workspace_id = ?",
                                    (changes["data_source_id"], workspace_id)).fetchone():
                    raise LookupError("That data source doesn't exist in this workspace.")
            cur = conn.execute(f"UPDATE memory_items SET {assignments} WHERE id = ? AND workspace_id = ?",
                               values + [memory_id, workspace_id])
            return cur.rowcount == 1

    def delete(self, workspace_id: str, memory_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM memory_items WHERE id = ? AND workspace_id = ?",
                               (memory_id, workspace_id))
            return cur.rowcount == 1

    def delete_for_conversation(self, workspace_id: str, conversation_id: str) -> int:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM memory_items WHERE workspace_id = ? AND conversation_id = ?",
                               (workspace_id, conversation_id))
            return cur.rowcount

    def delete_all(self, workspace_id: str) -> int:
        with self.db.connect() as conn:
            return conn.execute("DELETE FROM memory_items WHERE workspace_id = ?", (workspace_id,)).rowcount

    def search(self, workspace_id: str, words: list[str]) -> dict[str, float]:
        """memory id -> rank (0 is best) for this workspace's memories that mention any of the words.

        Full-text index when present, plain LIKE otherwise. Words are quoted or
        escaped, so text from a question can't act as search syntax.
        """
        words = [w for w in words if w][:16]
        if not words:
            return {}
        with self.db.connect() as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'memory_search'").fetchone():
                query = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)
                rows = conn.execute(
                    "SELECT memory_id FROM memory_search WHERE memory_search MATCH ? AND workspace_id = ? "
                    "ORDER BY bm25(memory_search) LIMIT 100", (query, workspace_id)).fetchall()
            else:
                clause = " OR ".join("content LIKE ? ESCAPE '\\'" for _ in words)
                patterns = ["%" + w.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                            for w in words]
                rows = conn.execute(f"SELECT id AS memory_id FROM memory_items WHERE workspace_id = ? AND ({clause}) "
                                    "LIMIT 100", [workspace_id] + patterns).fetchall()
        return {r["memory_id"]: float(position) for position, r in enumerate(rows)}

    def recent_questions(self, workspace_id: str, data_source_id: str, limit: int = 200) -> list[str]:
        """The latest visible questions asked about a data source (used to notice habits)."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT m.content FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE c.workspace_id = ? AND c.data_source_id = ? AND m.role = 'user' AND m.active = 1 "
                "ORDER BY m.created_at DESC, m.rowid DESC LIMIT ?", (workspace_id, data_source_id, int(limit)))
            return [r[0] for r in rows]
