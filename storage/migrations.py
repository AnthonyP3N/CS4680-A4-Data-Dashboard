"""Versioned changes to the metadata database.

The database structure is only ever changed by adding a migration to
MIGRATIONS. Each one runs once, inside a transaction, and is recorded in the
`schema_migrations` table. To change the structure:

    1. Append Migration(<next number>, "<what it does>", (<SQL statements>...)).
    2. Never edit or reorder a migration that has shipped.

On start-up `migrate()` applies whatever is pending. Before upgrading a
database that already holds data it writes a backup copy beside it. If a
migration fails it is rolled back and the database stays at its old version.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class MigrationError(Exception):
    """The metadata database could not be brought to the version this app needs."""


@dataclass(frozen=True)
class Migration:
    """One structural change.

    statements: must all succeed, or the whole migration is rolled back.
    optional:   a group that is applied only if every statement in it works on
                this SQLite build (used for the FTS5 search index). If any
                fails, the group is undone and the migration still counts as
                applied; the feature that needs it falls back at runtime.
    """
    version: int
    name: str
    statements: tuple[str, ...]
    optional: tuple[str, ...] = ()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "workspaces, data sources, conversations, messages, memory, tags, query runs", (
        """CREATE TABLE workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_workspaces_owner ON workspaces(owner_id)",

        """CREATE TABLE data_sources (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            dialect TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            connection TEXT NOT NULL DEFAULT '{}',
            semantic_config TEXT NOT NULL DEFAULT '{}',
            writable INTEGER NOT NULL DEFAULT 0,
            is_sample INTEGER NOT NULL DEFAULT 0,
            last_schema_refresh TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_data_sources_workspace ON data_sources(workspace_id)",

        """CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            data_source_id TEXT REFERENCES data_sources(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT '',
            archived INTEGER NOT NULL DEFAULT 0,
            pinned INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT
        )""",
        "CREATE INDEX idx_conversations_workspace ON conversations(workspace_id, last_message_at)",

        """CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            parent_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            intent TEXT NOT NULL DEFAULT '',
            sql TEXT NOT NULL DEFAULT '',
            explanation TEXT NOT NULL DEFAULT '',
            reasoning_summary TEXT NOT NULL DEFAULT '',
            chart_spec TEXT NOT NULL DEFAULT '{}',
            result_metadata TEXT NOT NULL DEFAULT '{}',
            request_id TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            position INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_messages_conversation ON messages(conversation_id, position)",

        """CREATE TABLE memory_items (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
            importance REAL NOT NULL DEFAULT 0.5,
            confidence REAL NOT NULL DEFAULT 0.5,
            user_approved INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_memory_workspace ON memory_items(workspace_id)",

        """CREATE TABLE tags (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            color TEXT NOT NULL DEFAULT '',
            UNIQUE (workspace_id, name)
        )""",

        """CREATE TABLE conversation_tags (
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
            PRIMARY KEY (conversation_id, tag_id)
        )""",

        """CREATE TABLE query_runs (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            sql TEXT NOT NULL,
            validation_status TEXT NOT NULL,
            execution_status TEXT NOT NULL,
            row_count INTEGER,
            execution_time REAL,
            error_message TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_query_runs_message ON query_runs(message_id)",
    )),

    Migration(2, "answer versions, saved items, and a search index", (
        # Regenerating or editing keeps the earlier version but hides it
        "ALTER TABLE messages ADD COLUMN active INTEGER NOT NULL DEFAULT 1",

        """CREATE TABLE saved_items (
            id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('result', 'chart', 'query')),
            title TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )""",
        "CREATE INDEX idx_saved_items_workspace ON saved_items(workspace_id, created_at)",
        "CREATE INDEX idx_conversations_category ON conversations(workspace_id, category)",
    ), optional=(
        # Full-text search over questions, answers and SQL. Kept in step by triggers.
        """CREATE VIRTUAL TABLE message_search USING fts5(
            body, message_id UNINDEXED, conversation_id UNINDEXED, workspace_id UNINDEXED,
            tokenize = 'porter unicode61')""",
        """CREATE TRIGGER message_search_insert AFTER INSERT ON messages BEGIN
            INSERT INTO message_search (body, message_id, conversation_id, workspace_id)
            VALUES (NEW.content || ' ' || NEW.sql, NEW.id, NEW.conversation_id,
                    (SELECT workspace_id FROM conversations WHERE id = NEW.conversation_id));
        END""",
        """CREATE TRIGGER message_search_delete AFTER DELETE ON messages BEGIN
            DELETE FROM message_search WHERE message_id = OLD.id;
        END""",
        """CREATE TRIGGER message_search_update AFTER UPDATE OF content ON messages BEGIN
            UPDATE message_search SET body = NEW.content || ' ' || NEW.sql WHERE message_id = NEW.id;
        END""",
        """INSERT INTO message_search (body, message_id, conversation_id, workspace_id)
            SELECT m.content || ' ' || m.sql, m.id, m.conversation_id, c.workspace_id
            FROM messages m JOIN conversations c ON c.id = m.conversation_id""",
    )),

    Migration(3, "long-term memory: scope, status, pinning, workspace settings, and a memory search index", (
        "ALTER TABLE workspaces ADD COLUMN settings TEXT NOT NULL DEFAULT '{}'",
        # NULL means the memory applies to every data source in the workspace
        "ALTER TABLE memory_items ADD COLUMN data_source_id TEXT REFERENCES data_sources(id) ON DELETE CASCADE",
        # 'suggested' (waiting for the user), 'active' (in use) or 'rejected' (kept so it isn't suggested again)
        "ALTER TABLE memory_items ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        "ALTER TABLE memory_items ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE memory_items ADD COLUMN last_used_at TEXT",
        "CREATE INDEX idx_memory_status ON memory_items(workspace_id, status)",
    ), optional=(
        """CREATE VIRTUAL TABLE memory_search USING fts5(
            content, memory_id UNINDEXED, workspace_id UNINDEXED, tokenize = 'porter unicode61')""",
        """CREATE TRIGGER memory_search_insert AFTER INSERT ON memory_items BEGIN
            INSERT INTO memory_search (content, memory_id, workspace_id)
            VALUES (NEW.content, NEW.id, NEW.workspace_id);
        END""",
        """CREATE TRIGGER memory_search_delete AFTER DELETE ON memory_items BEGIN
            DELETE FROM memory_search WHERE memory_id = OLD.id;
        END""",
        """CREATE TRIGGER memory_search_update AFTER UPDATE OF content ON memory_items BEGIN
            UPDATE memory_search SET content = NEW.content WHERE memory_id = NEW.id;
        END""",
        """INSERT INTO memory_search (content, memory_id, workspace_id)
            SELECT content, id, workspace_id FROM memory_items""",
    )),
)

LATEST_VERSION = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    """The highest migration applied to this database (0 for an empty one)."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'").fetchone()
    if not exists:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0] or 0)


def backup_file(path: Path, version: int) -> Path:
    """Copy the database beside itself before an upgrade, using SQLite's own backup routine."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.v{version}.{stamp}.bak")
    source = sqlite3.connect(path)
    try:
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return target


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS, *,
            backup_path: Path | None = None) -> list[int]:
    """Apply pending migrations in order. Returns the versions applied.

    `conn` must be in autocommit mode (isolation_level=None) so each migration
    can run in its own explicit transaction. `backup_path` is the database file
    to copy before upgrading a database that already has a version.
    """
    known_latest = migrations[-1].version if migrations else 0
    try:
        version = current_version(conn)
    except sqlite3.DatabaseError as e:
        raise MigrationError(f"The metadata database can't be read ({e}).") from None
    if version > known_latest:
        raise MigrationError(
            f"The metadata database is at version {version}, but this app only knows version {known_latest}. "
            "It was created by a newer release; update the app rather than using this file.")

    pending = [m for m in migrations if m.version > version]
    if not pending:
        return []
    if version > 0 and backup_path is not None and backup_path.exists():
        backup_file(backup_path, version)

    conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)""")
    applied = []
    for migration in pending:
        try:
            conn.execute("BEGIN IMMEDIATE")
            for statement in migration.statements:
                conn.execute(statement)
            if migration.optional:
                conn.execute("SAVEPOINT optional_group")
                try:
                    for statement in migration.optional:
                        conn.execute(statement)
                    conn.execute("RELEASE optional_group")
                except sqlite3.Error:
                    conn.execute("ROLLBACK TO optional_group")
                    conn.execute("RELEASE optional_group")
            conn.execute("INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                         (migration.version, migration.name, datetime.now(timezone.utc).isoformat(timespec="seconds")))
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise MigrationError(
                f"Migration {migration.version} ({migration.name}) failed and was rolled back: {e}. "
                f"The database is still at version {version}.") from None
        applied.append(migration.version)
        version = migration.version
    return applied
