"""The metadata database: where conversations, workspaces and settings are kept.

It is a different file from any dataset being analysed. Analytical data never
goes in here, and nothing in here is ever shown to the query-writing model.

`Storage.open(path)` never fails. If the file can't be opened or upgraded, it
returns a storage object backed by a private in-memory database, with
`persistent = False` and a `warning` explaining why, so the app keeps working
and the UI can say that conversations aren't being saved.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .migrations import LATEST_VERSION, MigrationError, current_version, migrate


class MetadataDB:
    """Connection handling for the metadata database.

    File mode opens a short-lived connection per operation, which is safe across
    the threads a web server uses. Memory mode keeps one connection behind a lock.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.RLock()
        self._memory: sqlite3.Connection | None = None
        if path is None:
            self._memory = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
            self._configure(self._memory)

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """An autocommit connection. Use `transaction()` for anything that writes more than one row."""
        if self._memory is not None:
            with self._lock:
                yield self._memory
            return
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        try:
            self._configure(conn)
            conn.execute("PRAGMA busy_timeout = 5000")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All statements inside commit together, or none do."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def version(self) -> int:
        with self.connect() as conn:
            return current_version(conn)


class Storage:
    """The metadata database plus one repository per kind of record."""

    def __init__(self, db: MetadataDB, *, persistent: bool, warning: str = ""):
        from .repositories import (ConversationRepository, DataSourceRepository, MemoryRepository,
                                   MessageRepository, QueryRunRepository, SavedItemRepository, TagRepository,
                                   WorkspaceRepository)
        self.db = db
        self.persistent = persistent
        self.warning = warning
        self.workspaces = WorkspaceRepository(db)
        self.data_sources = DataSourceRepository(db)
        self.conversations = ConversationRepository(db)
        self.messages = MessageRepository(db)
        self.query_runs = QueryRunRepository(db)
        self.tags = TagRepository(db)
        self.saved_items = SavedItemRepository(db)
        self.memories = MemoryRepository(db)

    @property
    def path(self) -> Path | None:
        return self.db.path

    @property
    def version(self) -> int:
        return self.db.version()

    @classmethod
    def in_memory(cls, warning: str = "") -> "Storage":
        """A working store that lasts only as long as this object. Used as the fallback and in tests."""
        db = MetadataDB(None)
        with db.connect() as conn:
            migrate(conn)
        return cls(db, persistent=False, warning=warning)

    @classmethod
    def open(cls, path: str | Path) -> "Storage":
        """Open (creating or upgrading as needed) the metadata database at `path`.

        Never raises: on any failure the result is an in-memory store with `warning` set.
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = MetadataDB(path)
            with db.connect() as conn:
                conn.execute("PRAGMA journal_mode = WAL")
                migrate(conn, backup_path=path)
                if current_version(conn) != LATEST_VERSION:
                    raise MigrationError("The metadata database did not reach the expected version.")
            return cls(db, persistent=True)
        except MigrationError as e:
            return cls.in_memory(str(e))
        except (sqlite3.Error, OSError) as e:
            return cls.in_memory(f"The metadata database at {path.name} can't be used ({e}).")
