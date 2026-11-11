"""The data-source interface.

Everything above this layer (the agent pipeline, the conversation service, the
UI) talks to a `DataSourceAdapter` and never to a database driver directly.
Adding a new kind of database (PostgreSQL, MySQL, ...) means writing one
adapter class and registering it in `data/registry.py`; see docs in the README.

Adapters only ever receive SQL that has already passed `agent.validator`
(a `ValidatedSQL`), so the safety checks cannot be skipped by going through here.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from .metadata import SemanticConfig

if TYPE_CHECKING:  # type-only imports keep this module free of runtime dependencies on agent/
    from agent.executor import QueryResult, WritePreview
    from agent.schema import Column, SchemaInfo
    from agent.validator import ValidatedSQL


class DataSourceError(Exception):
    """A data source could not be opened, read or modified."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Secrets never leave the adapter
# --------------------------------------------------------------------------- #
SECRET_KEY = re.compile(
    r"(password|passwd|passphrase|(?<![a-z])pwd(?![a-z])|secret|(?<![a-z])token(?!s)|api[_-]?key|"
    r"credential|private[_-]?key|authorization|access[_-]?key)", re.I)
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-z][a-z0-9+.\-]*://)[^/@\s]+@", re.I)
# Common API-key shapes, in case one is pasted into a question or a setting
_KEY_SHAPES = re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|gsk_[A-Za-z0-9]{16,}|AIza[0-9A-Za-z_\-]{20,}|"
                         r"xox[abp]-[A-Za-z0-9\-]{10,}|gh[pousr]_[A-Za-z0-9]{20,})")
REDACTED = "***"


def redact(value: Any) -> Any:
    """A copy of `value` that is safe to log, store or show.

    Removes the values of secret-looking keys (password, token, api_key, ...)
    credentials embedded in URLs (postgres://user:pass@host -> postgres://***@host),
    and anything shaped like an API key.
    """
    if isinstance(value, dict):
        return {k: (REDACTED if SECRET_KEY.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = _URL_CREDENTIALS.sub(lambda m: m.group("scheme") + REDACTED + "@", value)
        return _KEY_SHAPES.sub(REDACTED, value)
    return value


# --------------------------------------------------------------------------- #
# Description of a data source
# --------------------------------------------------------------------------- #
@dataclass
class DataSourceLimits:
    query_timeout_s: float = 8.0     # a running query is stopped after this long
    max_rows: int = 1000             # most rows a single answer may return


@dataclass
class DataSourceInfo:
    """Descriptive metadata for one data source. Never holds passwords or API keys:
    `connection` is passed through `redact` on the way in."""

    id: str
    name: str
    type: str                      # "sqlite", "csv", "postgres", ...
    dialect: str                   # SQL dialect name understood by sqlglot, e.g. "sqlite"
    description: str = ""
    writable: bool = False         # whether confirmed INSERT / UPDATE / DELETE are supported
    is_sample: bool = False        # True for the built-in demo dataset
    connection: dict[str, Any] = field(default_factory=dict)   # non-secret details, e.g. {"file": "store.db"}
    limits: DataSourceLimits = field(default_factory=DataSourceLimits)
    created_at: datetime = field(default_factory=_now)
    last_schema_refresh: datetime | None = None

    def __post_init__(self) -> None:
        self.connection = redact(self.connection)

    def public(self) -> dict[str, Any]:
        """Safe-to-display summary (for logs, exports and the UI)."""
        return {
            "id": self.id, "name": self.name, "type": self.type, "dialect": self.dialect,
            "description": self.description, "writable": self.writable,
            "connection": redact(self.connection),
            "query_timeout_s": self.limits.query_timeout_s, "max_rows": self.limits.max_rows,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "last_schema_refresh": (self.last_schema_refresh.isoformat(timespec="seconds")
                                    if self.last_schema_refresh else None),
        }


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
class DataSourceAdapter(ABC):
    """What the agent needs from any relational data source.

    Required: `schema`, `preview`, `run_read`.
    Optional: `dry_run_write` / `commit_write` (only for writable sources).
    Each adapter enforces its own `info.limits` (query timeout, row cap) and
    must run reads on a connection that cannot modify data.
    """

    info: DataSourceInfo
    semantics: SemanticConfig

    @property
    def dialect(self) -> str:
        return self.info.dialect

    # -- structure --------------------------------------------------------- #
    @abstractmethod
    def schema(self, refresh: bool = False) -> "SchemaInfo":
        """Tables, columns, sample rows and categorical values. Cached until refresh=True."""

    @abstractmethod
    def preview(self, table: str, limit: int = 50) -> pd.DataFrame:
        """First rows of one table, for the data browser. The table must exist in schema()."""

    def tables(self) -> list[str]:
        return list(self.schema().tables)

    def columns(self, table: str) -> list["Column"]:
        return list(self._table(table).columns)

    def sample_rows(self, table: str, limit: int = 3) -> list[tuple]:
        return list(self._table(table).sample_rows[:limit])

    def distinct_values(self, table: str, column: str) -> list[str]:
        """Known values of a categorical column (empty for free-text, numeric or high-cardinality columns)."""
        for c in self._table(table).columns:
            if c.name.lower() == column.lower():
                return list(c.distinct_values)
        raise DataSourceError(f"Unknown column '{table}.{column}'.")

    def _table(self, table: str):
        for name, t in self.schema().tables.items():
            if name.lower() == table.lower():
                return t
        raise DataSourceError(f"Unknown table '{table}'.")

    # -- execution --------------------------------------------------------- #
    @abstractmethod
    def run_read(self, validated: "ValidatedSQL") -> "QueryResult":
        """Run a validated SELECT on a connection that cannot write."""

    def dry_run_write(self, validated: "ValidatedSQL") -> "WritePreview":
        """Try a validated write, report what it would change, and undo it."""
        raise DataSourceError(f"'{self.info.name}' is read-only.")

    def commit_write(self, sql: str, expected_rows: int | None = None) -> int:
        """Apply a confirmed write. Must roll back if the row count differs from the dry run."""
        raise DataSourceError(f"'{self.info.name}' is read-only.")
