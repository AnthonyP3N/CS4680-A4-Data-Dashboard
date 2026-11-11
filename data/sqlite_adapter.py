"""SQLite data source.

Execution is delegated to `agent.executor`, which keeps its protections:
reads on a `mode=ro` connection, a per-query time budget, a hard row cap, dry
runs inside a rolled-back transaction, and a row-count recheck before commit.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from agent import executor
from agent.schema import SchemaInfo, introspect

from .adapters import DataSourceAdapter, DataSourceError, DataSourceInfo, DataSourceLimits, _now
from .metadata import SemanticConfig, merge_suggestions, suggest_config


class SQLiteAdapter(DataSourceAdapter):
    """
    semantics:    the data source's business context. When omitted, a starting
                  configuration is suggested from the schema and kept up to date
                  as tables are added (user edits are never overwritten).
    """

    def __init__(self, path: str | Path, *, name: str = "SQLite database", description: str = "",
                 writable: bool = True, is_sample: bool = False, source_id: str | None = None,
                 source_type: str = "sqlite", semantics: SemanticConfig | None = None,
                 limits: DataSourceLimits | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise DataSourceError(f"Database file not found: {self.path.name}")
        self.info = DataSourceInfo(id=source_id or uuid.uuid4().hex[:10], name=name, type=source_type,
                                   dialect="sqlite", description=description, writable=writable,
                                   is_sample=is_sample, connection={"file": self.path.name},
                                   limits=limits or DataSourceLimits())
        self._auto_semantics = semantics is None
        self.semantics = semantics or SemanticConfig(description=description)
        self._schema: SchemaInfo | None = None
        self.schema()   # fail early if the file can't be read, and build the suggested configuration

    # -- structure --------------------------------------------------------- #
    def _read_only_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=5)

    def schema(self, refresh: bool = False) -> SchemaInfo:
        if self._schema is None or refresh:
            conn = self._read_only_connection()
            try:
                self._schema = introspect(conn)
            except sqlite3.Error as e:
                raise DataSourceError(f"Couldn't read the structure of '{self.info.name}': {e}") from None
            finally:
                conn.close()
            self.info.last_schema_refresh = _now()
            if self._auto_semantics:
                self._auto_semantics = False     # suggest once; later edits belong to the user
                self.suggest_semantics()
        return self._schema

    def suggest_semantics(self, tables: list[str] | None = None) -> None:
        """Add suggested date fields, joins, sensitive columns and starter questions for `tables`
        (all tables when omitted). Existing entries, including the user's edits, are kept."""
        schema = self.schema()
        wanted = None if tables is None else {t.lower() for t in tables}
        subset = SchemaInfo({n: t for n, t in schema.tables.items() if wanted is None or n.lower() in wanted})
        merge_suggestions(self.semantics, suggest_config(subset, self.info.description))

    def preview(self, table: str, limit: int = 50) -> pd.DataFrame:
        # Only names that exist in the schema are accepted, so the name can't carry SQL
        known = {t.lower(): t for t in self.schema().tables}
        real = known.get(table.lower())
        if real is None:
            raise DataSourceError(f"Unknown table '{table}'.")
        conn = self._read_only_connection()
        try:
            quoted = '"' + real.replace('"', '""') + '"'
            return pd.read_sql_query(f"SELECT * FROM {quoted} LIMIT {int(limit)}", conn)
        finally:
            conn.close()

    # -- execution --------------------------------------------------------- #
    def run_read(self, validated):
        return executor.run_read(self.path, validated, timeout_s=self.info.limits.query_timeout_s,
                                 max_rows=self.info.limits.max_rows)

    def dry_run_write(self, validated):
        if not self.info.writable:
            return super().dry_run_write(validated)
        return executor.dry_run_write(self.path, validated, timeout_s=self.info.limits.query_timeout_s)

    def commit_write(self, sql, expected_rows=None):
        if not self.info.writable:
            return super().commit_write(sql, expected_rows)
        n = executor.commit_write(self.path, sql, expected_rows=expected_rows,
                                  timeout_s=self.info.limits.query_timeout_s)
        self.schema(refresh=True)
        return n
