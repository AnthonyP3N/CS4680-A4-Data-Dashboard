"""Execute validated SQL against SQLite, safely.

- Reads run on a *read-only* connection (mode=ro), so even a validator bug
  cannot modify data through a SELECT.
- Every query has a time budget enforced with SQLite's progress handler.
- Writes are first *dry-run* inside a transaction that is rolled back, to count
  affected rows and preview them; they are committed only after the user confirms.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .validator import ValidatedSQL, write_preview_select

QUERY_TIMEOUT_S = 8.0


MAX_VALUE_BYTES = 10_000_000   # largest text or blob one query may build (stops memory-exhausting expressions)


class ExecutionError(Exception):
    pass


@dataclass
class QueryResult:
    df: pd.DataFrame
    elapsed_ms: float
    truncated: bool = False


@dataclass
class WritePreview:
    affected_rows: int
    preview: pd.DataFrame | None   # rows that will change / be deleted (UPDATE/DELETE)
    elapsed_ms: float


def _connect(db_path: str | Path, read_only: bool, timeout_s: float = QUERY_TIMEOUT_S) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    else:
        conn = sqlite3.connect(path, timeout=5, isolation_level=None)  # manual transactions
    if hasattr(conn, "setlimit"):        # Python 3.11+: no single text or blob value may exceed this
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    deadline = time.monotonic() + timeout_s
    # Returning non-zero from the handler aborts the running statement
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5_000)
    return conn


def _friendly(e: Exception, timeout_s: float = QUERY_TIMEOUT_S) -> str:
    msg = str(e)
    if "interrupted" in msg.lower():
        return f"The query took longer than {timeout_s:g}s and was stopped."
    return msg


def run_read(db_path: str | Path, v: ValidatedSQL, *, timeout_s: float = QUERY_TIMEOUT_S,
             max_rows: int | None = None) -> QueryResult:
    """Run a validated SELECT on a read-only connection.

    The row cap is enforced here as well as by the LIMIT the validator adds, so a
    LIMIT the validator could not interpret still cannot return an unbounded result.
    """
    from .validator import MAX_ROWS
    cap = max_rows or MAX_ROWS
    t0 = time.perf_counter()
    conn = _connect(db_path, read_only=True, timeout_s=timeout_s)
    try:
        cursor = conn.execute(v.sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(cap)
        df = pd.DataFrame.from_records(rows, columns=columns, coerce_float=True)
    except Exception as e:  # sqlite3.Error and pandas wrappers
        raise ExecutionError(_friendly(e, timeout_s)) from None
    finally:
        conn.close()
    return QueryResult(df=df, elapsed_ms=(time.perf_counter() - t0) * 1000, truncated=len(df) >= cap)


def dry_run_write(db_path: str | Path, v: ValidatedSQL, *, timeout_s: float = QUERY_TIMEOUT_S) -> WritePreview:
    """Execute the write inside a transaction, count affected rows, then roll back."""
    t0 = time.perf_counter()
    conn = _connect(db_path, read_only=False, timeout_s=timeout_s)
    preview_df = None
    try:
        preview_sql = write_preview_select(v)
        if preview_sql:
            preview_df = pd.read_sql_query(preview_sql, conn)
        conn.execute("BEGIN")
        cur = conn.execute(v.sql)
        affected = cur.rowcount
        conn.execute("ROLLBACK")
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise ExecutionError(_friendly(e, timeout_s)) from None
    finally:
        conn.close()
    return WritePreview(affected_rows=affected, preview=preview_df,
                        elapsed_ms=(time.perf_counter() - t0) * 1000)


def commit_write(db_path: str | Path, sql: str, expected_rows: int | None = None, *,
                 timeout_s: float = QUERY_TIMEOUT_S) -> int:
    """Run a confirmed write. If the row count no longer matches the preview, roll back."""
    conn = _connect(db_path, read_only=False, timeout_s=timeout_s)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(sql)
        affected = cur.rowcount
        if expected_rows is not None and affected != expected_rows:
            conn.execute("ROLLBACK")
            raise ExecutionError(
                f"Data changed since the preview ({expected_rows} rows expected, {affected} now). "
                "Nothing was changed; please ask again."
            )
        conn.execute("COMMIT")
        return affected
    except ExecutionError:
        raise
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise ExecutionError(_friendly(e, timeout_s)) from None
    finally:
        conn.close()
