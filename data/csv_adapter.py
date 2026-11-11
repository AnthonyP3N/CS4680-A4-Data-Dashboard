"""CSV uploads: each file becomes a table inside a SQLite data source.

Uploaded files are untrusted. Table and column names are rebuilt from a strict
character set, so nothing in a file name or header can end up as SQL.

Date columns are rewritten as 'YYYY-MM-DD' (or 'YYYY-MM-DD HH:MM:SS') text, the
one form SQLite's date functions understand, so "last month" and "by month"
questions work no matter how the spreadsheet formatted its dates.
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

from .adapters import DataSourceError
from .sqlite_adapter import SQLiteAdapter

MAX_ROWS = 200_000
MAX_COLUMNS = 200

_NUMERIC_DATE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
_NAMED_MONTH = re.compile(r"^\s*([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4})")


@dataclass
class CsvImport:
    table: str
    rows: int
    columns: list[str]
    truncated: bool
    date_columns: list[str] = field(default_factory=list)   # columns whose dates were standardised


def safe_identifier(raw: str, fallback: str) -> str:
    """Lower-case letters, digits and underscores only; never empty; never starts with a digit."""
    name = re.sub(r"\W+", "_", str(raw), flags=re.ASCII).strip("_").lower()
    if not name:
        name = fallback
    if name[0].isdigit():
        name = "t_" + name
    return name[:60]


def standardise_dates(df: pd.DataFrame) -> list[str]:
    """Rewrite text columns that hold dates into ISO form, in place. Returns the columns changed.

    A column is only converted when nearly all of its values parse as dates with
    one consistent format; anything ambiguous or mixed is left exactly as it was.
    """
    converted = []
    for column in df.columns:
        series = df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        present = series.dropna().astype(str)
        if present.empty:
            continue
        sample = present.head(200)
        if (sample.str.match(_NUMERIC_DATE) | sample.str.match(_NAMED_MONTH)).mean() < 0.9:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError, OverflowError):
            continue
        if not pd.api.types.is_datetime64_any_dtype(parsed) or parsed.notna().sum() < 0.95 * len(present):
            continue
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        valid = parsed.dropna()
        fmt = "%Y-%m-%d" if (valid == valid.dt.normalize()).all() else "%Y-%m-%d %H:%M:%S"
        df[column] = parsed.dt.strftime(fmt).astype(object).where(parsed.notna(), None)
        converted.append(column)
    return converted


def import_csv(source: SQLiteAdapter, file: IO | str | Path, filename: str) -> CsvImport:
    """Load a CSV into `source` as a new table, refresh its schema and suggest semantics for the new table."""
    try:
        df = pd.read_csv(file, nrows=MAX_ROWS + 1)
    except Exception as e:  # pandas raises many types for malformed files
        raise DataSourceError(f"Couldn't read that CSV: {e}") from None
    if df.empty or len(df.columns) == 0:
        raise DataSourceError("That CSV has no data rows.")
    if len(df.columns) > MAX_COLUMNS:
        raise DataSourceError(f"That CSV has {len(df.columns)} columns; the limit is {MAX_COLUMNS}.")
    truncated = len(df) > MAX_ROWS
    df = df.head(MAX_ROWS).copy()

    columns: list[str] = []
    for i, c in enumerate(df.columns):
        col = safe_identifier(c, f"col{i}")
        while col in columns:          # two headers that clean to the same name
            col += "_2"
        columns.append(col)
    df.columns = columns
    date_columns = standardise_dates(df)

    table = safe_identifier(Path(filename).stem, "uploaded")
    if table in source.schema().table_names:
        table = f"{table}_upload"

    conn = sqlite3.connect(source.path)
    try:
        df.to_sql(table, conn, index=False, if_exists="replace")
    except Exception as e:
        raise DataSourceError(f"Couldn't load that CSV: {e}") from None
    finally:
        conn.close()
    source.schema(refresh=True)
    source.suggest_semantics([table])
    return CsvImport(table=table, rows=len(df), columns=columns, truncated=truncated, date_columns=date_columns)
