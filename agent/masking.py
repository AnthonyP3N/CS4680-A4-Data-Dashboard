"""Hiding the values of sensitive columns in results.

A data source's business context names its sensitive columns ("customers.email")
and a masking mode:

  partial  (default) enough is left to tell values apart: "j•••@example.com", "M••• R•••"
  full     every value becomes "••••"
  off      values are shown as stored (the owner's choice)

Which result columns to mask is worked out from the SQL, not from column names:
sqlglot traces each output column back to the table columns it was computed from
(through aliases, joins, subqueries, CTEs, UNIONs and `SELECT *`). Any output column
built from a sensitive column is masked, so `UPPER(email) AS contact` is caught and
`products.name` is not mistaken for `customers.name`. The one exception is COUNT,
which says how many values there are without revealing any.

If the SQL can't be traced, the guard errs on the side of hiding: every text column,
and any column named like a sensitive one, is masked.

Masking is applied once, right after the query runs, so the table, chart, CSV
download, saved snapshots and exports all receive the masked values. What it can't
stop is inference through filters (`WHERE email LIKE 'a%'` still changes a count);
see docs/security.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import sqlglot
from sqlglot import exp

if TYPE_CHECKING:
    from .schema import SchemaInfo

DOT = "•"
FULL_MASK = DOT * 4
MAX_CELL_CHARS = 1000                          # longer text values are cut, so one cell can't flood the page
_EMAIL = re.compile(r"^([^@\s]+)@([^@\s]+)$")


@dataclass
class GuardResult:
    df: pd.DataFrame
    masked_columns: list[str] = field(default_factory=list)   # result column names whose values were hidden
    traced: bool = True                                        # False when the SQL couldn't be traced
    clipped_cells: int = 0

    def notes(self, mode: str) -> list[str]:
        out = []
        if self.masked_columns:
            how = "hidden" if mode == "full" else "partly hidden"
            names = ", ".join(self.masked_columns)
            out.append(f"Values in {names} are {how} because "
                       f"{'it comes' if len(self.masked_columns) == 1 else 'they come'} from a column marked "
                       "sensitive. The data's owner can change this in the Data tab.")
            if not self.traced:
                out.append("AskData couldn't trace every column of this query, so it hid all text columns to be safe.")
        if self.clipped_cells:
            out.append(f"{self.clipped_cells} very long value{'s were' if self.clipped_cells != 1 else ' was'} "
                       f"cut to {MAX_CELL_CHARS:,} characters.")
        return out


# --------------------------------------------------------------------------- #
# Masking single values
# --------------------------------------------------------------------------- #
def mask_value(value: object, mode: str = "partial") -> object:
    """The masked form of one value. Missing values stay missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return value
    if mode == "off":
        return value
    if mode == "full" or isinstance(value, (int, float)) and not isinstance(value, bool):
        return FULL_MASK                         # numbers are never partly shown: digits are the secret
    text = str(value).strip()
    if not text:
        return text
    email = _EMAIL.match(text)
    if email:
        return f"{email.group(1)[0]}{DOT * 3}@{email.group(2)}"
    digits = sum(ch.isdigit() for ch in text)
    if digits >= max(4, len(text) // 2):        # phone numbers, account and card numbers, ids
        return DOT * 4 + text[-2:]
    return " ".join(word[0] + DOT * 3 for word in text.split())


def mask_columns(df: pd.DataFrame, columns: list[str], mode: str = "partial") -> pd.DataFrame:
    """A copy of `df` with every column named in `columns` masked (as text). `df` itself is not changed."""
    if df is None or not columns or mode == "off":
        return df
    wanted = set(columns)
    return _mask_positions(df, [i for i, c in enumerate(df.columns) if c in wanted], mode)


def _mask_positions(df: pd.DataFrame, positions: list[int], mode: str) -> pd.DataFrame:
    """Mask columns by position, so results with repeated column names (a.*, b.*) are handled exactly."""
    if not positions or mode == "off":
        return df
    out = df.copy()
    for i in positions:
        out.isetitem(i, out.iloc[:, i].map(lambda v: mask_value(v, mode)).astype(object))
    return out


def clip_cells(df: pd.DataFrame, limit: int = MAX_CELL_CHARS) -> tuple[pd.DataFrame, int]:
    """Cut text values longer than `limit` characters. Returns the frame and how many were cut."""
    if df is None or df.empty:
        return df, 0
    clipped = 0
    out = df
    for i in range(df.shape[1]):
        series = df.iloc[:, i]
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            continue
        long = series.map(lambda v: isinstance(v, str) and len(v) > limit)
        count = int(long.sum())
        if count:
            if out is df:
                out = df.copy()
            out.isetitem(i, series.map(lambda v: v[:limit] + "…" if isinstance(v, str) and len(v) > limit else v))
            clipped += count
    return out, clipped


# --------------------------------------------------------------------------- #
# Which result columns come from sensitive columns
# --------------------------------------------------------------------------- #
def schema_mapping(schema: "SchemaInfo") -> dict[str, dict[str, str]]:
    return {t.name: {c.name: (c.type or "TEXT") for c in t.columns} for t in schema.tables.values()}


def _bare(name: str) -> str:
    return name.strip().strip('"`[]').lower()


def traced_sources(sql: str, schema: "SchemaInfo", dialect: str = "sqlite") -> list[set[str]] | None:
    """For each output column of a SELECT, in order: the "table.column"s it is computed from.

    Columns that are a COUNT get an empty set (a count reveals no values). Returns
    None if the statement can't be traced.
    """
    from sqlglot.lineage import lineage
    from sqlglot.optimizer.qualify import qualify

    mapping = schema_mapping(schema)
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
        tree = qualify(tree, schema=mapping, dialect=dialect, validate_qualify_columns=False)
        selects = list(tree.selects)
        if not selects or any(isinstance(select, exp.Star) for select in selects):
            return None
        # Give each output column a unique name first: lineage looks columns up by name,
        # and "SELECT p.name, c.name" has two columns called "name"
        names = [f"askdata_col_{i}" for i in range(len(selects))]
        counts = [isinstance(select.unalias(), exp.Count) for select in selects]
        for name, select in zip(names, selects):
            select.replace(exp.alias_(select.unalias().copy(), name))
        out: list[set[str]] = []
        for name, is_count in zip(names, counts):
            if is_count:
                out.append(set())
                continue
            node = lineage(name, tree, schema=mapping, dialect=dialect)
            found: set[str] = set()
            for n in node.walk():
                if n.downstream:
                    continue
                if isinstance(n.source, exp.Table):
                    found.add(f"{_bare(n.source.name)}.{_bare(n.name.split('.')[-1])}")
            out.append(found)
        return out
    except Exception:  # sqlglot raises many kinds of errors for SQL it can't follow
        return None


def protect(df: pd.DataFrame, sql: str, schema: "SchemaInfo", sensitive: set[str], mode: str = "partial",
            dialect: str = "sqlite", *, flagged: bool = False) -> GuardResult:
    """Mask result columns computed from sensitive columns, and cut over-long cells.

    sensitive: lower-case "table.column" names
    flagged:   the validator found that the SQL may read a sensitive column; when the
               SQL can't be traced, this decides whether to mask defensively
    """
    if df is None:
        return GuardResult(df)
    df, clipped = clip_cells(df)
    if not sensitive or mode == "off" or df.empty and not len(df.columns):
        return GuardResult(df, clipped_cells=clipped)

    columns = list(df.columns)
    traced = traced_sources(sql, schema, dialect)
    sensitive_names = {ref.split(".", 1)[-1] for ref in sensitive}
    positions: list[int] = []
    if traced is not None and len(traced) == len(columns):
        positions = [i for i, sources in enumerate(traced) if sources & sensitive]
        ok = True
    else:
        ok = False
        for i, c in enumerate(columns):
            named_like = _bare(str(c)) in sensitive_names
            text_column = flagged and not pd.api.types.is_numeric_dtype(df.iloc[:, i])
            if named_like or text_column:
                positions.append(i)
    names = list(dict.fromkeys(str(columns[i]) for i in positions))
    return GuardResult(_mask_positions(df, positions, mode), names, traced=ok, clipped_cells=clipped)


def protect_table(df: pd.DataFrame, table: str, sensitive: set[str], mode: str = "partial") -> pd.DataFrame:
    """Mask a straight copy of a table's rows (a data preview, or the rows a change will touch)."""
    if df is None or mode == "off" or not sensitive:
        return df
    prefix = table.lower() + "."
    return _mask_positions(df, [i for i, c in enumerate(df.columns) if prefix + str(c).lower() in sensitive], mode)
