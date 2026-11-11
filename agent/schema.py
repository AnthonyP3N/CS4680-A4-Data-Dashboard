"""Read a SQLite database's structure and turn it into prompt-ready text.

Giving the model the exact schema, a few sample rows and the distinct values
of low-cardinality text columns ("schema grounding") is the single biggest
accuracy boost for text-to-SQL. It stops the model from inventing columns
and lets it match values like status = 'Delivered' exactly.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field


@dataclass
class Column:
    name: str
    type: str
    pk: bool = False
    fk: str | None = None              # "table.column" when this is a foreign key
    distinct_values: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: list[Column]
    row_count: int
    sample_rows: list[tuple]


@dataclass
class SchemaInfo:
    tables: dict[str, Table]

    @property
    def table_names(self) -> set[str]:
        return {t.lower() for t in self.tables}

    def columns_of(self, table: str) -> set[str]:
        for name, t in self.tables.items():
            if name.lower() == table.lower():
                return {c.name.lower() for c in t.columns}
        return set()

    def to_prompt(self, max_sample_rows: int = 3, *, sensitive: set[str] | None = None,
                  table_notes: dict[str, str] | None = None,
                  column_notes: dict[str, str] | None = None) -> str:
        """Render as compact CREATE TABLE statements with comments.

        sensitive:    lower-case "table.column" names whose values must not be shown
        table_notes:  table -> description, written above the table
        column_notes: "table.column" -> description, written beside the column
        """
        sensitive = sensitive or set()
        table_notes = {k.lower(): v for k, v in (table_notes or {}).items()}
        column_notes = {k.lower(): v for k, v in (column_notes or {}).items()}
        parts = []
        for t in self.tables.values():
            hidden = [f"{t.name}.{c.name}".lower() in sensitive for c in t.columns]
            lines = [f"-- {t.name}: {t.row_count:,} rows"]
            if t.name.lower() in table_notes:
                lines.append(f"-- {_one_line(table_notes[t.name.lower()])}")
            lines.append(f"CREATE TABLE {t.name} (")
            col_lines = []
            for c, is_hidden in zip(t.columns, hidden):
                bits = [f"  {c.name} {c.type or 'TEXT'}"]
                if c.pk:
                    bits.append("PRIMARY KEY")
                if c.fk:
                    bits.append(f"REFERENCES {c.fk.replace('.', '(')})")
                line = " ".join(bits)
                comments = []
                note = column_notes.get(f"{t.name}.{c.name}".lower())
                if note:
                    comments.append(_one_line(note))
                if is_hidden:
                    comments.append("sensitive: values hidden")
                elif c.distinct_values:
                    comments.append("values: " + ", ".join(repr(v) for v in c.distinct_values))
                if comments:
                    line += "  -- " + "; ".join(comments)
                col_lines.append(line)
            lines.append(",\n".join(col_lines))
            lines.append(");")
            if t.sample_rows:
                lines.append(f"-- sample rows ({', '.join(c.name for c in t.columns)}):")
                for row in t.sample_rows[:max_sample_rows]:
                    shown = tuple("<hidden>" if h else _clip(v) for v, h in zip(row, hidden))
                    lines.append("--   " + repr(shown))
            parts.append("\n".join(lines))
        return "\n\n".join(parts)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())[:300]


def _clip(value, limit: int = 80):
    """Keep sample values short, and on one line, so a long cell can't crowd or reshape the prompt."""
    if isinstance(value, str):
        flat = " ".join(value.split())
        return flat if len(flat) <= limit else flat[:limit] + "..."
    return value


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def introspect(conn: sqlite3.Connection, *, distinct_limit: int = 12) -> SchemaInfo:
    """Inspect every user table in *conn*."""
    tables: dict[str, Table] = {}
    names = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    for name in names:
        q = _quote(name)
        fks = {
            row[3]: f"{row[2]}.{row[4]}"
            for row in conn.execute(f"PRAGMA foreign_key_list({q})")
        }
        cols = []
        for cid, cname, ctype, _notnull, _default, pk in conn.execute(f"PRAGMA table_info({q})"):
            col = Column(name=cname, type=ctype, pk=bool(pk), fk=fks.get(cname))
            if (ctype or "").upper() in ("TEXT", "") and not col.pk:
                vals = conn.execute(
                    f"SELECT DISTINCT {_quote(cname)} FROM {q} WHERE {_quote(cname)} IS NOT NULL "
                    f"LIMIT {distinct_limit + 1}"
                ).fetchall()
                # Only list values for categorical columns: a few short labels, not names, dates
                # or free text (free text is also where instructions planted in data would hide)
                if (0 < len(vals) <= distinct_limit and not _looks_like_date(vals[0][0])
                        and all(_is_label(v[0]) for v in vals)):
                    col.distinct_values = sorted(str(_clip(str(v[0]), 60)) for v in vals)
            cols.append(col)
        row_count = conn.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
        sample = conn.execute(f"SELECT * FROM {q} LIMIT 3").fetchall()
        tables[name] = Table(name=name, columns=cols, row_count=row_count, sample_rows=sample)
    return SchemaInfo(tables=tables)


def _is_label(value, max_length: int = 40) -> bool:
    text = str(value)
    return len(text) <= max_length and "\n" not in text and "\r" not in text


def _looks_like_date(v) -> bool:
    s = str(v)
    return len(s) >= 10 and s[4] == "-" and s[7] == "-" and s[:4].isdigit()
