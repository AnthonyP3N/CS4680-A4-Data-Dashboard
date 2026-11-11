"""Validate LLM-generated SQL *before* anything touches the database.

The model's output is treated as untrusted input. Checks, in order:
  1. It parses as SQLite (sqlglot), and is exactly one statement.
  2. Its statement type is allowed: SELECT always; INSERT/UPDATE/DELETE only when
     the user has enabled data changes; everything else (DROP, ALTER, PRAGMA,
     ATTACH, CREATE, VACUUM, ...) is always blocked.
  3. Every table it references exists (CTE names are allowed).
  4. UPDATE / DELETE must have a WHERE clause (no whole-table wipes).
  4b. ...and that WHERE must mention a column (WHERE 1 = 1 is refused).
  5. Dangerous functions (load_extension, readfile, pragma_*, randomblob, ...) are blocked.
  6. SELECTs without a LIMIT get one added, so a bad query can't dump millions of rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .schema import SchemaInfo

READ = "read"
WRITE = "write"

MAX_ROWS = 1000
DEFAULT_DIALECT = "sqlite"

# Functions that read files, load code, reach other servers or stall the database.
# One list for every dialect: blocking a function a dialect doesn't have costs nothing.
BLOCKED_FUNCTIONS = {
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",                    # SQLite
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_sleep", "pg_sleep_for",       # PostgreSQL
    "lo_import", "lo_export", "dblink", "dblink_exec", "pg_terminate_backend", "set_config",
    "load_file", "sleep", "benchmark", "sys_exec", "sys_eval",                             # MySQL
    "xp_cmdshell", "openrowset",                                                           # SQL Server
    "randomblob", "zeroblob",                                         # can ask for gigabytes of memory at once
}
# Function names starting with these read the database's settings or files (SQLite's pragma_* table functions)
BLOCKED_FUNCTION_PREFIXES = ("pragma_",)
# Schemas and name prefixes that hold a database's own catalogue, users and settings
SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast", "mysql", "performance_schema", "sys"}
SYSTEM_TABLE_PREFIXES = ("sqlite_", "pg_", "pragma_")

# Node types that must never appear anywhere in a statement (getattr keeps this
# working across sqlglot versions that lack some of these classes).
_FORBIDDEN_NODES = tuple(
    getattr(exp, n) for n in ("Drop", "Alter", "Create", "Command", "Pragma", "Attach", "Detach",
                              "Into", "Grant", "Revoke", "Set", "Use", "Copy", "LoadData", "TruncateTable",
                              "Transaction", "Commit", "Rollback", "Merge")
    if hasattr(exp, n)
)


class ValidationError(Exception):
    """Raised when generated SQL fails a safety or correctness check.

    policy=True means the SQL was refused by a safety rule (asking the model to
    "fix" it won't help); policy=False means it is a correctness problem the
    model can repair (bad syntax, unknown table).
    """

    def __init__(self, message: str, policy: bool = False):
        super().__init__(message)
        self.policy = policy


@dataclass
class ValidatedSQL:
    sql: str                          # final SQL to execute (may have LIMIT added)
    kind: str                         # READ or WRITE
    statement: str                    # "SELECT", "UPDATE", ...
    tables: list[str] = field(default_factory=list)
    target_table: str | None = None   # for writes
    notes: list[str] = field(default_factory=list)
    expression: exp.Expression | None = None
    dialect: str = "sqlite"
    sensitive_columns: list[str] = field(default_factory=list)   # sensitive "table.column"s the SQL may expose


def _statement_name(node: exp.Expression) -> str:
    return type(node).__name__.upper()


def sensitive_references(node: exp.Expression, sensitive: set[str]) -> list[str]:
    """Which sensitive columns ("table.column", lower case) a statement might read.

    Deliberately over-cautious: a column counts if its name matches a sensitive
    column of any table the statement uses, whatever alias it is given, and
    `SELECT *` counts for every sensitive column of those tables. COUNT(*) does not.
    """
    if not sensitive:
        return []
    used_tables = {t.name.lower() for t in node.find_all(exp.Table) if t.name}
    by_column: dict[str, set[str]] = {}
    for ref in sensitive:
        table, _, column = ref.partition(".")
        if table in used_tables:
            by_column.setdefault(column, set()).add(ref)
    found: set[str] = set()
    for column in node.find_all(exp.Column):
        found |= by_column.get(column.name.lower(), set())
    for star in node.find_all(exp.Star):
        if not isinstance(star.parent, exp.AggFunc):          # SELECT * or t.*, but not COUNT(*)
            for refs in by_column.values():
                found |= refs
    return sorted(found)


def validate_sql(sql: str, schema: SchemaInfo, *, allow_writes: bool = False,
                 max_rows: int = MAX_ROWS, dialect: str = DEFAULT_DIALECT,
                 sensitive: set[str] | None = None) -> ValidatedSQL:
    """Check one generated statement.

    dialect:   the data source's SQL dialect (a sqlglot name)
    sensitive: lower-case "table.column" names; any the statement may read are reported
               on the result as `sensitive_columns`
    """
    if not sql or not sql.strip():
        raise ValidationError("The model returned no SQL.")

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        msg = str(e).splitlines()[0]
        raise ValidationError(f"SQL could not be parsed: {msg}") from None

    if len(statements) != 1:
        raise ValidationError(f"Expected exactly one SQL statement, got {len(statements)}.")
    node = statements[0]
    notes: list[str] = []

    # 2. Statement type
    if isinstance(node, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        kind, stmt = READ, "SELECT"
    elif isinstance(node, (exp.Insert, exp.Update, exp.Delete)):
        kind, stmt = WRITE, _statement_name(node)
        if not allow_writes:
            raise ValidationError(
                f"{stmt} statements change data and are disabled. "
                "Turn on 'Allow data changes' in the sidebar to enable them.",
                policy=True,
            )
    else:
        raise ValidationError(
            f"{_statement_name(node)} statements are not allowed. Only SELECT (and, when enabled, "
            "INSERT / UPDATE / DELETE) can run.",
            policy=True,
        )

    # Nested writes inside a SELECT, or any DDL hidden anywhere, are blocked too
    for bad in _FORBIDDEN_NODES:
        if node.find(bad):
            raise ValidationError(f"{bad.__name__.upper()} is not allowed.", policy=True)

    # 3. Table whitelist
    cte_names = {c.alias_or_name.lower() for c in node.find_all(exp.CTE)}
    referenced = []
    for t in node.find_all(exp.Table):
        name = t.name
        if not name:
            continue
        low = name.lower()
        if low in cte_names:
            continue
        qualifier = (t.db or "").lower()
        if low.startswith(SYSTEM_TABLE_PREFIXES) or qualifier in SYSTEM_SCHEMAS or (t.catalog or ""):
            raise ValidationError("System tables and other databases cannot be queried.", policy=True)
        if low not in schema.table_names:
            known = ", ".join(sorted(schema.tables))
            raise ValidationError(f"Unknown table '{name}'. Available tables: {known}.")
        if name not in referenced:
            referenced.append(name)

    # 5. Dangerous functions
    for fn in node.find_all(exp.Func):
        fn_name = str(fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
        if fn_name in BLOCKED_FUNCTIONS or fn_name.startswith(BLOCKED_FUNCTION_PREFIXES):
            raise ValidationError(f"Function '{fn_name}' is not allowed.", policy=True)

    target = None
    if kind == WRITE:
        tbl = node.find(exp.Table)
        target = tbl.name if tbl is not None else None
        # 4. Require WHERE on UPDATE / DELETE
        if isinstance(node, (exp.Update, exp.Delete)) and not node.args.get("where"):
            raise ValidationError(
                f"{stmt} without a WHERE clause would affect every row in '{target}'. "
                "Please be more specific about which rows to change.",
                policy=True,
            )
        # ...and a WHERE that names a column: "WHERE 1 = 1" is a whole-table change in disguise
        where = node.args.get("where")
        if isinstance(node, (exp.Update, exp.Delete)) and where is not None and where.find(exp.Column) is None:
            raise ValidationError(
                f"{stmt} with a condition that doesn't mention any column (such as WHERE 1 = 1) would affect "
                f"every row in '{target}'. Please be more specific about which rows to change.",
                policy=True,
            )

    # 6. Enforce a row limit on reads
    final_sql = sql.strip().rstrip(";")
    if kind == READ:
        limit_node = node.args.get("limit")
        if limit_node is None:
            node = node.limit(max_rows) if isinstance(node, exp.Select) else \
                exp.select("*").from_(node.subquery("q")).limit(max_rows)
            final_sql = node.sql(dialect=dialect, pretty=True)
            notes.append(f"Added LIMIT {max_rows} to keep results manageable.")
        else:
            try:
                n = int(limit_node.expression.this)
                if n > max_rows:
                    node.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                    final_sql = node.sql(dialect=dialect, pretty=True)
                    notes.append(f"Reduced LIMIT {n} to {max_rows}.")
            except (AttributeError, ValueError, TypeError):
                # LIMIT is an expression we can't read (e.g. 2+100000): replace it with the cap
                node.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
                final_sql = node.sql(dialect=dialect, pretty=True)
                notes.append(f"Replaced an unreadable LIMIT with LIMIT {max_rows}.")

    return ValidatedSQL(sql=final_sql, kind=kind, statement=stmt, tables=referenced,
                        target_table=target, notes=notes, expression=node, dialect=dialect,
                        sensitive_columns=sensitive_references(node, sensitive or set()))


def write_preview_select(v: ValidatedSQL, limit: int = 20) -> str | None:
    """For UPDATE/DELETE, build a SELECT showing the rows that would be affected."""
    node = v.expression
    if not isinstance(node, (exp.Update, exp.Delete)) or v.target_table is None:
        return None
    where = node.args.get("where")
    q = exp.select("*").from_(v.target_table)
    if where is not None:
        q = q.where(where.this.copy())
    return q.limit(limit).sql(dialect=v.dialect)
