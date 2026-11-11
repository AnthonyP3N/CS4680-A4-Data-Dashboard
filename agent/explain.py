"""Explaining a result in plain words, from its SQL. No model call, so it can't make anything up.

`describe_query` reads a statement and lists what it does: which tables it
reads and how they are joined, what it measures, how it groups, which rows it
keeps, how it sorts and how many rows it returns. `explain_result` turns that,
plus what is known about the turn (row count, remembered context, business
rules), into a few sentences for someone who doesn't read SQL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp

if TYPE_CHECKING:
    from data.metadata import SemanticConfig

    from .pipeline import AgentResult

_AGG_WORDS = {"count": "the number of", "sum": "the total of", "avg": "the average of", "min": "the lowest",
              "max": "the highest"}


@dataclass
class QueryFacts:
    tables: list[str] = field(default_factory=list)
    joins: int = 0
    measures: list[str] = field(default_factory=list)       # human phrases, e.g. "the total of quantity * unit_price"
    groups: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)        # one per AND-ed condition
    group_filters: list[str] = field(default_factory=list)  # HAVING conditions
    order: list[str] = field(default_factory=list)          # e.g. "return_rate_pct, highest first"
    limit: int | None = None
    uses_cte: bool = False


def _sql(node: exp.Expression, dialect: str) -> str:
    return " ".join(node.sql(dialect=dialect).split())


def _conditions(node: exp.Expression | None, dialect: str) -> list[str]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _conditions(node.left, dialect) + _conditions(node.right, dialect)
    return [_sql(node, dialect)]


def _measure(projection: exp.Expression, dialect: str) -> str:
    """A simple aggregate in words ("the total of amount"); anything more complex as its formula."""
    aggs = list(projection.find_all(exp.AggFunc))
    if not aggs:
        return ""
    alias = projection.alias_or_name
    body = projection.this if isinstance(projection, exp.Alias) else projection
    if len(aggs) == 1 and body is aggs[0]:
        agg = aggs[0]
        name, inner = agg.key.lower(), agg.this
        if name == "count":
            target = "rows" if inner is None or isinstance(inner, exp.Star) else _sql(inner, dialect)
            phrase = f"the number of {target}"
        else:
            phrase = f"{_AGG_WORDS.get(name, name)} {_sql(inner, dialect) if inner is not None else ''}".strip()
        return f"{alias} = {phrase}" if alias else phrase
    return f"{alias} = {_sql(body, dialect)}" if alias else _sql(body, dialect)


def describe_query(sql: str, dialect: str = "sqlite") -> QueryFacts | None:
    """What a SELECT does, or None when it can't be read."""
    try:
        node = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # sqlglot raises several types; explaining must never fail
        return None
    if not isinstance(node, exp.Select):
        return None
    ctes = {c.alias_or_name.lower() for c in node.find_all(exp.CTE)}
    facts = QueryFacts(uses_cte=bool(ctes))
    for table in node.find_all(exp.Table):
        if table.name and table.name.lower() not in ctes and table.name not in facts.tables:
            facts.tables.append(table.name)
    facts.joins = len(list(node.find_all(exp.Join)))
    facts.measures = [m for m in (_measure(p, dialect) for p in node.expressions) if m]
    group = node.args.get("group")
    if group is not None:
        facts.groups = [_sql(g, dialect) for g in group.expressions]
    where = node.args.get("where")
    facts.filters = _conditions(where.this if where is not None else None, dialect)
    having = node.args.get("having")
    facts.group_filters = _conditions(having.this if having is not None else None, dialect)
    order = node.args.get("order")
    if order is not None:
        for ordered in order.expressions:
            direction = "highest first" if ordered.args.get("desc") else "lowest first"
            facts.order.append(f"{_sql(ordered.this, dialect)}, {direction}")
    limit = node.args.get("limit")
    try:
        facts.limit = int(limit.expression.this) if limit is not None else None
    except (AttributeError, TypeError, ValueError):
        facts.limit = None
    return facts


def explain_result(result: "AgentResult", dialect: str = "sqlite",
                   semantics: "SemanticConfig | None" = None) -> list[str]:
    """Plain sentences describing how this result was produced."""
    if not result.sql:
        return ["No query was run for this answer."]
    facts = describe_query(result.sql, dialect)
    if facts is None:
        return ["This answer came from a query that can't be summarised automatically. The SQL is shown below."]
    lines = []
    tables = ", ".join(facts.tables) or "the data"
    lines.append(f"It reads {tables}" + (f", joining {facts.joins + 1} tables together." if facts.joins else "."))
    if facts.measures:
        lines.append("It calculates " + "; ".join(facts.measures) + ".")
    if facts.groups:
        lines.append("It works this out separately for each " + ", ".join(facts.groups) + ".")
    if facts.filters:
        lines.append("It only includes rows where " + " and ".join(facts.filters) + ".")
    else:
        lines.append("It doesn't filter out any rows.")
    if facts.group_filters:
        lines.append("It then keeps only groups where " + " and ".join(facts.group_filters) + ".")
    if facts.order:
        lines.append("It sorts by " + "; then ".join(facts.order) + ".")
    if facts.limit is not None:
        lines.append(f"It returns at most {facts.limit:,} rows.")
    if result.row_count is not None:
        lines.append(f"The result had {result.row_count:,} row{'s' if result.row_count != 1 else ''}.")
    if semantics is not None:
        rules = list(semantics.default_filters) + list(semantics.ownership_rules)
        if rules:
            lines.append("These business rules for this data were given to the AI with the question: "
                         + "; ".join(rules) + ". Check the filters above to see which it applied.")
    if result.memories:
        lines.append("Remembered context was included: " + "; ".join(m.get("content", "") for m in result.memories) + ".")
    return lines
