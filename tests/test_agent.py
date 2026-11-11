"""Offline tests: no API key needed. A scripted fake LLM stands in for Gemini.

Run:  python -m pytest -q
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.audit import AuditLog
from agent.llm import LLMError, LLMResponse, RateLimiter, parse_agent_json
from agent.pipeline import AskDataAgent
from agent.schema import introspect
from agent.validator import ValidationError, validate_sql
from data.seed import build_database


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def master_db(tmp_path_factory) -> Path:
    return build_database(tmp_path_factory.mktemp("db") / "store.db", n_orders=800)


@pytest.fixture
def db(master_db, tmp_path) -> Path:
    p = tmp_path / "copy.db"
    p.write_bytes(master_db.read_bytes())
    return p


@pytest.fixture
def schema(master_db):
    conn = sqlite3.connect(master_db)
    try:
        return introspect(conn)
    finally:
        conn.close()


def reply(intent="query", sql="", chart="table", x="", y="", explanation="ok", **kw):
    d = {"reasoning": "r", "intent": intent, "sql": sql, "explanation": explanation,
         "chart_type": chart, "x": x, "y": y, "color": "", "chart_title": "", "follow_ups": ["a", "b"]}
    d.update(kw)
    return json.dumps(d)


class FakeLLM:
    """Returns scripted responses in order and records what it was sent."""

    model = "fake"

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[list[tuple[str, str]]] = []

    def generate(self, system, turns):
        self.calls.append(turns)
        if not self.responses:
            raise LLMError("no more scripted responses")
        text = self.responses.pop(0)
        return LLMResponse(data=parse_agent_json(text), raw_text=text, model=self.model, attempts=1, latency_ms=1)


def make_agent(db, tmp_path, *responses):
    llm = FakeLLM(*responses)
    return AskDataAgent(db, llm, AuditLog("test", tmp_path / "audit.jsonl")), llm


# --------------------------------------------------------------------------- #
# Seed data
# --------------------------------------------------------------------------- #
def test_seed_has_recent_data(master_db):
    conn = sqlite3.connect(master_db)
    last_month = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE order_date >= date('now','start of month','-1 month') "
        "AND order_date < date('now','start of month')").fetchone()[0]
    returns = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    conn.close()
    assert last_month > 10
    assert returns > 50


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def test_select_gets_limit(schema):
    v = validate_sql("SELECT name FROM products", schema)
    assert v.kind == "read" and "LIMIT 1000" in v.sql.upper()


def test_cte_allowed(schema):
    v = validate_sql("WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t", schema)
    assert v.kind == "read"


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "ALTER TABLE orders ADD COLUMN x TEXT",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "CREATE TABLE x (a INT)",
    "SELECT * FROM sqlite_master",
    "SELECT load_extension('evil')",
])
def test_dangerous_sql_blocked(schema, sql):
    with pytest.raises(ValidationError) as ei:
        validate_sql(sql, schema, allow_writes=True)
    assert ei.value.policy


def test_multiple_statements_blocked(schema):
    with pytest.raises(ValidationError):
        validate_sql("SELECT 1; DELETE FROM orders WHERE 1=1", schema, allow_writes=True)


def test_writes_need_toggle(schema):
    with pytest.raises(ValidationError) as ei:
        validate_sql("DELETE FROM returns WHERE return_id = 1", schema, allow_writes=False)
    assert ei.value.policy
    v = validate_sql("DELETE FROM returns WHERE return_id = 1", schema, allow_writes=True)
    assert v.kind == "write" and v.target_table == "returns"


def test_delete_without_where_blocked(schema):
    with pytest.raises(ValidationError):
        validate_sql("DELETE FROM returns", schema, allow_writes=True)


def test_unknown_table_is_repairable(schema):
    with pytest.raises(ValidationError) as ei:
        validate_sql("SELECT * FROM sales", schema)
    assert not ei.value.policy and "Available tables" in str(ei.value)


# --------------------------------------------------------------------------- #
# LLM output parsing + rate limiter
# --------------------------------------------------------------------------- #
def test_parse_handles_code_fence():
    d = parse_agent_json("```json\n" + reply(sql="SELECT 1") + "\n```")
    assert d["sql"] == "SELECT 1"


def test_parse_rejects_bad_intent():
    with pytest.raises(LLMError):
        parse_agent_json(reply(intent="hack"))


def test_parse_rejects_non_json():
    with pytest.raises(LLMError):
        parse_agent_json("Sure! Here is your SQL: SELECT 1")


def test_rate_limiter_refuses_when_wait_too_long():
    rl = RateLimiter(max_calls=2, period=60)
    rl.acquire(); rl.acquire()
    with pytest.raises(LLMError):
        rl.acquire(max_wait=0.1)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
RETURN_RATE_SQL = """
SELECT p.name, ROUND(100.0 * COUNT(r.return_id) / COUNT(oi.order_item_id), 1) AS return_rate_pct
FROM order_items oi
JOIN orders o ON o.order_id = oi.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN returns r ON r.order_item_id = oi.order_item_id
WHERE o.order_date >= date('now','start of month','-1 month') AND o.order_date < date('now','start of month')
GROUP BY p.name ORDER BY return_rate_pct DESC LIMIT 10
"""


def test_answer_with_chart(db, tmp_path):
    agent, _ = make_agent(db, tmp_path, reply(sql=RETURN_RATE_SQL, chart="bar", x="name", y="return_rate_pct"))
    r = agent.ask("Which products had the highest return rate last month?")
    assert r.status == "answer", r.error
    assert len(r.df) == 10
    assert r.chart.type == "bar" and r.figure is not None


def test_self_correction_after_db_error(db, tmp_path):
    agent, llm = make_agent(
        db, tmp_path,
        reply(sql="SELECT product_name FROM products"),          # wrong column -> SQLite error
        reply(sql="SELECT name FROM products", chart="table"),   # fixed
    )
    r = agent.ask("list products")
    assert r.status == "answer"
    assert len(r.failed_attempts) == 1 and r.failed_attempts[0].stage == "execution"
    # The error was fed back to the model
    assert "no such column" in llm.calls[1][-1][1]


def test_gives_up_after_max_repairs(db, tmp_path):
    bad = reply(sql="SELECT nope FROM products")
    agent, _ = make_agent(db, tmp_path, bad, bad, bad)
    r = agent.ask("list products")
    assert r.status == "error" and len(r.failed_attempts) == 3


def test_clarify_passthrough(db, tmp_path):
    agent, _ = make_agent(db, tmp_path, reply(intent="clarify", explanation="By spend or by orders?"))
    r = agent.ask("best customers")
    assert r.status == "clarify" and r.df is None


def test_policy_block_is_not_retried(db, tmp_path):
    agent, llm = make_agent(db, tmp_path, reply(intent="modify", sql="DROP TABLE orders"))
    r = agent.ask("drop the orders table")
    assert r.status == "blocked" and len(llm.calls) == 1


def test_write_needs_confirmation_then_commits(db, tmp_path):
    sql = "UPDATE orders SET status = 'Refunded' WHERE order_id = 5"
    agent, _ = make_agent(db, tmp_path, reply(intent="modify", sql=sql))
    r = agent.ask("mark order 5 refunded", allow_writes=True)
    assert r.status == "needs_confirmation"
    assert r.write_preview.affected_rows == 1 and len(r.write_preview.preview) == 1

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT status FROM orders WHERE order_id = 5").fetchone()[0] != "Refunded"
    conn.close()

    agent.confirm(r)
    assert r.status == "committed"
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT status FROM orders WHERE order_id = 5").fetchone()[0] == "Refunded"
    conn.close()


def test_dry_run_does_not_change_data(db, tmp_path):
    sql = "DELETE FROM returns WHERE reason = 'Changed mind'"
    agent, _ = make_agent(db, tmp_path, reply(intent="modify", sql=sql))
    conn = sqlite3.connect(db)
    n_before = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    conn.close()
    r = agent.ask("delete changed-mind returns", allow_writes=True)
    assert r.status == "needs_confirmation" and r.write_preview.affected_rows > 0
    agent.cancel(r)
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0] == n_before
    conn.close()


def test_select_runs_on_read_only_connection(db, tmp_path):
    from agent.executor import run_read
    from agent.validator import READ, ValidatedSQL
    # Even if a write slipped past the validator, the read-only connection refuses it
    v = ValidatedSQL(sql="DELETE FROM returns WHERE 1=1", kind=READ, statement="SELECT")
    with pytest.raises(Exception):
        run_read(db, v)


def test_followup_history_is_sent(db, tmp_path):
    agent, llm = make_agent(db, tmp_path,
                            reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n"),
                            reply(sql="SELECT channel, COUNT(*) AS n FROM orders GROUP BY channel", chart="bar",
                                  x="channel", y="n"))
    first = agent.ask("how many orders?")
    agent.ask("by channel", history=[first])
    sent = llm.calls[1]
    assert ("user", "how many orders?") in sent


def test_audit_log_written(db, tmp_path):
    agent, _ = make_agent(db, tmp_path, reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n"))
    agent.ask("how many orders?")
    events = list(agent.audit.read()["event"])
    for ev in ("question", "llm_response", "validation", "execute", "result"):
        assert ev in events
