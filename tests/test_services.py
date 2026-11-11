"""Tests for the layers under the UI: data sources, memory, context, the conversation service."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pytest

from agent.audit import AuditLog
from agent.context import ContextBuilder
from agent.conversation import ConversationError, ConversationService
from agent.memory import ShortTermMemory
from agent.pipeline import AgentResult, AskDataAgent
from agent.validator import validate_sql
from data.adapters import DataSourceError
from data.csv_adapter import import_csv, safe_identifier
from data.registry import DataSourceRegistry
from data.seed import build_database
from data.sqlite_adapter import SQLiteAdapter
from fakes import STORE_ANSWERS, KeywordLLM, reply
from services import build_services
from storage import Storage


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory) -> Path:
    """A directory holding a small master sample database, shared by all tests."""
    d = tmp_path_factory.mktemp("sample")
    build_database(d / "sample.db", n_orders=600)
    return d


@pytest.fixture
def registry(sample_dir, tmp_path, monkeypatch) -> DataSourceRegistry:
    # Reuse the small master copy instead of rebuilding the full one
    monkeypatch.setattr("data.registry.master_sample_db", lambda directory, builder=None: sample_dir / "sample.db")
    reg = DataSourceRegistry(sample_dir=sample_dir, work_dir=tmp_path / "work")
    reg.create_sample()
    return reg


@pytest.fixture
def source(registry) -> SQLiteAdapter:
    return registry.active


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog("test", tmp_path / "audit.jsonl")


def make_conversation(source, audit, answers=STORE_ANSWERS):
    llm = KeywordLLM(answers)
    return ConversationService(AskDataAgent(source, llm, audit), audit), llm


def order_status(source, order_id=7):
    conn = sqlite3.connect(source.path)
    try:
        return conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# SQLite adapter
# --------------------------------------------------------------------------- #
def test_adapter_schema_is_cached_and_refreshable(source):
    first = source.schema()
    assert source.schema() is first
    stamp = source.info.last_schema_refresh
    assert source.schema(refresh=True) is not first
    assert source.info.last_schema_refresh >= stamp
    assert {"orders", "products", "returns"} <= first.table_names


def test_adapter_preview_only_accepts_known_tables(source):
    assert len(source.preview("orders", limit=5)) == 5
    for bad in ("nope", 'orders"; DROP TABLE orders; --', "sqlite_master"):
        with pytest.raises(DataSourceError):
            source.preview(bad)
    assert "orders" in source.schema(refresh=True).table_names


def test_adapter_reads_through_validated_sql(source):
    v = validate_sql("SELECT COUNT(*) AS n FROM orders", source.schema())
    assert source.run_read(v).df["n"].iloc[0] > 0


def test_read_only_source_refuses_writes_at_every_level(sample_dir, tmp_path, audit):
    path = tmp_path / "ro.db"
    path.write_bytes((sample_dir / "sample.db").read_bytes())
    ro = SQLiteAdapter(path, name="Finance (read-only)", writable=False)
    v = validate_sql("UPDATE orders SET status = 'On hold' WHERE order_id = 7", ro.schema(), allow_writes=True)
    with pytest.raises(DataSourceError):
        ro.dry_run_write(v)
    with pytest.raises(DataSourceError):
        ro.commit_write(v.sql)

    # The pipeline blocks it too, even with data changes switched on
    agent = AskDataAgent(ro, KeywordLLM(STORE_ANSWERS), audit)
    r = agent.ask("put order 7 on hold", allow_writes=True)
    assert r.status == "blocked" and "read-only" in r.error
    assert order_status(ro) != "On hold"


def test_missing_database_file_is_reported(tmp_path):
    with pytest.raises(DataSourceError):
        SQLiteAdapter(tmp_path / "missing.db")


# --------------------------------------------------------------------------- #
# CSV import
# --------------------------------------------------------------------------- #
def test_csv_becomes_a_queryable_table(source, audit):
    csv = io.StringIO("Species,Age (years)\ncat,3\ndog,5\ncat,7\n")
    imported = import_csv(source, csv, "My Pets.csv")
    assert imported.table == "my_pets" and imported.rows == 3
    assert imported.columns == ["species", "age_years"]
    assert "my_pets" in source.schema().table_names

    answers = {"pets": reply(sql="SELECT species, COUNT(*) AS n FROM my_pets GROUP BY species ORDER BY n DESC",
                             chart="bar", x="species", y="n")}
    r = AskDataAgent(source, KeywordLLM(answers), audit).ask("how many pets of each species?")
    assert r.status == "answer" and list(r.df["n"]) == [2, 1]


def test_csv_names_cannot_carry_sql(source):
    hostile = io.StringIO('"id; DROP TABLE orders","x"" OR 1=1 --"\n1,2\n')
    imported = import_csv(source, hostile, 'orders"; DROP TABLE orders; --.csv')
    assert all(name.replace("_", "").isalnum() for name in imported.columns + [imported.table])
    assert "orders" in source.schema().table_names           # original table untouched
    assert safe_identifier("123 sales!", "x") == "t_123_sales"
    assert safe_identifier("???", "fallback") == "fallback"


def test_csv_duplicate_headers_and_name_clashes(source):
    imported = import_csv(source, io.StringIO("a,A,a \n1,2,3\n"), "orders.csv")
    assert imported.table == "orders_upload"                 # never replaces an existing table
    assert len(set(imported.columns)) == 3


@pytest.mark.parametrize("content", ["", "only_a_header\n"])
def test_empty_csv_is_rejected(source, content):
    with pytest.raises(DataSourceError):
        import_csv(source, io.StringIO(content), "empty.csv")


# --------------------------------------------------------------------------- #
# Registry: isolation between sessions
# --------------------------------------------------------------------------- #
def test_sessions_do_not_share_data(sample_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("data.registry.master_sample_db", lambda directory, builder=None: sample_dir / "sample.db")
    alice = DataSourceRegistry(sample_dir=sample_dir, work_dir=tmp_path / "alice").create_sample()
    bob = DataSourceRegistry(sample_dir=sample_dir, work_dir=tmp_path / "bob").create_sample()
    assert alice.path != bob.path and alice.info.id != bob.info.id

    import_csv(alice, io.StringIO("secret\n42\n"), "private.csv")
    alice.commit_write("UPDATE orders SET status = 'On hold' WHERE order_id = 7")
    assert "private" in alice.schema().table_names
    assert "private" not in bob.schema(refresh=True).table_names
    assert order_status(alice) == "On hold"
    assert order_status(bob) == order_status(SQLiteAdapter(sample_dir / "sample.db"))


def test_reset_restores_sample_and_cleanup_removes_files(registry, source):
    import_csv(source, io.StringIO("a\n1\n"), "extra.csv")
    registry.reset_sample(source)
    assert "extra" not in source.schema().table_names
    work_dir = registry.work_dir
    assert work_dir.exists()
    with pytest.raises(DataSourceError):
        registry.get("someone-elses-id")


# --------------------------------------------------------------------------- #
# Memory and context
# --------------------------------------------------------------------------- #
def turn(question, status):
    return AgentResult(request_id=question, question=question, status=status, sql="SELECT 1", explanation="e")


def test_short_term_memory_keeps_recent_useful_turns_only():
    history = [turn("q1", "answer"), turn("q2", "error"), turn("q3", "blocked"), turn("q4", "clarify"),
               turn("q5", "cancelled"), turn("q6", "committed"), turn("q7", "answer")]
    picked = ShortTermMemory(max_turns=3).select(history)
    assert [h.question for h in picked] == ["q4", "q6", "q7"]
    assert ShortTermMemory(max_turns=0).select(history) == []


def test_context_contains_schema_examples_memory_and_question(source):
    history = [turn("earlier question", "answer"), turn("failed question", "error")]
    ctx = ContextBuilder().build("new question", history, source)
    users = [text for role, text in ctx.turns if role == "user"]
    assert "CREATE TABLE orders" in ctx.system
    assert users[-1] == "new question" and "earlier question" in users and "failed question" not in users
    assert ctx.remembered == 1


@pytest.fixture
def hr_source(tmp_path) -> SQLiteAdapter:
    path = tmp_path / "hr.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, department TEXT, salary REAL)")
    conn.commit()
    conn.close()
    return SQLiteAdapter(path, name="HR")


def test_store_sql_examples_are_not_sent_for_other_datasets(hr_source):
    ctx = ContextBuilder().build("average salary by department", [], hr_source)
    examples = " ".join(text for _, text in ctx.turns[:-1])
    assert "order_items" not in examples and "FROM orders" not in examples


def test_prompt_is_domain_neutral_for_other_datasets(hr_source):
    ctx = ContextBuilder().build("average salary by department", [], hr_source)
    everything = (ctx.system + " ".join(text for _, text in ctx.turns[:-1])).lower()
    for store_term in ("order_items", "unit_price", "return rate", "store", "customer", "revenue", "product"):
        assert store_term not in everything
    assert "employees" in everything and "## business context" not in everything   # nothing defined yet


# --------------------------------------------------------------------------- #
# Conversation service
# --------------------------------------------------------------------------- #
def test_follow_up_sees_previous_turn(source, audit):
    conversation, llm = make_conversation(source, audit)
    conversation.ask("how many orders?")
    conversation.ask("and the return rate?")
    _, turns = llm.calls[1]
    assert ("user", "how many orders?") in turns
    assert len(conversation.history) == 2


def test_change_needs_confirmation_and_matching_request(source, audit):
    conversation, _ = make_conversation(source, audit)
    before = order_status(source)
    r = conversation.ask("put order 7 on hold", allow_writes=True)
    assert r.status == "needs_confirmation" and conversation.pending is r
    assert order_status(source) == before                    # dry run changed nothing

    with pytest.raises(ConversationError):
        conversation.confirm("some-other-id", allow_writes=True)
    assert order_status(source) == before

    conversation.confirm(r.request_id, allow_writes=True)
    assert r.status == "committed" and order_status(source) == "On hold"
    with pytest.raises(ConversationError):                    # can't be applied twice
        conversation.confirm(r.request_id, allow_writes=True)


def test_change_cannot_be_applied_after_switching_changes_off(source, audit):
    conversation, _ = make_conversation(source, audit)
    before = order_status(source)
    r = conversation.ask("put order 7 on hold", allow_writes=True)
    with pytest.raises(ConversationError, match="switched off"):
        conversation.confirm(r.request_id, allow_writes=False)
    assert order_status(source) == before and r.status == "needs_confirmation"
    assert "blocked" in list(audit.read()["status"])


def test_new_question_abandons_unconfirmed_change(source, audit):
    conversation, _ = make_conversation(source, audit)
    before = order_status(source)
    pending = conversation.ask("put order 7 on hold", allow_writes=True)
    conversation.ask("how many orders?")
    assert pending.status == "cancelled" and conversation.pending is None
    assert order_status(source) == before
    with pytest.raises(ConversationError):
        conversation.confirm(pending.request_id, allow_writes=True)


def test_cancel_and_clear(source, audit):
    conversation, _ = make_conversation(source, audit)
    before = order_status(source)
    r = conversation.ask("put order 7 on hold", allow_writes=True)
    conversation.cancel(r.request_id)
    assert r.status == "cancelled" and order_status(source) == before

    r2 = conversation.ask("put order 7 on hold", allow_writes=True)
    conversation.clear()
    assert conversation.history == [] and r2.status == "cancelled" and order_status(source) == before


# --------------------------------------------------------------------------- #
# Composition root
# --------------------------------------------------------------------------- #
def test_build_services_with_an_llm(registry, audit):
    services = build_services(lambda name, default=None: default, llm=KeywordLLM(STORE_ANSWERS),
                              registry=registry, audit=audit, storage=Storage.in_memory())
    assert services.ready and services.setup_error == "" and services.model_label == "fake · keyword"
    assert services.conversation.ask("how many orders?").status == "answer"

    services.import_csv(io.StringIO("a\n1\n"), "notes.csv")
    assert "upload_csv" in list(audit.read()["event"])
    services.reset()
    assert services.conversation.history == [] and "notes" not in services.source.schema().table_names


def test_build_services_without_a_key_still_offers_the_data(registry, audit):
    services = build_services(lambda name, default=None: default, registry=registry, audit=audit,
                          storage=Storage.in_memory())
    assert not services.ready and "No API key" in services.setup_error
    assert "orders" in services.source.schema().table_names
