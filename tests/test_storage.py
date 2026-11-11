"""Phase 3 tests: the metadata database, migrations, isolation, and conversations that survive a reload."""

from __future__ import annotations

import io
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from agent.audit import AuditLog
from agent.conversation import ConversationError, snapshot, title_from
from data.adapters import DataSourceError
from data.seed import build_database
from fakes import STORE_ANSWERS, KeywordLLM, reply
from services import META_FILE, build_services
from storage import Storage, identity
from storage.database import MetadataDB
from storage.migrations import LATEST_VERSION, MIGRATIONS, Migration, MigrationError, current_version, migrate
from storage.models import Message, QueryRun

NO_SETTINGS = lambda name, default=None: default   # noqa: E731


@pytest.fixture(autouse=True)
def small_sample(tmp_path_factory, monkeypatch):
    """Every workspace's sample store is copied from one small database built for this test."""
    master = build_database(tmp_path_factory.mktemp("master") / "sample.db", n_orders=250)
    monkeypatch.setattr("data.registry.master_sample_db", lambda directory, builder=None: master)
    return master


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage.open(tmp_path / "data" / META_FILE)


def open_app(tmp_path, owner="local", answers=STORE_ANSWERS, **settings):
    """Build the services the way the app does on start-up, against the same data folder."""
    get = lambda name, default=None: settings.get(name, default)   # noqa: E731
    return build_services(get, owner_id=owner, llm=KeywordLLM(answers), data_dir=tmp_path / "data",
                          audit=AuditLog("s", tmp_path / "audit.jsonl"))


def raw(tmp_path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "data" / META_FILE)


def order_status(services, order_id=7):
    conn = sqlite3.connect(services.registry.sample.path)
    try:
        return conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
def test_fresh_install_reaches_the_latest_version(tmp_path):
    store = Storage.open(tmp_path / "new" / "meta.db")
    assert store.persistent and store.warning == "" and store.version == LATEST_VERSION
    with store.db.connect() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"workspaces", "data_sources", "conversations", "messages", "memory_items", "tags",
                "conversation_tags", "query_runs", "schema_migrations"} <= tables
        assert migrate(conn) == []                              # running again changes nothing
        assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == len(MIGRATIONS)


def test_upgrade_keeps_data_and_writes_a_backup(tmp_path, storage):
    workspace = storage.workspaces.ensure_default("local")
    storage.conversations.create(workspace.id, "Kept through the upgrade")
    newer = MIGRATIONS + (Migration(LATEST_VERSION + 1, "add a notes column",
                                    ("ALTER TABLE conversations ADD COLUMN notes TEXT NOT NULL DEFAULT ''",)),)
    with storage.db.connect() as conn:
        assert migrate(conn, newer, backup_path=storage.path) == [LATEST_VERSION + 1]
        assert current_version(conn) == LATEST_VERSION + 1
        assert conn.execute("SELECT title, notes FROM conversations").fetchone()[:] == ("Kept through the upgrade", "")
    backups = list(storage.path.parent.glob(f"{META_FILE}.v{LATEST_VERSION}.*.bak"))
    assert len(backups) == 1
    assert sqlite3.connect(backups[0]).execute("SELECT title FROM conversations").fetchone()[0] == "Kept through the upgrade"


def test_failed_migration_is_rolled_back_completely(storage):
    broken = MIGRATIONS + (Migration(LATEST_VERSION + 1, "half-finished", (
        "CREATE TABLE halfway (id TEXT)", "ALTER TABLE no_such_table ADD COLUMN x TEXT")),)
    with storage.db.connect() as conn:
        with pytest.raises(MigrationError, match="rolled back"):
            migrate(conn, broken)
        assert current_version(conn) == LATEST_VERSION
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'halfway'").fetchone()[0] == 0


def test_database_from_a_newer_release_is_left_alone(tmp_path, storage):
    with storage.db.connect() as conn:
        conn.execute("INSERT INTO schema_migrations VALUES (?, 'from the future', '2030-01-01')", (LATEST_VERSION + 5,))
    before = storage.path.read_bytes()
    reopened = Storage.open(storage.path)
    assert not reopened.persistent and "newer release" in reopened.warning
    assert storage.path.read_bytes() == before                   # not modified, not replaced
    assert reopened.workspaces.ensure_default("local").name      # ...and the app still works, in memory


def test_unusable_database_file_falls_back_without_touching_it(tmp_path):
    corrupt = tmp_path / "meta.db"
    corrupt.write_bytes(b"definitely not a database " * 50)
    store = Storage.open(corrupt)
    assert not store.persistent and "can't be used" in store.warning
    assert corrupt.read_bytes().startswith(b"definitely not a database")

    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    assert not Storage.open(blocker / "nested" / "meta.db").persistent     # the folder can't be created


def test_foreign_keys_are_enforced(storage):
    with storage.db.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO conversations (id, workspace_id, title, created_at, updated_at) "
                         "VALUES ('c', 'no-such-workspace', 't', 'now', 'now')")


def test_transaction_commits_everything_or_nothing():
    db = MetadataDB(None)
    with db.connect() as conn:
        migrate(conn)
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute("INSERT INTO workspaces (id, name, owner_id, created_at, updated_at) "
                         "VALUES ('w', 'n', 'o', 't', 't')")
            raise RuntimeError("something went wrong half way")
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# Repositories and isolation
# --------------------------------------------------------------------------- #
def test_workspaces_belong_to_their_owner(storage):
    mine = storage.workspaces.ensure_default("owner-a")
    assert storage.workspaces.ensure_default("owner-a").id == mine.id          # not created twice
    theirs = storage.workspaces.create("owner-b", "Their workspace")
    assert [w.id for w in storage.workspaces.list("owner-a")] == [mine.id]
    assert storage.workspaces.get("owner-a", theirs.id) is None
    assert storage.workspaces.rename("owner-a", theirs.id, "Hijacked") is False
    assert storage.workspaces.delete("owner-a", theirs.id) is False
    assert storage.workspaces.get("owner-b", theirs.id).name == "Their workspace"


def test_conversation_listing_archive_pin_and_delete(storage):
    ws = storage.workspaces.ensure_default("local")
    first = storage.conversations.create(ws.id, "First")
    second = storage.conversations.create(ws.id, "Second")
    third = storage.conversations.create(ws.id, "  Third '; DROP TABLE conversations; --  ")
    for c in (first, second, third):
        storage.messages.add(ws.id, Message(conversation_id=c.id, role="user", content="hi"))
    assert [c.title for c in storage.conversations.list(ws.id)][0].startswith("Third '; DROP TABLE")   # stored as text

    storage.conversations.update(ws.id, first.id, pinned=True)
    storage.conversations.update(ws.id, second.id, archived=True)
    assert [c.title for c in storage.conversations.list(ws.id)][0] == "First"            # pinned comes first
    assert second.id not in [c.id for c in storage.conversations.list(ws.id)]            # hidden when archived...
    assert storage.conversations.get(ws.id, second.id).archived                          # ...but still there
    assert [c.id for c in storage.conversations.list(ws.id, only_archived=True)] == [second.id]
    assert len(storage.conversations.list(ws.id, include_archived=True)) == 3
    storage.conversations.update(ws.id, second.id, archived=False)
    assert len(storage.conversations.list(ws.id)) == 3

    with pytest.raises(ValueError):
        storage.conversations.update(ws.id, first.id, data_source_id="someone-elses")    # not an editable field
    assert storage.conversations.delete(ws.id, third.id) is True
    assert storage.conversations.get(ws.id, third.id) is None and len(storage.conversations.list(ws.id)) == 2


def test_another_workspace_cannot_read_change_or_delete_anything(storage):
    a = storage.workspaces.ensure_default("owner-a")
    b = storage.workspaces.ensure_default("owner-b")
    conversation = storage.conversations.create(a.id, "Private")
    user = storage.messages.add(a.id, Message(conversation_id=conversation.id, role="user", content="secret plan"))
    answer = storage.messages.add(a.id, Message(conversation_id=conversation.id, role="assistant",
                                                parent_message_id=user.id, status="answer", sql="SELECT 1"))
    storage.query_runs.add(a.id, QueryRun(message_id=answer.id, sql="SELECT 1", validation_status="passed",
                                          execution_status="ok", row_count=1))
    tag = storage.tags.create(a.id, "finance")

    assert storage.conversations.get(b.id, conversation.id) is None
    assert storage.conversations.list(b.id) == []
    assert storage.messages.list(b.id, conversation.id) == [] and storage.messages.get(b.id, user.id) is None
    assert storage.query_runs.list(b.id, answer.id) == []
    assert storage.conversations.update(b.id, conversation.id, title="Renamed by B") is False
    assert storage.messages.update(b.id, user.id, content="overwritten") is False
    assert storage.conversations.delete(b.id, conversation.id) is False
    assert storage.tags.assign(b.id, conversation.id, tag.id) is False
    with pytest.raises(LookupError):
        storage.messages.add(b.id, Message(conversation_id=conversation.id, role="user", content="intrusion"))
    with pytest.raises(LookupError):
        storage.query_runs.add(b.id, QueryRun(message_id=answer.id, sql="x", validation_status="passed",
                                              execution_status="ok"))
    # A's data is exactly as it was
    assert [m.content for m in storage.messages.list(a.id, conversation.id)] == ["secret plan", ""]
    assert storage.conversations.get(a.id, conversation.id).title == "Private"


def test_messages_keep_their_order_and_deleting_cascades(storage):
    ws = storage.workspaces.ensure_default("local")
    conversation = storage.conversations.create(ws.id, "Ordered")
    ids = []
    for i in range(4):
        m = storage.messages.add(ws.id, Message(conversation_id=conversation.id,
                                                role="user" if i % 2 == 0 else "assistant", content=f"m{i}",
                                                chart_spec={"type": "bar"}, result_metadata={"row_count": i}))
        ids.append(m.id)
    listed = storage.messages.list(ws.id, conversation.id)
    assert [m.position for m in listed] == [1, 2, 3, 4] and [m.content for m in listed] == ["m0", "m1", "m2", "m3"]
    assert listed[1].chart_spec == {"type": "bar"} and listed[3].result_metadata == {"row_count": 3}
    assert storage.conversations.get(ws.id, conversation.id).last_message_at == listed[-1].created_at
    with pytest.raises(ValueError):
        storage.messages.update(ws.id, ids[0], sql="DROP TABLE x")       # stored SQL can't be rewritten via the API

    storage.query_runs.add(ws.id, QueryRun(message_id=ids[1], sql="SELECT 1", validation_status="passed",
                                           execution_status="ok"))
    storage.conversations.delete(ws.id, conversation.id)
    with storage.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM query_runs").fetchone()[0] == 0


def test_tags(storage):
    ws = storage.workspaces.ensure_default("local")
    conversation = storage.conversations.create(ws.id, "Tagged")
    tag = storage.tags.create(ws.id, "Quarterly review", "#FFD84D")
    assert storage.tags.create(ws.id, "Quarterly review").id == tag.id      # same name, same tag
    assert storage.tags.assign(ws.id, conversation.id, tag.id) is True
    assert storage.tags.assign(ws.id, conversation.id, tag.id) is False     # already tagged
    assert [t.name for t in storage.tags.for_conversation(ws.id, conversation.id)] == ["Quarterly review"]
    assert storage.tags.unassign(ws.id, conversation.id, tag.id) is True
    assert storage.tags.for_conversation(ws.id, conversation.id) == []
    with pytest.raises(ValueError):
        storage.tags.create(ws.id, "   ")


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_identity_modes_and_owner_ids():
    assert identity.resolve_mode(None, "localhost:8501") == "local"
    assert identity.resolve_mode(None, "https://askdata.streamlit.app/") == "link"   # never one shared owner in public
    assert identity.resolve_mode(None, None) == "link"
    assert identity.resolve_mode("LOGIN", "localhost") == "login"
    assert identity.resolve_mode("nonsense", "example.com") == "link"

    token = identity.new_token()
    assert identity.valid_token(token) and not identity.valid_token("short") and not identity.valid_token("a b" * 20)
    owner = identity.owner_from_token(token)
    assert owner == identity.owner_from_token(token) and token not in owner          # stable, and only a hash
    assert owner != identity.owner_from_token(identity.new_token())
    with pytest.raises(ValueError):
        identity.owner_from_token("../../etc/passwd")
    assert identity.owner_from_login("Ana@Example.com") == identity.owner_from_login(" ana@example.com ")
    assert "ana" not in identity.owner_from_login("ana@example.com")


# --------------------------------------------------------------------------- #
# Conversations survive a reload
# --------------------------------------------------------------------------- #
def test_conversation_is_there_after_restarting_the_app(tmp_path):
    first = open_app(tmp_path)
    assert first.persistent and first.conversation.history == []
    first.conversation.ask("Which products had the highest return rate?")
    first.conversation.ask("how many orders are there?")
    original = first.conversation.history

    second = open_app(tmp_path)                                   # brand-new objects, same folder
    reloaded = second.conversation.history
    assert [r.question for r in reloaded] == [r.question for r in original]
    assert second.conversation.title == "Which products had the highest return rate?"
    for before, after in zip(original, reloaded):
        assert (after.status, after.sql, after.explanation, after.request_id) == \
               (before.status, before.sql, before.explanation, before.request_id)
        assert after.restored and after.source_id == before.source_id and after.model == before.model
        assert after.chart.type == before.chart.type and after.row_count == before.row_count
        assert after.df.to_dict("list") == before.df.to_dict("list")              # the saved preview
    assert reloaded[0].figure is not None

    # Follow-ups after a reload still get the earlier turns as context
    second.conversation.ask("and how many orders was that?")
    _, turns = second.llm.calls[-1]
    assert ("user", "Which products had the highest return rate?") in turns


def test_each_turn_is_stored_as_two_messages_with_its_query_runs(tmp_path):
    # The first reply uses a column that doesn't exist; when sent the error, the model corrects itself
    app = open_app(tmp_path, answers={
        "your previous response": reply(sql="SELECT name FROM products", explanation="All product names."),
        "list products": reply(sql="SELECT product_name FROM products"),
    })
    conversation = app.conversation
    result = conversation.ask("list products")
    assert result.status == "answer" and len(result.failed_attempts) == 1

    messages = app.storage.messages.list(app.workspace.id, conversation.conversation.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    user, assistant = messages
    assert user.content == "list products" and assistant.parent_message_id == user.id
    assert (assistant.status, assistant.intent, assistant.request_id) == ("answer", "query", result.request_id)
    assert "SELECT" in assistant.sql and assistant.explanation == "All product names."
    assert assistant.result_metadata["row_count"] == 24 and assistant.result_metadata["source_name"] == "Sample store"
    assert result.message_id == assistant.id

    runs = app.storage.query_runs.list(app.workspace.id, assistant.id)
    assert [(r.validation_status, r.execution_status) for r in runs] == [("passed", "failed"), ("passed", "ok")]
    assert "no such column" in runs[0].error_message and runs[1].row_count == 24


def test_results_with_sensitive_columns_are_never_saved(tmp_path):
    app = open_app(tmp_path)
    result = app.conversation.ask("who are our top customers?")
    assert result.status == "answer" and result.sensitive_columns == ["customers.name"]
    names = list(result.df["name"])
    assert any("aren't saved" in note for note in result.notes)

    stored = (tmp_path / "data" / META_FILE).read_bytes()
    for name in names:
        assert name.encode() not in stored                         # no customer name anywhere in the database file

    reloaded = open_app(tmp_path).conversation
    turn = reloaded.history[0]
    assert turn.df is None and turn.restored and turn.sensitive_columns == ["customers.name"]
    assert turn.sql and turn.explanation                           # the question, SQL and explanation are kept
    reloaded.rerun(turn.request_id)                                # runs against the data again, on demand
    assert list(turn.df["name"]) == names and turn.figure is not None


def test_result_saving_can_be_switched_off(tmp_path):
    app = open_app(tmp_path, ASKDATA_STORE_RESULTS="off")
    app.conversation.ask("how many orders are there?")
    turn = open_app(tmp_path, ASKDATA_STORE_RESULTS="off").conversation.history[0]
    assert turn.df is None and turn.row_count == 1 and turn.explanation


def test_large_results_are_saved_as_a_capped_preview(tmp_path):
    app = open_app(tmp_path)
    full = app.conversation.ask("show every order")
    assert len(full.df) == 250
    turn = open_app(tmp_path).conversation.history[0]
    assert len(turn.df) == 200 and turn.preview_truncated and turn.row_count == 250
    assert snapshot(None) is None


def test_pasted_api_key_is_not_stored(tmp_path):
    app = open_app(tmp_path)
    app.conversation.ask("how many orders are there? my key is sk-abcdefghijklmnopqrstuvwxyz123456")
    stored = (tmp_path / "data" / META_FILE).read_bytes()
    assert b"sk-abcdefghijklmnopqrstuvwxyz123456" not in stored
    assert "sk-abc" not in app.conversation.title and "how many orders" in app.conversation.title
    assert title_from("word " * 40).endswith("...") and len(title_from("word " * 40)) <= 63


def test_business_context_and_uploads_persist(tmp_path):
    app = open_app(tmp_path)
    config = app.source.semantics
    config.glossary["Whale"] = "a customer with more than 20 orders"
    app.update_semantics(config)
    app.import_csv(io.StringIO("species,age\ncat,3\ndog,5\n"), "pets.csv")
    app.conversation.ask("how many pets of each species?")

    again = open_app(tmp_path)
    assert again.source.info.name == "Uploaded files"               # reopens where the owner left off
    assert again.source.schema().table_names == {"pets"}
    assert [r.question for r in again.conversation.history] == ["how many pets of each species?"]
    assert again.registry.sample.semantics.glossary["Whale"] == "a customer with more than 20 orders"
    assert again.registry.sample.semantics.few_shot                 # the worked examples came back too
    assert {s.info.id for s in again.sources} == {s.info.id for s in app.sources}   # same ids after a restart

    tables, conversations = again.uploads_summary()
    assert (tables, conversations) == (1, 1)
    uploads_file = again.registry.uploads.path
    again.remove_uploads()
    assert not uploads_file.exists() and again.source.info.is_sample
    assert [c.title for c in again.list_conversations()] == []      # the conversation about it went too
    assert [s.info.name for s in open_app(tmp_path).sources] == ["Sample store"]


def test_sample_data_is_rebuilt_when_it_is_from_another_day(tmp_path):
    app = open_app(tmp_path)
    app.registry.sample.commit_write("UPDATE orders SET status = 'On hold' WHERE order_id = 7")
    assert order_status(app) == "On hold"
    assert order_status(open_app(tmp_path)) == "On hold"            # same day: the edit is kept

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    conn = raw(tmp_path)
    conn.execute("UPDATE data_sources SET connection = replace(connection, ?, ?) WHERE is_sample = 1",
                 (date.today().isoformat(), yesterday))
    conn.commit()
    conn.close()
    refreshed = open_app(tmp_path)
    assert order_status(refreshed) != "On hold"
    assert refreshed.registry.sample.info.connection["built_on"] == date.today().isoformat()


# --------------------------------------------------------------------------- #
# Stored content is never trusted to run
# --------------------------------------------------------------------------- #
def test_pending_change_is_dry_run_again_after_a_reload_before_it_can_apply(tmp_path):
    app = open_app(tmp_path)
    pending = app.conversation.ask("put order 7 on hold", allow_writes=True)
    assert pending.status == "needs_confirmation"
    before = order_status(app)

    reopened = open_app(tmp_path)
    waiting = reopened.conversation.pending
    assert waiting is not None and waiting.restored and order_status(reopened) == before
    assert waiting.write_preview.affected_rows == 1 and len(waiting.write_preview.preview) == 1   # a fresh preview
    with pytest.raises(ConversationError):
        reopened.conversation.confirm(waiting.request_id, allow_writes=False)
    reopened.conversation.confirm(waiting.request_id, allow_writes=True)
    assert order_status(reopened) == "On hold"

    final = open_app(tmp_path)
    assert final.conversation.history[0].status == "committed" and final.conversation.pending is None
    runs = final.storage.query_runs.list(final.workspace.id, final.conversation.history[0].message_id)
    assert [r.execution_status for r in runs] == ["dry_run", "committed"]


@pytest.mark.parametrize("tampered", ["DROP TABLE orders", "DELETE FROM orders", "SELECT * FROM orders",
                                      "UPDATE no_such_table SET a = 1 WHERE b = 2"])
def test_tampered_pending_change_is_cancelled_not_run(tmp_path, tampered):
    app = open_app(tmp_path)
    app.conversation.ask("put order 7 on hold", allow_writes=True)
    conn = raw(tmp_path)                                            # someone edits the database file directly
    conn.execute("UPDATE messages SET sql = ? WHERE role = 'assistant'", (tampered,))
    conn.commit()
    conn.close()

    reopened = open_app(tmp_path)
    turn = reopened.conversation.history[0]
    assert turn.status == "cancelled" and "re-checked" in turn.error and reopened.conversation.pending is None
    sample = sqlite3.connect(reopened.registry.sample.path)
    assert sample.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 250
    assert open_app(tmp_path).conversation.history[0].status == "cancelled"      # and that outcome is saved


@pytest.mark.parametrize("tampered", ["DELETE FROM orders WHERE 1 = 1", "DROP TABLE orders",
                                      "SELECT * FROM sqlite_master", "UPDATE orders SET status = 'x' WHERE 1 = 1"])
def test_tampered_saved_query_cannot_be_run_again(tmp_path, tampered):
    app = open_app(tmp_path)
    app.conversation.ask("how many orders are there?")
    conn = raw(tmp_path)
    conn.execute("UPDATE messages SET sql = ? WHERE role = 'assistant'", (tampered,))
    conn.commit()
    conn.close()

    reopened = open_app(tmp_path)
    with pytest.raises(ConversationError, match="Couldn't run that again"):
        reopened.conversation.rerun(reopened.conversation.history[0].request_id)
    sample = sqlite3.connect(reopened.registry.sample.path)
    assert sample.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 250
    assert "failed" in list(reopened.audit.read()["status"])


# --------------------------------------------------------------------------- #
# Several conversations
# --------------------------------------------------------------------------- #
def test_new_open_rename_archive_and_delete(tmp_path):
    app = open_app(tmp_path)
    app.conversation.ask("how many orders are there?")
    first_id = app.conversation.conversation.id
    app.new_conversation()
    assert app.conversation.history == [] and app.conversation.conversation is None    # nothing stored until asked
    app.conversation.ask("Which products had the highest return rate?")
    second_id = app.conversation.conversation.id
    assert [c.id for c in app.list_conversations()] == [second_id, first_id]

    _, turns = app.llm.calls[-1]
    assert not any("how many orders" in text for _, text in turns)          # a new conversation starts clean

    reopened = app.open_conversation(first_id)
    assert [r.question for r in reopened.history] == ["how many orders are there?"]
    assert app.conversation is reopened

    assert app.rename_conversation(first_id, "Order count") and app.conversation.title == "Order count"
    assert app.pin_conversation(first_id) and app.list_conversations()[0].id == first_id
    assert app.archive_conversation(second_id)
    assert [c.id for c in app.list_conversations()] == [first_id]
    assert [c.id for c in app.list_conversations(only_archived=True)] == [second_id]    # archived, not gone
    assert [r.question for r in app.open_conversation(second_id).history] == ["Which products had the highest return rate?"]

    assert app.delete_conversation(second_id) is True
    assert app.conversation.history == [] and app.storage.conversations.get(app.workspace.id, second_id) is None
    with pytest.raises(ConversationError):
        app.open_conversation(second_id)
    assert [c.title for c in open_app(tmp_path).list_conversations()] == ["Order count"]


def test_opening_another_conversation_abandons_an_unconfirmed_change(tmp_path):
    app = open_app(tmp_path)
    app.conversation.ask("how many orders are there?")
    first_id = app.conversation.conversation.id
    app.new_conversation()
    pending = app.conversation.ask("put order 7 on hold", allow_writes=True)
    before = order_status(app)
    app.open_conversation(first_id)
    assert pending.status == "cancelled" and order_status(app) == before


# --------------------------------------------------------------------------- #
# Owners and workspaces are isolated end to end
# --------------------------------------------------------------------------- #
def test_two_owners_share_nothing(tmp_path):
    alice = open_app(tmp_path, owner=identity.owner_from_token("a" * 32))
    alice.conversation.ask("how many orders are there?")
    alice.import_csv(io.StringIO("secret_project\nApollo\n"), "plans.csv")
    alice_conversation = alice.list_conversations()[0].id

    bob = open_app(tmp_path, owner=identity.owner_from_token("b" * 32))
    assert bob.workspace.id != alice.workspace.id
    assert bob.list_conversations() == [] and bob.conversation.history == []
    assert [s.info.name for s in bob.sources] == ["Sample store"]              # no sign of Alice's upload
    assert bob.registry.work_dir != alice.registry.work_dir
    with pytest.raises(ConversationError):
        bob.open_conversation(alice_conversation)
    with pytest.raises(DataSourceError):
        bob.switch_workspace(alice.workspace.id)
    with pytest.raises(DataSourceError):
        bob.select_source(alice.registry.uploads.info.id)
    assert bob.delete_conversation(alice_conversation) is False
    assert bob.rename_conversation(alice_conversation, "Mine now") is False
    bob.conversation.ask("how many orders are there?")
    assert bob.audit.read()["question"].dropna().tolist() == ["how many orders are there?"]   # only Bob's log rows

    back = open_app(tmp_path, owner=identity.owner_from_token("a" * 32))
    assert len(back.list_conversations()) == 1 and back.registry.uploads is not None


def test_workspaces_of_one_owner_are_separate(tmp_path):
    app = open_app(tmp_path)
    app.conversation.ask("how many orders are there?")
    home = app.workspace
    project = app.create_workspace("  Q4 planning  ")
    assert project.name == "Q4 planning" and app.workspace.id == project.id
    assert app.list_conversations() == [] and app.conversation.history == []
    assert [s.info.name for s in app.sources] == ["Sample store"] and app.registry.sample.path.parent.name == project.id
    app.conversation.ask("Which products had the highest return rate?")
    assert set(app.audit.read()["event"]) >= {"switch_workspace", "question"}
    assert "how many orders are there?" not in app.audit.read().get("question", []).tolist()

    app.switch_workspace(home.id)
    assert [c.title for c in app.list_conversations()] == ["how many orders are there?"]
    assert {w.name for w in app.workspaces()} == {"My workspace", "Q4 planning"}

    with pytest.raises(DataSourceError):
        app.delete_workspace(home.id)                              # not the one you're in
    folder = tmp_path / "data" / "workspaces" / project.id
    assert folder.exists() and app.delete_workspace(project.id) is True
    assert not folder.exists() and {w.name for w in app.workspaces()} == {"My workspace"}


# --------------------------------------------------------------------------- #
# When saving isn't possible
# --------------------------------------------------------------------------- #
def test_app_works_without_a_usable_database(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / META_FILE).write_bytes(b"corrupt" * 100)
    app = open_app(tmp_path)
    assert not app.persistent and app.data_dir is None and "can't be used" in app.storage.warning
    assert app.conversation.ask("how many orders are there?").status == "answer"
    app.new_conversation()
    assert len(app.list_conversations()) == 1                      # works within the session...
    assert "storage_unavailable" in list(app.audit.read()["event"])
    assert (tmp_path / "data" / META_FILE).read_bytes() == b"corrupt" * 100
    assert not (tmp_path / "data" / "workspaces").exists()         # ...and nothing is written to the data folder


def test_answer_is_still_given_when_a_save_fails(tmp_path, monkeypatch):
    app = open_app(tmp_path)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(app.storage.messages, "add", fail)
    result = app.conversation.ask("how many orders are there?")
    assert result.status == "answer" and result.df is not None
    assert "couldn't be saved" in app.conversation.storage_error
    assert "storage_error" in list(app.audit.read()["event"])
