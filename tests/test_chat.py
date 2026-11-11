"""Phase 5 tests: regenerate, retry, edit, duplicate, search, filters, tags, saved items and export."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from agent.conversation import ConversationError, categorize
from agent.llm import LLMError
from fakes import STORE_ANSWERS, KeywordLLM, SequenceLLM, open_app, reply
from services import META_FILE
from storage import Storage, identity
from storage.migrations import LATEST_VERSION, MIGRATIONS, Migration, migrate

pytestmark = pytest.mark.usefixtures("small_sample")

COUNT = reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n", explanation="Total orders.")
BY_STATUS = reply(sql="SELECT status, COUNT(*) AS n FROM orders GROUP BY status", chart="bar", x="status", y="n",
                  explanation="Orders by status.")
BY_CHANNEL = reply(sql="SELECT channel, COUNT(*) AS n FROM orders GROUP BY channel", chart="pie", x="channel", y="n",
                   explanation="Orders by channel.")
HOLD = reply(intent="modify", sql="UPDATE orders SET status = 'On hold' WHERE order_id = 7",
             explanation="This will change the status of order #7 to On hold.")


def stored(app, include_inactive=True):
    conversation = app.conversation.conversation
    return app.storage.messages.list(app.workspace.id, conversation.id, include_inactive=include_inactive)


def order_status(app, order_id=7):
    conn = sqlite3.connect(app.registry.sample.path)
    try:
        return conn.execute("SELECT status FROM orders WHERE order_id = ?", (order_id,)).fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Regenerate and retry
# --------------------------------------------------------------------------- #
def test_regenerate_replaces_the_answer_without_duplicating_the_question(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, BY_STATUS, BY_CHANNEL))
    conversation = app.conversation
    conversation.ask("how many orders?")
    first = conversation.ask("orders by what?")
    assert first.explanation == "Orders by status."

    second = conversation.regenerate(first.request_id)
    assert second.explanation == "Orders by channel." and second.request_id != first.request_id
    assert [r.question for r in conversation.history] == ["how many orders?", "orders by what?"]   # still two turns
    assert conversation.history[-1] is second

    # The model was asked the same question with only the earlier turn as context, not the replaced answer
    _, turns = app.llm.calls[-1]
    assert turns[-1] == ("user", "orders by what?")
    assert not any("Orders by status." in text for _, text in turns)

    everything = stored(app)
    assert [m.role for m in everything] == ["user", "assistant", "user", "assistant", "assistant"]
    assert [m.active for m in everything] == [True, True, True, False, True]         # old answer hidden, not deleted
    assert everything[4].parent_message_id == everything[2].id == second.user_message_id
    assert sum(1 for m in everything if m.role == "user" and m.content == "orders by what?") == 1

    reloaded = open_app(tmp_path, SequenceLLM(COUNT)).conversation.history
    assert [(r.question, r.explanation) for r in reloaded] == [("how many orders?", "Total orders."),
                                                              ("orders by what?", "Orders by channel.")]


def test_only_the_latest_answer_can_be_regenerated(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, BY_STATUS))
    first = app.conversation.ask("how many orders?")
    app.conversation.ask("by status?")
    with pytest.raises(ConversationError, match="latest"):
        app.conversation.regenerate(first.request_id)
    with pytest.raises(ConversationError):
        app.conversation.regenerate("no-such-turn")
    assert len(stored(app)) == 4                                   # nothing was added or hidden


def test_failed_request_can_be_retried(tmp_path):
    app = open_app(tmp_path, SequenceLLM(LLMError("Groq is overloaded or not responding right now."), COUNT))
    failed = app.conversation.ask("how many orders?")
    assert failed.status == "error" and "overloaded" in failed.error
    retried = app.conversation.regenerate(failed.request_id)
    assert retried.status == "answer" and retried.df["n"].iloc[0] == 250
    assert [r.status for r in app.conversation.history] == ["answer"]
    assert [m.status for m in stored(app, include_inactive=False)] == ["", "answer"]
    assert [r.status for r in open_app(tmp_path, SequenceLLM(COUNT)).conversation.history] == ["answer"]


def test_regenerating_a_pending_change_cancels_it_first_and_applies_at_most_once(tmp_path):
    app = open_app(tmp_path, SequenceLLM(HOLD, HOLD, COUNT))
    conversation = app.conversation
    before = order_status(app)
    pending = conversation.ask("put order 7 on hold", allow_writes=True)
    again = conversation.regenerate(pending.request_id, allow_writes=True)
    assert pending.status == "cancelled" and again.status == "needs_confirmation" and order_status(app) == before
    with pytest.raises(ConversationError):
        conversation.confirm(pending.request_id, allow_writes=True)           # the old one can no longer be applied

    conversation.confirm(again.request_id, allow_writes=True)
    assert order_status(app) == "On hold"
    with pytest.raises(ConversationError, match="already applied"):
        conversation.regenerate(again.request_id, allow_writes=True)          # an applied change keeps its record
    assert [m.status for m in stored(app)] == ["", "cancelled", "committed"]


def test_interrupted_request_leaves_nothing_half_saved(tmp_path):
    class Interrupted(BaseException):          # how a "Stop" arrives: not an ordinary error
        pass
    app = open_app(tmp_path, SequenceLLM(COUNT, Interrupted()))
    app.conversation.ask("how many orders?")
    with pytest.raises(Interrupted):
        app.conversation.ask("this one gets stopped")
    assert [r.question for r in app.conversation.history] == ["how many orders?"]
    assert [m.content for m in stored(app) if m.role == "user"] == ["how many orders?"]
    assert [r.question for r in open_app(tmp_path, SequenceLLM(COUNT)).conversation.history] == ["how many orders?"]


# --------------------------------------------------------------------------- #
# Edit and resubmit
# --------------------------------------------------------------------------- #
def test_editing_a_question_replaces_it_and_everything_after(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, BY_STATUS, BY_CHANNEL, COUNT))
    conversation = app.conversation
    conversation.ask("how many orders?")
    middle = conversation.ask("orders by status")
    conversation.ask("orders by channel")

    edited = conversation.edit_and_resubmit(middle.request_id, "how many orders again?")
    assert [r.question for r in conversation.history] == ["how many orders?", "how many orders again?"]
    assert edited.status == "answer"
    _, turns = app.llm.calls[-1]                                   # only the first turn was kept as context
    assert ("user", "how many orders?") in turns and not any("by status" in t or "by channel" in t for _, t in turns)

    everything = stored(app)
    assert [m.active for m in everything] == [True, True, False, False, False, False, True, True]
    assert conversation.title == "how many orders?"                # the first question didn't change
    reloaded = open_app(tmp_path, SequenceLLM(COUNT)).conversation.history
    assert [r.question for r in reloaded] == ["how many orders?", "how many orders again?"]


def test_editing_the_first_question_updates_an_automatic_title_only(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, BY_STATUS))
    first = app.conversation.ask("how many orders?")
    renamed_later = app.conversation.edit_and_resubmit(first.request_id, "orders by status please")
    assert app.conversation.title == "orders by status please"

    app.rename_conversation(app.conversation.conversation.id, "My own title")
    app.conversation.edit_and_resubmit(renamed_later.request_id, "how many orders in total?")
    assert app.conversation.title == "My own title"                # a title the user chose is left alone


def test_editing_is_refused_across_an_applied_change(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, HOLD, BY_STATUS))
    conversation = app.conversation
    first = conversation.ask("how many orders?")
    change = conversation.ask("put order 7 on hold", allow_writes=True)
    conversation.confirm(change.request_id, allow_writes=True)
    later = conversation.ask("orders by status")

    for turn in (first, change):
        with pytest.raises(ConversationError, match="data change was applied"):
            conversation.edit_and_resubmit(turn.request_id, "something else")
    assert len(conversation.history) == 3 and all(m.active for m in stored(app))
    conversation.edit_and_resubmit(later.request_id, "how many orders now?")       # after the change is fine
    assert [r.status for r in conversation.history] == ["answer", "committed", "answer"]
    with pytest.raises(ConversationError, match="Type the new question"):
        conversation.edit_and_resubmit(first.request_id, "   ")


def test_editing_abandons_an_unconfirmed_change(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, HOLD, BY_STATUS))
    first = app.conversation.ask("how many orders?")
    pending = app.conversation.ask("put order 7 on hold", allow_writes=True)
    before = order_status(app)
    app.conversation.edit_and_resubmit(first.request_id, "orders by status")
    assert pending.status == "cancelled" and order_status(app) == before and app.conversation.pending is None


# --------------------------------------------------------------------------- #
# Duplicate
# --------------------------------------------------------------------------- #
def test_duplicate_is_an_independent_copy_and_never_carries_a_pending_change(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT, HOLD, BY_STATUS))
    app.conversation.ask("how many orders?")
    app.conversation.ask("put order 7 on hold", allow_writes=True)
    original_id = app.conversation.conversation.id
    app.set_tags(original_id, ["ops"])
    before = order_status(app)

    copy = app.duplicate_conversation(original_id)
    assert copy.title == "Copy of how many orders?" and copy.id != original_id and copy.category
    assert [t.name for t in app.tags_for(copy.id)] == ["ops"]
    copied = app.open_conversation(copy.id)
    assert [r.question for r in copied.history] == ["how many orders?", "put order 7 on hold"]
    assert copied.history[1].status == "cancelled" and copied.pending is None and order_status(app) == before

    copied.ask("orders by status")                                 # the copy grows; the original does not
    assert len(app.storage.messages.list(app.workspace.id, original_id)) == 4
    assert len(app.storage.messages.list(app.workspace.id, copy.id)) == 6
    with pytest.raises(ConversationError):
        app.duplicate_conversation("not-a-real-id")


# --------------------------------------------------------------------------- #
# Categories, tags, search and filters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question, intent, expected", [
    ("Show monthly revenue for the last 12 months", "query", "Trends over time"),
    ("Which products had the highest return rate?", "query", "Rankings"),
    ("What share of orders came from each channel?", "query", "Breakdowns"),
    ("How many orders are there?", "query", "Totals and counts"),
    ("List the customers in Seattle", "query", "Lookups"),
    ("Cancel order 2999", "modify", "Data changes"),
    ("hmm", "query", "General"),
])
def test_categories_follow_the_first_question(question, intent, expected):
    assert categorize(question, intent) == expected


def build_library(tmp_path):
    """Four saved conversations with different questions, categories and dates."""
    app = open_app(tmp_path, SequenceLLM(COUNT))
    ids = {}
    for key, question in [("returns", "Which products had the highest return rate?"),
                          ("revenue", "Show monthly revenue for the last 12 months"),
                          ("count", "How many orders are there?"),
                          ("channels", "What share of orders came from each channel?")]:
        app.new_conversation()
        app.conversation.ask(question)
        ids[key] = app.conversation.conversation.id
    return app, ids


def titles(found):
    return [c.title for c, _ in found]


def test_search_finds_conversations_by_their_words(tmp_path):
    app, ids = build_library(tmp_path)
    assert titles(app.find_conversations(text="revenue")) == ["Show monthly revenue for the last 12 months"]
    assert titles(app.find_conversations(text="returned products")) == ["Which products had the highest return rate?"]
    assert titles(app.find_conversations(text="share channel")) == ["What share of orders came from each channel?"]
    assert len(app.find_conversations(text="COUNT orders")) == 4   # the saved SQL is searchable too
    assert app.find_conversations(text="zebra") == []
    assert len(app.find_conversations()) == 4                      # no words: everything, newest first
    conversation, snippet = app.find_conversations(text="revenue")[0]
    assert "revenue" in snippet.lower()

    app.rename_conversation(ids["count"], "Headcount of sales")
    assert titles(app.find_conversations(text="headcount")) == ["Headcount of sales"]           # titles are searched
    for hostile in ['revenue" OR 1=1 --', "NEAR(", '"', "revenue*) UNION SELECT", "%", "_"]:
        app.find_conversations(text=hostile)                       # never an error, never another workspace's rows


def test_filters_by_category_source_tag_date_and_archive(tmp_path):
    app, ids = build_library(tmp_path)
    assert titles(app.find_conversations(category="Rankings")) == ["Which products had the highest return rate?"]
    assert set(app.categories()) == {"Rankings", "Trends over time", "Totals and counts", "Breakdowns"}
    app.set_category(ids["count"], "Board pack")
    assert titles(app.find_conversations(category="Board pack")) == ["How many orders are there?"]

    app.set_tags(ids["returns"], ["quality", "Q4"])
    app.set_tags(ids["revenue"], ["Q4"])
    q4 = next(t for t in app.tags() if t.name == "Q4")
    assert set(titles(app.find_conversations(tag_id=q4.id))) == {"Which products had the highest return rate?",
                                                                 "Show monthly revenue for the last 12 months"}
    app.set_tags(ids["returns"], ["quality"])                      # removing a tag
    assert titles(app.find_conversations(tag_id=q4.id)) == ["Show monthly revenue for the last 12 months"]
    assert titles(app.find_conversations(text="revenue", tag_id=q4.id, category="Trends over time")) != []

    sample_id = app.registry.sample.info.id
    assert len(app.find_conversations(data_source_id=sample_id)) == 4
    assert app.find_conversations(data_source_id="another-source") == []

    today = date.today()                                           # the local calendar day, as a person sees it
    assert len(app.find_conversations(since=today, until=today)) == 4
    assert len(app.find_conversations(since=today.isoformat(), until=today.isoformat())) == 4
    assert app.find_conversations(since=today + timedelta(days=1)) == []
    assert app.find_conversations(until=today - timedelta(days=1)) == []

    app.archive_conversation(ids["channels"])
    assert len(app.find_conversations()) == 3
    assert titles(app.find_conversations(archived=True)) == ["What share of orders came from each channel?"]
    assert titles(app.find_conversations(text="channel", archived=True)) != []
    app.archive_conversation(ids["channels"], archived=False)
    assert len(app.find_conversations()) == 4


@pytest.mark.parametrize("offset_hours", [-11, -8, 0, 5.5, 14])
def test_date_filters_mean_local_days_in_any_time_zone(tmp_path, offset_hours):
    """Timestamps are stored in UTC. A filter for "today" must mean the person's today, even when
    their evening (or morning) falls on a different UTC date."""
    zone = timezone(timedelta(hours=offset_hours))
    app = open_app(tmp_path, SequenceLLM(COUNT))
    app.conversation.ask("how many orders?")
    their_today = datetime.now(zone).date()
    assert len(app.find_conversations(since=their_today, until=their_today, tz=zone)) == 1
    assert len(app.find_conversations(since=their_today - timedelta(days=7), tz=zone)) == 1
    assert app.find_conversations(until=their_today - timedelta(days=1), tz=zone) == []
    assert app.find_conversations(since=their_today + timedelta(days=1), tz=zone) == []


def test_day_bounds_convert_local_midnights_to_utc():
    from services import day_bounds
    pacific = timezone(timedelta(hours=-8))
    assert day_bounds("2026-11-10", "2026-11-10", pacific) == ("2026-11-10T08:00:00.000+00:00",
                                                               "2026-11-11T08:00:00.000+00:00")
    # 6 pm Pacific on the 10th is 02:00 UTC on the 11th: inside the Pacific 10th, outside the UTC 10th
    evening = "2026-11-11T02:00:00.000+00:00"
    since, until = day_bounds("2026-11-10", "2026-11-10", pacific)
    assert since <= evening < until
    since, until = day_bounds("2026-11-10", "2026-11-10", timezone.utc)
    assert not (since <= evening < until)
    assert day_bounds(None, None) == (None, None)


def test_search_ignores_hidden_versions_and_other_owners(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    first = app.conversation.ask("zebra migration numbers")
    app.conversation.edit_and_resubmit(first.request_id, "how many orders?")
    assert app.find_conversations(text="zebra") == []              # the replaced question is no longer findable
    assert len(app.find_conversations(text="orders")) == 1

    other = open_app(tmp_path, SequenceLLM(COUNT), owner=identity.owner_from_token("z" * 32))
    assert other.find_conversations(text="orders") == [] and other.find_conversations() == []


def test_related_conversations_share_words_with_the_latest_question(tmp_path):
    app, ids = build_library(tmp_path)
    app.new_conversation()
    app.conversation.ask("What was the revenue last month?")
    related = app.related_conversations()
    assert related and related[0].id == ids["revenue"]
    assert app.conversation.conversation.id not in [c.id for c in related]
    app.new_conversation()
    assert app.related_conversations() == []                       # nothing asked yet


# --------------------------------------------------------------------------- #
# Saved results and bookmarks
# --------------------------------------------------------------------------- #
def test_results_charts_and_queries_can_be_saved(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS, COUNT))
    chart_turn = app.conversation.ask("orders by status")
    number_turn = app.conversation.ask("how many orders?")
    app.save_item(chart_turn.request_id, "chart", "Status mix", "for the Monday review")
    app.save_item(chart_turn.request_id, "query")
    app.save_item(number_turn.request_id, "result", "Order count")
    with pytest.raises(ConversationError, match="no chart"):
        app.save_item(number_turn.request_id, "chart")
    with pytest.raises(ValueError):
        app.save_item(number_turn.request_id, "screenshot")
    with pytest.raises(ConversationError):
        app.save_item("unknown-turn", "result")

    saved = app.saved_items()
    assert [(i.kind, i.title) for i in saved] == [("result", "Order count"), ("query", "orders by status"),
                                                  ("chart", "Status mix")]
    assert [i.title for i in app.saved_items("chart")] == ["Status mix"] and saved[2].note == "for the Monday review"
    assert saved[0].conversation_id == app.conversation.conversation.id

    other = open_app(tmp_path, SequenceLLM(COUNT), owner=identity.owner_from_token("y" * 32))
    assert other.saved_items() == [] and other.delete_saved_item(saved[0].id) is False
    assert app.delete_saved_item(saved[0].id) is True and len(app.saved_items()) == 2
    app.delete_conversation(app.conversation.conversation.id)
    assert app.saved_items() == []                                 # bookmarks go when their conversation is deleted


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def test_export_as_json_markdown_and_html(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS, COUNT, BY_CHANNEL))
    app.conversation.ask("orders by status")
    second = app.conversation.ask("how many orders?")
    app.conversation.regenerate(second.request_id)                 # leaves a hidden version behind
    conversation_id = app.conversation.conversation.id
    app.set_tags(conversation_id, ["weekly"])

    name, media, body = app.export_conversation(conversation_id, "json")
    document = json.loads(body)
    assert name == "orders_by_status.json" and media == "application/json"
    assert document["format"] == "askdata.conversation" and document["tags"] == ["weekly"]
    assert document["data_source"] == "Sample store" and "generated by an AI model" in document["disclaimer"]
    assert [t["question"] for t in document["turns"]] == ["orders by status", "how many orders?"]
    assert document["turns"][1]["explanation"] == "Orders by channel."            # the visible version only
    first = document["turns"][0]
    assert first["columns"] == ["status", "n"] and len(first["rows"]) == first["row_count"]
    assert first["chart"]["type"] == "bar" and "FROM orders" in first["sql"]

    name, media, body = app.export_conversation(conversation_id, "markdown")
    text = body.decode()
    assert name.endswith(".md") and text.startswith("# orders by status")
    assert "## 1. orders by status" in text and "```sql" in text and "| status | n |" in text
    assert "Orders by status." in text and "Tags: weekly" in text

    name, media, body = app.export_conversation(conversation_id, "html")
    page = body.decode()
    assert name.endswith(".html") and media == "text/html" and page.startswith("<!doctype html>")
    assert "<table>" in page and "<th>status</th>" in page and "@media print" in page
    assert "<script" not in page.lower() and "http" not in page.lower()            # self-contained, no scripts

    with pytest.raises(ConversationError):
        app.export_conversation(conversation_id, "docx")
    with pytest.raises(ConversationError):
        app.export_conversation("someone-elses-conversation", "json")


def test_exports_escape_content_and_leave_out_unsaved_rows(tmp_path):
    hostile = reply(sql="SELECT '<img src=x onerror=alert(1)>' AS label, '| pipe |' AS other",
                    explanation="<script>alert('x')</script> done")
    app = open_app(tmp_path, KeywordLLM({**STORE_ANSWERS, "markup": hostile}))
    app.conversation.ask("<b>markup</b> test")
    sensitive = app.conversation.ask("who are our top customers?")
    names = list(sensitive.df["name"])
    conversation_id = app.conversation.conversation.id

    page = app.export_conversation(conversation_id, "html")[2].decode()
    assert "<script>alert" not in page and "<img src=x" not in page and "<b>markup</b>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;markup&lt;/b&gt;" in page
    markdown = app.export_conversation(conversation_id, "markdown")[2].decode()
    assert "\\| pipe \\|" in markdown

    for fmt in ("json", "markdown", "html"):
        exported = app.export_conversation(conversation_id, fmt)[2].decode()
        assert not any(name in exported for name in names), fmt       # customer names never leave in an export
        assert "sensitive" in exported
    document = json.loads(app.export_conversation(conversation_id, "json")[2])
    assert document["turns"][1]["rows"] is None and document["turns"][1]["rows_not_saved_reason"] == "sensitive column"


def test_export_all_includes_archived_conversations(tmp_path):
    app, ids = build_library(tmp_path)
    app.archive_conversation(ids["count"])
    name, media, body = app.export_all()
    document = json.loads(body)
    assert name == "my_workspace_conversations.json" and document["format"] == "askdata.workspace"
    assert len(document["conversations"]) == 4
    assert sum(1 for c in document["conversations"] if c["archived"]) == 1


# --------------------------------------------------------------------------- #
# Details kept with each answer
# --------------------------------------------------------------------------- #
def test_query_timing_and_counts_are_kept_with_each_answer(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    live = app.conversation.ask("orders by status")
    assert live.execution_ms is not None and live.execution_ms <= live.elapsed_ms and live.intent == "query"
    saved = open_app(tmp_path, SequenceLLM(COUNT)).conversation.history[0]
    assert (saved.execution_ms, saved.row_count, saved.intent, saved.model, saved.source_name) == \
           (live.execution_ms, live.row_count, "query", live.model, "Sample store")


# --------------------------------------------------------------------------- #
# Upgrading a database made by the previous version
# --------------------------------------------------------------------------- #
def make_version_1(path):
    """A metadata database exactly as Step 3 left it, with one conversation in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    migrate(conn, MIGRATIONS[:1])
    conn.execute("INSERT INTO workspaces VALUES ('w1', 'My workspace', '', 'local', '2026-11-01', '2026-11-01')")
    conn.execute("INSERT INTO conversations (id, workspace_id, title, created_at, updated_at, last_message_at) "
                 "VALUES ('c1', 'w1', 'Old conversation', '2026-11-01', '2026-11-01', '2026-11-01')")
    conn.execute("INSERT INTO messages (id, conversation_id, role, content, position, created_at) "
                 "VALUES ('m1', 'c1', 'user', 'quarterly churn by region', 1, '2026-11-01')")
    conn.close()


def test_version_1_database_is_upgraded_with_its_data_and_becomes_searchable(tmp_path):
    path = tmp_path / "data" / META_FILE
    make_version_1(path)
    storage = Storage.open(path)
    assert storage.persistent and storage.version == LATEST_VERSION >= 2
    assert list(path.parent.glob(f"{META_FILE}.v1.*.bak"))                      # backed up before upgrading
    messages = storage.messages.list("w1", "c1")
    assert [m.content for m in messages] == ["quarterly churn by region"] and messages[0].active
    assert storage.conversations.has_search_index()
    assert [c.title for c, _ in storage.conversations.find("w1", text="churn")] == ["Old conversation"]   # backfilled


def test_search_still_works_without_the_full_text_index(tmp_path):
    """Some SQLite builds lack FTS5. The optional part of the migration is skipped and search uses LIKE."""
    without_fts = MIGRATIONS[:1] + (Migration(2, MIGRATIONS[1].name, MIGRATIONS[1].statements,
                                              optional=("CREATE VIRTUAL TABLE message_search USING no_such_module(x)",
                                                        "CREATE TABLE should_not_exist (a)")),)
    path = tmp_path / "meta.db"
    conn = sqlite3.connect(path, isolation_level=None)
    assert migrate(conn, without_fts) == [1, 2]
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'should_not_exist'").fetchone()[0] == 0
    conn.close()

    storage = Storage.open(path)
    assert storage.persistent and not storage.conversations.has_search_index()
    workspace = storage.workspaces.ensure_default("local")
    from storage.models import Message
    conversation = storage.conversations.create(workspace.id, "Budget review")
    storage.messages.add(workspace.id, Message(conversation_id=conversation.id, role="user",
                                               content="travel spend by team at 100% of budget"))
    assert [c.title for c, _ in storage.conversations.find(workspace.id, text="travel team")] == ["Budget review"]
    assert [c.title for c, _ in storage.conversations.find(workspace.id, text="budget")] == ["Budget review"]
    assert storage.conversations.find(workspace.id, text="payroll") == []
    assert storage.conversations.find(workspace.id, text="100%") != []          # % is matched literally
