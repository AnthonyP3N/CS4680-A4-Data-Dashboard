"""Phase 4 tests: short-term memory, the rolling summary, long-term memory, retrieval and controls."""

from __future__ import annotations

import io
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agent import citations
from agent.context import ContextBudget, ContextBuilder
from agent.memory import (MEMORY_TYPES, MemoryActionError, MemoryService, ShortTermMemory, classify, estimate_tokens,
                          keywords, looks_private, summarize_turns)
from agent.pipeline import AgentResult
from fakes import STORE_ANSWERS, KeywordLLM, SequenceLLM, open_app, reply
from services import META_FILE
from storage import Storage, identity
from storage.migrations import LATEST_VERSION, MIGRATIONS, migrate
from storage.models import MemoryItem

pytestmark = pytest.mark.usefixtures("small_sample")

COUNT = reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n", explanation="Total orders.")
BY_STATUS = reply(sql="SELECT status, COUNT(*) AS order_count FROM orders WHERE channel = 'Web' GROUP BY status",
                  chart="bar", x="status", y="order_count", explanation="Web orders by status. More detail follows.")
HOLD = reply(intent="modify", sql="UPDATE orders SET status = 'On hold' WHERE order_id = 7",
             explanation="This will change the status of order #7 to On hold.")


@pytest.fixture
def memory() -> MemoryService:
    storage = Storage.in_memory()
    workspace = storage.workspaces.ensure_default("local")
    return MemoryService(storage, workspace.id, "local")


def system_prompt(app) -> str:
    return app.llm.calls[-1][0]


def contents(found) -> list[str]:
    return [m.item.content for m in found]


# --------------------------------------------------------------------------- #
# Adding, editing and deleting
# --------------------------------------------------------------------------- #
def test_memories_can_be_added_edited_and_deleted(memory):
    item = memory.add("  Always   exclude cancelled orders  ")
    assert item.content == "Always exclude cancelled orders" and item.memory_type == "filter"
    assert item.status == "active" and item.user_approved and item.confidence == 1.0
    assert [m.id for m in memory.list()] == [item.id]

    assert memory.update(item.id, content="Always exclude cancelled and test orders", memory_type="preference",
                         pinned=True) is True
    edited = memory.get(item.id)
    assert (edited.content, edited.memory_type, edited.pinned) == \
           ("Always exclude cancelled and test orders", "preference", True)
    assert edited.updated_at >= item.updated_at
    assert memory.update(item.id) is False                                  # nothing to change

    assert memory.delete(item.id) is True and memory.list() == [] and memory.delete(item.id) is False


def test_bad_memories_are_refused_and_secrets_masked(memory):
    for empty in ("", "  ", "ok"):
        with pytest.raises(MemoryActionError):
            memory.add(empty)
    with pytest.raises(MemoryActionError, match="Unknown kind"):
        memory.add("Revenue excludes tax", "spell")
    item = memory.add("The reporting key is sk-abcdefghijklmnopqrstuvwxyz123456, use the finance schema")
    assert "sk-abc" not in item.content and "finance schema" in item.content
    long = memory.add("word " * 400)
    assert len(long.content) <= 500
    with pytest.raises(MemoryActionError):
        memory.update(long.id, memory_type="nonsense")
    with pytest.raises(MemoryActionError):
        memory.update(long.id, data_source_id="a-source-in-another-workspace")


def test_saying_the_same_thing_twice_does_not_duplicate(memory):
    first = memory.add("Revenue excludes tax.")
    second = memory.add("revenue   excludes tax")
    assert second.id == first.id and len(memory.list()) == 1


@pytest.mark.parametrize("text, expected", [
    ("Always exclude cancelled orders", "filter"),
    ("When I say whales I mean customers with more than 20 orders", "definition"),
    ("Churn means a customer with no order in 90 days", "definition"),
    ("I prefer bar charts", "chart"),
    ("By default use the last 12 months", "date_range"),
    ("From now on round money to whole dollars", "preference"),
    ("The finance team owns this dashboard", "note"),
])
def test_statements_are_sorted_into_kinds(text, expected):
    assert classify(text) == expected and expected in MEMORY_TYPES


# --------------------------------------------------------------------------- #
# Suggestions need approval
# --------------------------------------------------------------------------- #
def test_suggestions_do_nothing_until_approved(memory):
    offered = memory.suggest_from_question("Show revenue by month, and always exclude cancelled orders")
    assert len(offered) == 1 and offered[0].status == "suggested" and not offered[0].user_approved
    assert memory.list() == [] and memory.retrieve("revenue by month", "src") == []      # not in use yet
    assert [m.id for m in memory.list(status="suggested")] == [offered[0].id]

    assert memory.approve(offered[0].id) is True
    assert contents(memory.retrieve("anything at all", "src")) == [offered[0].content]   # a standing instruction


@pytest.mark.parametrize("question, expected", [
    ("Which products sell best? From now on always show the top 10", "From now on always show the top 10"),
    ("Show revenue by month, and always exclude cancelled orders", "Always exclude cancelled orders"),
    ("How many orders are there; by default leave out test accounts.", "By default leave out test accounts"),
    ("Always exclude cancelled orders", "Always exclude cancelled orders"),
    ("What was revenue? I prefer bar charts for comparisons", "I prefer bar charts for comparisons"),
])
def test_only_the_instruction_is_suggested_not_the_whole_question(memory, question, expected):
    from agent.memory import standing_clause
    assert standing_clause(question) == expected
    assert [m.content for m in memory.suggest_from_question(question)] == [expected]


def test_rejected_suggestions_are_not_offered_again(memory):
    offered = memory.suggest_from_question("From now on show money in whole dollars")
    assert memory.reject(offered[0].id) is True
    assert memory.list() == [] and memory.list(status="suggested") == []
    assert memory.suggest_from_question("From now on show money in whole dollars") == []
    assert memory.retrieve("show money", "src") == []
    assert len(memory.list(status="rejected")) == 1                  # kept, so the same thing isn't suggested again


def test_definitions_and_chart_preferences_are_recognised(memory):
    definition = memory.suggest_from_question("When I say whales, I mean customers with more than 20 orders")
    assert definition[0].memory_type == "definition"
    assert definition[0].content == '"whales" means customers with more than 20 orders'
    chart = memory.suggest_from_question("I prefer line charts for anything over time")
    assert chart[0].memory_type == "chart"


@pytest.mark.parametrize("question", [
    "How many orders are there?",
    "Which products had the highest return rate last month?",
    "Show the top 10 customers",
    "What does the returns table mean?",
])
def test_ordinary_questions_suggest_nothing(memory, question):
    assert memory.suggest_from_question(question) == []


@pytest.mark.parametrize("question", [
    "Always exclude orders from jane.doe@example.com",
    "From now on ignore customer 555-010-9999",
    "By default use token sk-abcdefghijklmnopqrstuvwxyz123456",
])
def test_personal_data_and_secrets_are_never_suggested(memory, question):
    assert looks_private(question)
    assert memory.suggest_from_question(question) == [] and memory.list(status=None) == []


def test_a_repeated_time_range_is_offered_as_a_default_once(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    for question in ("orders in the last 12 months", "returns in the last 12 months"):
        app.new_conversation()
        assert app.conversation.ask(question).suggestions == []          # twice is not yet a habit
    app.new_conversation()
    third = app.conversation.ask("revenue for the last 12 months")
    assert [s["content"] for s in third.suggestions] == ["When no time range is given, use last 12 months"]
    assert third.suggestions[0]["type"] == "date_range"
    app.new_conversation()
    assert app.conversation.ask("profit in the last 12 months").suggestions == []     # offered once, not every time


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_relevant_memories_are_retrieved_and_irrelevant_ones_left_out(memory):
    churn = memory.add("Churn means a customer with no order in 90 days", "definition")
    memory.add("Whales are customers with more than 20 orders", "definition")
    memory.add("The marketing calendar starts in February", "note")

    found = memory.retrieve("What was churn last quarter?", "src")
    assert contents(found) == [churn.content]
    assert contents(memory.retrieve("how many whale customers placed orders?", "src"))[0].startswith("Whales")
    assert memory.retrieve("What is the weather in Seattle?", "src") == []
    assert memory.get(churn.id).last_used_at is not None                 # use is recorded


def test_standing_instructions_and_pinned_memories_always_apply(memory):
    preference = memory.add("Always exclude cancelled orders")
    pinned = memory.add("The fiscal year starts in July", "note", pinned=True)
    memory.add("Whales are customers with more than 20 orders", "definition")
    found = memory.retrieve("How many products are there?", "src")       # shares no words with any of them
    assert contents(found) == [pinned.content, preference.content]        # pinned first; the definition is left out


def test_ranking_prefers_overlap_origin_and_recency(memory):
    storage, workspace = memory.storage, memory.workspace_id
    here = storage.conversations.create(workspace, "This conversation")
    old = memory.add("Revenue means gross sales before refunds", "definition")
    close = memory.add("Net revenue means revenue after refunds and tax", "definition", conversation_id=here.id)
    with storage.db.connect() as conn:                                    # make the first one a year old
        conn.execute("UPDATE memory_items SET updated_at = ? WHERE id = ?",
                     ((datetime.now(timezone.utc) - timedelta(days=365)).isoformat(timespec="milliseconds"), old.id))
    found = memory.retrieve("show net revenue after refunds", "src", conversation_id=here.id)
    assert contents(found) == [close.content, old.content] and found[0].score > found[1].score


def test_token_budget_and_limit_cap_what_is_retrieved(memory):
    for i in range(12):
        memory.add(f"Always apply reporting rule number {i} to every revenue report without exception")
    everything = memory.retrieve("revenue report", "src", token_budget=10_000, limit=50)
    assert len(everything) == 12
    capped = memory.retrieve("revenue report", "src", token_budget=60, limit=50)
    assert 1 <= len(capped) < 12
    assert sum(estimate_tokens(m.item.content) + 6 for m in capped) <= 60
    assert len(memory.retrieve("revenue report", "src", token_budget=10_000, limit=3)) == 3
    assert memory.retrieve("revenue report", "src", token_budget=0) == []


def test_memory_can_be_switched_off_without_losing_it(memory):
    item = memory.add("Always exclude cancelled orders")
    assert memory.enabled
    memory.set_enabled(False)
    assert not memory.enabled and memory.retrieve("orders", "src") == []
    assert memory.suggest_from_question("From now on use whole dollars") == []
    assert [m.id for m in memory.list()] == [item.id]                     # still there to view, edit or delete
    memory.set_enabled(True)
    assert contents(memory.retrieve("orders", "src")) == [item.content]


# --------------------------------------------------------------------------- #
# Isolation: workspaces, owners and data sources
# --------------------------------------------------------------------------- #
def test_another_workspace_cannot_see_or_touch_memories():
    storage = Storage.in_memory()
    a = MemoryService(storage, storage.workspaces.ensure_default("owner-a").id, "owner-a")
    b = MemoryService(storage, storage.workspaces.ensure_default("owner-b").id, "owner-b")
    secret = a.add("Always exclude the Apollo project accounts")
    theirs = storage.conversations.create(a.workspace_id, "A's conversation")

    assert b.list(status=None) == [] and b.get(secret.id) is None
    assert b.retrieve("Apollo project accounts", "src") == []
    assert b.update(secret.id, content="overwritten by B") is False and b.delete(secret.id) is False
    assert b.approve(secret.id) is False and b.reject(secret.id) is False
    assert b.clear_all() == 0 and b.clear_conversation(theirs.id) == 0
    with pytest.raises(MemoryActionError):
        b.add("A note attached to someone else's conversation", conversation_id=theirs.id)
    b.set_enabled(False)
    assert a.enabled and a.get(secret.id).content == "Always exclude the Apollo project accounts"


def test_memories_follow_their_data_source(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    sample_id = app.source.info.id
    app.memory.add("Always exclude cancelled orders", data_source_id=sample_id)
    app.memory.add("Always round money to whole dollars")                 # no data source: applies everywhere
    app.import_csv(io.StringIO("species,age\ncat,3\ndog,5\n"), "pets.csv")
    uploads_id = app.source.info.id

    assert contents(app.memory.retrieve("how many?", uploads_id)) == ["Always round money to whole dollars"]
    assert set(contents(app.memory.retrieve("how many?", sample_id))) == {"Always exclude cancelled orders",
                                                                          "Always round money to whole dollars"}
    app.conversation.ask("how many pets?")
    assert "cancelled orders" not in system_prompt(app) and "whole dollars" in system_prompt(app)

    app.remove_uploads()                                                  # memories tied to it go with it
    app.memory.add("Pets data is refreshed weekly", data_source_id=sample_id)
    assert all(m.data_source_id in (None, sample_id) for m in app.memory.list(status=None))


def test_two_owners_never_share_memory(tmp_path):
    alice = open_app(tmp_path, SequenceLLM(COUNT), owner=identity.owner_from_token("a" * 32))
    alice.conversation.ask("Remember that the Apollo launch is confidential and excluded from revenue")
    bob = open_app(tmp_path, SequenceLLM(COUNT), owner=identity.owner_from_token("b" * 32))
    bob.conversation.ask("What is the revenue including the Apollo launch?")
    assert "Apollo launch is confidential" not in system_prompt(bob)
    assert bob.memory.list(status=None) == [] and bob.conversation.history[0].memories == []
    assert len(alice.memory.list()) == 1


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
def test_citations_say_where_a_memory_came_from():
    today = datetime(2026, 11, 20, tzinfo=timezone.utc)
    stamp = "2026-11-08T17:30:00.000+00:00"
    from_chat = MemoryItem("w", "preference", "x", conversation_id="c1", created_at=stamp)
    typed = MemoryItem("w", "note", "x", created_at=stamp)
    saved = MemoryItem("w", "saved_analysis", "x", conversation_id="c1", created_at=stamp)
    assert citations.describe(citations.memory_citation(from_chat), timezone.utc, today) == \
        "Based on a previous conversation from November 8"
    assert citations.describe(citations.memory_citation(typed), timezone.utc, today) == \
        "Based on a memory you added on November 8"
    assert citations.describe(citations.memory_citation(saved), timezone.utc, today) == \
        "Based on an analysis you saved on November 8"

    # The day is the reader's day: 02:00 UTC on the 9th is still the evening of the 8th in Los Angeles
    late = MemoryItem("w", "note", "x", conversation_id="c1", created_at="2026-11-09T02:00:00.000+00:00")
    pacific = timezone(timedelta(hours=-8))
    assert citations.describe(citations.memory_citation(late), pacific, today).endswith("November 8")
    assert citations.describe(citations.memory_citation(late), timezone.utc, today).endswith("November 9")
    assert citations.format_day("2025-03-02T10:00:00+00:00", timezone.utc, today) == "March 2, 2025"
    assert citations.format_day("not a date") == "an earlier date"

    query = citations.query_citation("Sample store", ["orders", "customers"])
    assert citations.describe(query) == "A query on Sample store (orders, customers)"
    round_trip = citations.Citation.from_dict(citations.memory_citation(from_chat).to_dict())
    assert round_trip == citations.memory_citation(from_chat)


# --------------------------------------------------------------------------- #
# The rolling summary
# --------------------------------------------------------------------------- #
def turn(question, status="answer", sql="", explanation="An answer."):
    return AgentResult(request_id=question, question=question, status=status, sql=sql, explanation=explanation)


def test_summary_records_questions_measures_groupings_filters_changes_and_open_questions():
    turns = [
        turn("monthly web revenue", sql="SELECT strftime('%Y-%m', order_date) AS month, SUM(amount) AS revenue "
                                        "FROM orders WHERE channel = 'Web' AND status != 'Cancelled' GROUP BY month",
             explanation="Web revenue by month. It peaks in November."),
        turn("put order 7 on hold", status="committed", explanation="Order 7 was put on hold."),
        turn("best customers?", status="clarify", explanation="Best by spend or by number of orders?"),
        turn("failed one", status="error"),
    ]
    summary = summarize_turns(turns)
    assert summary.startswith("Summary of 3 earlier turns in this conversation:")
    assert 'Asked "monthly web revenue": Web revenue by month.' in summary and "It peaks" not in summary
    assert "measures: revenue" in summary and "grouped by: month" in summary
    assert "filter: channel = 'Web' AND status <> 'Cancelled'" in summary
    assert 'Changed data after "put order 7 on hold"' in summary
    assert "the assistant asked: Best by spend or by number of orders?" in summary
    assert "failed one" not in summary
    assert summarize_turns([]) == "" and summarize_turns([turn("x", status="error")]) == ""


def test_summary_drops_the_oldest_turns_when_it_is_too_long():
    turns = [turn(f"question number {i} about a fairly long topic", explanation="A reasonably long answer. " * 3)
             for i in range(40)]
    summary = summarize_turns(turns, max_chars=600)
    assert len(summary) < 800 and "left out for length" in summary
    assert "question number 39" in summary and "question number 0 " not in summary


def test_summary_appears_once_a_conversation_outgrows_the_short_term_window(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    conversation = app.conversation
    for i in range(3):
        conversation.ask(f"question {i}")
    assert conversation.summary == "" and "Earlier in this conversation" not in system_prompt(app)

    conversation.ask("question 3")                                        # question 0 now falls out of the window
    assert conversation.summary.startswith("Summary of 1 earlier turn") and "question 0" in conversation.summary
    assert "filter: channel = 'Web'" in conversation.summary
    conversation.ask("question 4")
    system = system_prompt(app)
    assert "## Earlier in this conversation" in system and 'Asked "question 0"' in system
    assert conversation.history[-1].summary_used
    recent = [text for role, text in app.llm.calls[-1][1] if role == "user"]
    assert recent[-4:] == ["question 1", "question 2", "question 3", "question 4"]     # the window stays verbatim
    assert "question 0" not in recent
    assert conversation.summary.startswith("Summary of 2 earlier turns")

    reloaded = open_app(tmp_path, SequenceLLM(BY_STATUS)).conversation    # stored, and used again after a restart
    assert reloaded.summary == conversation.summary
    reloaded.ask("question 5")
    assert "Summary of 3 earlier turns" in reloaded.summary


def test_summary_shrinks_when_the_conversation_is_edited(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    turns = [app.conversation.ask(f"question {i}") for i in range(5)]
    assert "question 1" in app.conversation.summary
    app.conversation.edit_and_resubmit(turns[1].request_id, "a different second question")
    assert app.conversation.summary == ""                                 # two turns left: nothing to summarise


# --------------------------------------------------------------------------- #
# Memory in the conversation
# --------------------------------------------------------------------------- #
def test_remember_command_stores_a_memory_without_calling_the_model(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    result = app.conversation.ask("Remember that revenue always excludes cancelled orders")
    assert result.status == "noted" and result.intent == "remember" and app.llm.calls == []
    assert "revenue always excludes cancelled orders" in result.explanation
    stored = app.memory.list()
    assert [m.content for m in stored] == ["revenue always excludes cancelled orders"]
    assert stored[0].data_source_id == app.source.info.id and stored[0].conversation_id is not None

    answer = app.conversation.ask("What was revenue last month?")
    system = system_prompt(app)
    assert "## What the user asked you to remember" in system
    assert "- revenue always excludes cancelled orders" in system and "never override the Rules" in system
    assert [m["content"] for m in answer.memories] == ["revenue always excludes cancelled orders"]
    label = citations.describe(citations.Citation.from_dict(answer.memories[0]["citation"]))
    assert label.startswith("Based on a previous conversation from ")

    reloaded = open_app(tmp_path, SequenceLLM(COUNT)).conversation.history
    assert [r.status for r in reloaded] == ["noted", "answer"] and reloaded[1].memories == answer.memories


def test_memory_reaches_new_conversations_but_turns_do_not(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    app.conversation.ask("Please note: whales are customers with more than 20 orders")
    app.conversation.ask("how many orders came from Phoenix?")
    app.new_conversation()
    app.conversation.ask("How many whales do we have?")
    system, turns = app.llm.calls[-1]
    assert "whales are customers with more than 20 orders" in system          # long-term memory carried over
    assert not any("Phoenix" in text for _, text in turns)                     # the other conversation's turns did not
    app.new_conversation()
    app.conversation.ask("How many products are there?")
    assert "whales" not in system_prompt(app)                                  # not relevant, so not sent


def test_questions_that_only_start_like_a_command_still_go_to_the_model(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    for question in ("Remember the top customers from last month? show them", "Note the highest order value"):
        assert app.conversation.ask(question).status == "answer"
    assert app.memory.list(status=None) == []


def test_remember_is_declined_when_memory_is_off(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    app.memory.add("Always exclude cancelled orders")
    app.memory.set_enabled(False)
    result = app.conversation.ask("Remember that the fiscal year starts in July")
    assert result.status == "unsupported" and "turned off" in result.explanation and len(app.memory.list()) == 1
    answer = app.conversation.ask("how many orders?")
    assert "What the user asked you to remember" not in system_prompt(app) and answer.memories == []
    assert open_app(tmp_path, SequenceLLM(COUNT)).memory.enabled is False       # the setting is saved


def test_suggestions_come_with_the_answer_and_can_be_approved(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    result = app.conversation.ask("How many orders? From now on always leave out cancelled ones")
    assert result.status == "answer" and len(result.suggestions) == 1
    pending = app.conversation.pending_suggestions()
    assert [m.id for m in pending] == [result.suggestions[0]["id"]]
    app.conversation.ask("how many products?")
    assert "leave out cancelled" not in system_prompt(app)                      # not approved yet
    app.memory.approve(pending[0].id)
    app.conversation.ask("how many products?")
    assert "leave out cancelled" in system_prompt(app) and app.conversation.pending_suggestions() == []

    first = app.conversation.history[0]
    assert app.conversation.regenerate(app.conversation.history[-1].request_id).suggestions == []
    assert len(app.memory.list(status=None)) == 1 and first.suggestions                # no duplicates from re-asking


def test_results_and_sensitive_values_never_become_memory(tmp_path):
    app = open_app(tmp_path, KeywordLLM(STORE_ANSWERS))
    result = app.conversation.ask("who are our top customers? always show five")
    names = list(result.df["name"])
    for memory_item in app.memory.list(status=None):
        assert not any(name in memory_item.content for name in names)
    with pytest.raises(Exception):
        app.save_item("missing", "result")
    app.save_item(result.request_id, "query", "Top customers")
    assert [m for m in app.memory.list() if m.memory_type == "saved_analysis"] == []    # reads a sensitive column
    stored = (tmp_path / "data" / META_FILE).read_bytes()
    assert not any(name.encode() in stored for name in names)


def test_saving_an_analysis_makes_it_findable_later(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    turn_ = app.conversation.ask("web orders by status")
    app.save_item(turn_.request_id, "chart", "Web status mix")
    saved = [m for m in app.memory.list() if m.memory_type == "saved_analysis"]
    assert len(saved) == 1 and "Web status mix" in saved[0].content and "web orders by status" in saved[0].content
    assert "order_count" not in saved[0].content                              # the name and question, not the result

    app.new_conversation()
    answer = app.conversation.ask("show me that web status mix again")
    assert "Web status mix" in system_prompt(app)
    label = citations.describe(citations.Citation.from_dict(answer.memories[0]["citation"]))
    assert label.startswith("Based on an analysis you saved on ")
    app.new_conversation()
    app.conversation.ask("how many products?")
    assert "Web status mix" not in system_prompt(app)


def test_clearing_memory_for_one_conversation_and_for_the_workspace(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    app.conversation.ask("Remember that the fiscal year starts in July")
    for i in range(4):
        app.conversation.ask(f"question {i}")
    first_id = app.conversation.conversation.id
    assert app.conversation.summary
    app.new_conversation()
    app.conversation.ask("Remember that whales are customers with more than 20 orders")
    assert len(app.memory.list()) == 2

    first = app.open_conversation(first_id)
    assert first.summary and first.clear_memory() == 1
    assert first.summary == "" and len(first.history) == 5                    # the turns themselves stay
    assert [m.content for m in app.memory.list()] == ["whales are customers with more than 20 orders"]
    assert open_app(tmp_path, SequenceLLM(COUNT)).storage.conversations.get(app.workspace.id, first_id).summary == ""
    first.ask("question 4")                                                   # carrying on rebuilds it from the turns
    assert first.summary.startswith("Summary of 2 earlier turns")

    app.memory.suggest_from_question("From now on use whole dollars")
    assert app.memory.clear_all() == 2 and app.memory.list(status=None) == []
    app.conversation.ask("how many orders?")
    assert "What the user asked you to remember" not in system_prompt(app)


# --------------------------------------------------------------------------- #
# The context budget
# --------------------------------------------------------------------------- #
def test_budget_shares_and_token_estimates():
    budget = ContextBudget(total_tokens=1000, memory_share=0.3, summary_share=0.25)
    assert (budget.memory_tokens, budget.summary_tokens) == (300, 250)
    assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 1 and estimate_tokens("abcde") == 2
    assert keywords("Show the monthly Returns for products") == {"monthly", "return", "product"}


def test_prompt_never_exceeds_the_context_budget(tmp_path):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS), ASKDATA_CONTEXT_TOKENS="240")
    assert app.context_builder.budget.total_tokens == 240
    for i in range(10):
        app.memory.add(f"Always follow reporting convention number {i} when totals are shown in any report")
    for i in range(7):
        app.conversation.ask(f"question {i} " + "with a long tail of extra words " * 4)
    context = app.context_builder.build("one more question", app.conversation.history, app.source,
                                        summary=app.conversation.summary)
    tokens = context.tokens
    assert tokens["memories"] <= 240 * 0.30 and tokens["summary"] <= 240 * 0.25
    assert context.remembered >= 1                                            # the latest turn is always kept...
    assert context.remembered < 3                                             # ...older ones only while they fit
    assert 0 < len(context.memories) < 10
    assert tokens["budget"] == 240

    roomy = ContextBuilder(ShortTermMemory(), app.memory, ContextBudget(total_tokens=20_000))
    assert roomy.build("one more question", app.conversation.history, app.source).remembered == 3
    assert open_app(tmp_path, SequenceLLM(COUNT), ASKDATA_CONTEXT_TOKENS="nonsense").context_builder.budget.total_tokens == 2000


def test_what_went_into_the_prompt_is_logged_without_the_memory_text(tmp_path):
    app = open_app(tmp_path, SequenceLLM(COUNT))
    app.conversation.ask("Remember that the Apollo launch is excluded from revenue")
    app.conversation.ask("What is revenue without the Apollo launch?")
    log = (tmp_path / "audit.jsonl").read_text()
    assert '"event": "context"' in log and '"memory_added"' in log
    context_lines = [line for line in log.splitlines() if '"event": "context"' in line or "memory_" in line]
    assert context_lines and not any("Apollo launch is excluded" in line for line in context_lines)


# --------------------------------------------------------------------------- #
# Upgrading, and working without the full-text index
# --------------------------------------------------------------------------- #
def test_version_2_database_is_upgraded_and_keeps_its_memories(tmp_path):
    path = tmp_path / "data" / META_FILE
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path, isolation_level=None)
    migrate(conn, MIGRATIONS[:2])
    conn.execute("INSERT INTO workspaces (id, name, owner_id, created_at, updated_at) "
                 "VALUES ('w1', 'My workspace', 'local', '2026-11-01', '2026-11-01')")
    conn.execute("INSERT INTO memory_items (id, workspace_id, memory_type, content, user_approved, created_at, "
                 "updated_at) VALUES ('m1', 'w1', 'definition', 'Churn means no order in 90 days', 1, "
                 "'2026-11-01T00:00:00+00:00', '2026-11-01T00:00:00+00:00')")
    conn.close()

    storage = Storage.open(path)
    assert storage.persistent and storage.version == LATEST_VERSION >= 3
    assert list(path.parent.glob(f"{META_FILE}.v2.*.bak"))
    service = MemoryService(storage, "w1", "local")
    assert service.enabled                                                # settings default to "on"
    item = service.get("m1")
    assert (item.status, item.pinned, item.data_source_id) == ("active", False, None)
    assert contents(service.retrieve("what was churn last year?", "any-source")) == ["Churn means no order in 90 days"]


def test_retrieval_works_without_the_full_text_index(memory):
    with memory.storage.db.connect() as conn:
        for trigger in ("memory_search_insert", "memory_search_delete", "memory_search_update"):
            conn.execute(f"DROP TRIGGER {trigger}")
        conn.execute("DROP TABLE memory_search")
    churn = memory.add("Churn means a customer with no order in 90 days", "definition")
    memory.add("Budgets are tracked at 100% of plan", "note")
    assert contents(memory.retrieve("What was churn last quarter?", "src")) == [churn.content]
    assert memory.retrieve("the weather", "src") == []
    found = memory.storage.memories.search(memory.workspace_id, ["100%"])     # % is matched literally, not as "anything"
    assert len(found) == 1 and memory.storage.memories.search(memory.workspace_id, ["_"]) == {}
