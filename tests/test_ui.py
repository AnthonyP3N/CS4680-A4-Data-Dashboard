"""Interface tests: the real app.py is run headlessly with a scripted model."""

from __future__ import annotations

import html
import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.llm import PROVIDERS
from agent.pipeline import AgentResult, Attempt
from fakes import STORE_ANSWERS, KeywordLLM
from ui import theme
from ui.chat_view import describe_steps

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """No real API keys, and the activity log goes to a temp file."""
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider["key"], raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ASKDATA_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASKDATA_DATA_DIR", str(tmp_path / "data"))     # saved conversations go to a temp folder
    monkeypatch.setenv("ASKDATA_AUTH", "local")
    monkeypatch.delenv("ASKDATA_STORE_RESULTS", raising=False)


@pytest.fixture
def llm(monkeypatch) -> KeywordLLM:
    fake = KeywordLLM(STORE_ANSWERS)
    monkeypatch.setattr("agent.llm.build_client", lambda get_secret: fake)
    return fake


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def page_html(at: AppTest) -> str:
    """All markdown/HTML on the page, with entities such as &#x27; decoded back to text."""
    return html.unescape("\n".join(m.value for m in at.markdown))


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def toggle(at: AppTest, label: str = "Allow data changes"):
    return next(t for t in at.toggle if t.label == label)


def selectbox(at: AppTest, label: str):
    return next(s for s in at.selectbox if s.label == label)


def has_selectbox(at: AppTest, label: str) -> bool:
    return any(s.label == label for s in at.selectbox)


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
def test_empty_state_shows_prompt_and_starter_questions(llm):
    at = start()
    page = page_html(at)
    assert "What do you want to know?" in page and 'class="wordmark"' in page
    assert [t.label for t in at.tabs] == ["Ask", "Saved", "Memory", "Data", "Activity log", "Settings"]
    assert button(at, "Which products had the highest return rate last month?").disabled is False
    assert at.chat_input[0].disabled is False


def test_visitor_is_asked_for_a_key_when_the_app_has_none():
    at = start()
    page = page_html(at)
    assert "One step before you start." in page and "Setup needed." not in page
    assert any(t.label == "API key" for t in at.text_input)
    assert at.chat_input[0].disabled is True
    assert button(at, "Which products had the highest return rate last month?").disabled is True


def test_setup_state_when_no_key_and_personal_keys_are_off(monkeypatch):
    monkeypatch.setenv("ASKDATA_ALLOW_USER_KEYS", "off")
    at = start()
    page = page_html(at)
    assert "Setup needed." in page and "person running this app" in page
    assert not any(t.label == "API key" for t in at.text_input) and at.chat_input[0].disabled is True


def test_error_state_when_the_model_fails(monkeypatch):
    broken = KeywordLLM({}, fail_with="Groq is overloaded or not responding right now.")
    monkeypatch.setattr("agent.llm.build_client", lambda get_secret: broken)
    at = ask(start(), "how many orders?")
    page = page_html(at)
    assert "Couldn't answer that." in page and "overloaded" in page and "Model unavailable" in page


# --------------------------------------------------------------------------- #
# Chat flows
# --------------------------------------------------------------------------- #
def test_chart_answer_metric_answer_and_clarification(llm):
    at = ask(start(), "Which products had the highest return rate last month?")
    assert len(at.get("plotly_chart")) == 1 and "Drew bar chart" in page_html(at)
    assert "What do you want to know?" not in page_html(at)

    ask(at, "how many orders are there?")
    assert 'class="metric"' in page_html(at) and "Showed the number" in page_html(at)

    ask(at, "who are our best customers?")
    assert "notice ask" in page_html(at)
    assert [r.status for r in at.session_state.services.conversation.history] == ["answer", "answer", "clarify"]
    assert button(at, "Next question A") is not None        # follow-up suggestions on the latest turn


def test_starter_button_asks_its_question(llm):
    at = start()
    button(at, "Which products had the highest return rate last month?").click().run()
    assert not at.exception
    assert at.session_state.services.conversation.history[0].question.startswith("Which products")


def test_change_is_refused_while_changes_are_off(llm):
    at = ask(start(), "put order 7 on hold")
    assert "Not run." in page_html(at) and "Blocked by safety check" in page_html(at)
    assert at.session_state.services.conversation.history[0].status == "blocked"


def test_change_is_previewed_then_applied_on_confirmation(llm):
    at = start()
    toggle(at).set_value(True).run()
    ask(at, "put order 7 on hold")
    assert 'class="confirm"' in page_html(at) and "change 1 row" in page_html(at)

    source = at.session_state.services.source
    conn = sqlite3.connect(source.path)
    assert conn.execute("SELECT status FROM orders WHERE order_id = 7").fetchone()[0] != "On hold"

    button(at, "Apply change").click().run()
    assert not at.exception and "Change applied." in page_html(at)
    assert conn.execute("SELECT status FROM orders WHERE order_id = 7").fetchone()[0] == "On hold"
    conn.close()


def test_apply_is_disabled_once_changes_are_switched_off(llm):
    at = start()
    toggle(at).set_value(True).run()
    ask(at, "put order 7 on hold")
    toggle(at).set_value(False).run()
    assert button(at, "Apply change").disabled is True
    assert "Data changes are switched off" in page_html(at)
    button(at, "Cancel").click().run()
    assert "Change cancelled." in page_html(at)


def test_destructive_statement_is_blocked_even_with_changes_on(llm):
    at = start()
    toggle(at).set_value(True).run()
    ask(at, "drop the orders table")
    assert "DROP" in page_html(at) and "not allowed" in page_html(at)
    assert "orders" in at.session_state.services.source.schema(refresh=True).table_names


def test_new_conversation_starts_fresh_and_keeps_the_old_one(llm):
    at = ask(start(), "how many orders are there?")
    button(at, "New conversation").click().run()
    assert not at.exception
    assert at.session_state.services.conversation.history == []
    assert "What do you want to know?" in page_html(at)
    assert [c.title for c in at.session_state.services.list_conversations()] == ["how many orders are there?"]


# --------------------------------------------------------------------------- #
# Other tabs
# --------------------------------------------------------------------------- #
def test_data_tab_lists_tables_and_activity_log_records_steps(llm):
    at = ask(start(), "how many orders are there?")
    assert any("Sample store" in m.value and "5 tables" in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert any(label.startswith("orders") for label in labels)
    events = list(at.session_state.services.audit.read()["event"])
    for expected in ("session_start", "llm_config", "question", "validation", "execute", "result"):
        assert expected in events


# --------------------------------------------------------------------------- #
# Presentation logic and theme (no browser needed)
# --------------------------------------------------------------------------- #
def result(status, **kw) -> AgentResult:
    return AgentResult(request_id="r", question="q", status=status, **kw)


def test_step_trail_describes_each_outcome():
    labels = lambda r: [label for _, _, label in describe_steps(r)]
    assert labels(result("error")) == ["Model unavailable"]
    assert labels(result("clarify")) == ["Understood", "Needs clarification"]
    assert labels(result("unsupported"))[-1] == "Out of scope"
    assert labels(result("blocked", sql="DROP TABLE x", statement="DROP"))[-1] == "Blocked by safety check"
    assert labels(result("needs_confirmation", sql="UPDATE t SET a=1 WHERE b=2", statement="UPDATE"))[-1] \
        == "Waiting for your confirmation"
    fixed = result("answer", sql="SELECT 1", statement="SELECT", failed_attempts=[Attempt("x", "execution", "e")])
    assert "Self-corrected 1×" in labels(fixed)


def test_stylesheet_is_built_from_tokens_and_safe_for_markdown():
    for name in theme.THEMES:
        css = theme.stylesheet(name)
        for token, value in theme.PALETTES[name].items():
            assert f"--{token}:{value};" in css
        assert "/*" not in css and "\n\n" not in css         # comments or blank lines would leak onto the page
        assert "prefers-reduced-motion" in css and ":focus-visible" in css and "max-width: 640px" in css
    assert theme.stylesheet("dark") != theme.stylesheet("light") and theme.stylesheet("nonsense") == theme.stylesheet()


# --------------------------------------------------------------------------- #
# Several data sources and the business context editor
# --------------------------------------------------------------------------- #
def test_uploaded_data_gets_its_own_source_selector_and_starter_questions(llm):
    import io
    at = ask(start(), "how many orders are there?")
    assert not has_selectbox(at, "Data source")                # one source: nothing to choose
    services = at.session_state.services
    sample_id = services.source.info.id
    services.import_csv(io.StringIO("species,age\ncat,3\ndog,5\ncat,9\n"), "pets.csv")
    at.run()
    assert not at.exception

    assert selectbox(at, "Data source").value == services.source.info.id
    page = page_html(at)
    assert "What do you want to know?" in page and "Tables created from the files you uploaded." in page
    labels = [b.label for b in at.button]
    assert "How many pets are there for each species?" in labels
    assert "Which products had the highest return rate last month?" not in labels

    ask(at, "how many pets of each species?")
    assert len(at.get("plotly_chart")) == 1
    assert at.session_state.services.conversation.history[0].source_name == "Uploaded files"

    selectbox(at, "Data source").set_value(sample_id).run()   # back to the sample: its own conversation returns
    assert not at.exception
    assert [r.question for r in at.session_state.services.conversation.history] == ["how many orders are there?"]
    assert "Which products had the highest return rate last month?" in [b.label for b in at.button]


def test_business_context_can_be_edited_and_saved(llm):
    at = start()
    page = page_html(at)
    assert "Business context" in page and "hidden from the AI" not in page
    description = next(t for t in at.text_area if t.label == "What this data is")
    description.set_value("A demo shop used for training.")
    filters = next(t for t in at.text_area if t.label == "Default filters (one per line)")
    filters.set_value("orders.status != 'Cancelled'\n\n  orders.channel != 'Marketplace'  ")
    button(at, "Save business context").click().run()
    assert not at.exception

    semantics = at.session_state.services.source.semantics
    assert semantics.description == "A demo shop used for training."
    assert semantics.default_filters == ["orders.status != 'Cancelled'", "orders.channel != 'Marketplace'"]
    assert semantics.metrics and semantics.few_shot and "customers.email" in semantics.sensitive_columns   # kept
    assert any("Business context saved." in s.value for s in at.success)
    assert "A demo shop used for training." in page_html(at)                                                # empty state


def test_sensitive_columns_are_marked_in_the_data_tab(llm):
    at = start()
    frames = [d.value for d in at.dataframe]
    marked = [f for f in frames if "values" in f.columns and (f["values"] == "hidden from the AI (sensitive)").any()]
    assert marked, "the customers table should show name and email as hidden"
    assert "sensitive: values hidden" in at.session_state.services.schema_text()


# --------------------------------------------------------------------------- #
# Saved conversations
# --------------------------------------------------------------------------- #
def test_conversation_and_chart_are_back_after_reloading_the_page(llm):
    first = ask(start(), "Which products had the highest return rate last month?")
    ask(first, "how many orders are there?")

    second = start()                                           # a new browser session on the same data folder
    page = page_html(second)
    assert "Which products had the highest return rate last month?" in page and "how many orders are there?" in page
    assert len(second.get("plotly_chart")) == 1 and 'class="metric"' in page
    assert "What do you want to know?" not in page and "aren't being saved" not in page
    recent = [b for b in second.button if b.label == "Which products had the highest return rate last month?"
              and b.key and b.key.startswith("conv_")]
    assert len(recent) == 1 and recent[0].disabled             # listed under Recent, and it is the open one

    button(second, "New conversation").click().run()
    assert "What do you want to know?" in page_html(second)
    reopen = next(b for b in second.button if b.key and b.key.startswith("conv_"))
    assert reopen.disabled is False
    reopen.click().run()
    assert not second.exception and "how many orders are there?" in page_html(second)


def test_sensitive_answer_is_not_saved_but_can_be_run_again(llm):
    ask(start(), "who are our top customers?")
    reloaded = start()
    page = page_html(reloaded)
    assert "The five customers with the most orders." in page
    assert "weren't saved because it includes a column marked sensitive" in page
    assert len(reloaded.get("plotly_chart")) == 0
    button(reloaded, "Run again").click().run()
    assert not reloaded.exception and len(reloaded.get("plotly_chart")) == 1


def test_private_links_keep_visitors_apart(llm, monkeypatch):
    monkeypatch.setenv("ASKDATA_AUTH", "link")

    def visit(token=None):
        at = AppTest.from_file(APP, default_timeout=60)
        if token:
            at.query_params["u"] = token
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    alice = visit()
    alice_token = alice.query_params["u"][0] if isinstance(alice.query_params["u"], list) else alice.query_params["u"]
    assert len(alice_token) >= 24                              # a private link was created for this browser
    ask(alice, "how many orders are there?")

    stranger = visit()                                         # no link: a different, empty identity
    assert "how many orders are there?" not in page_html(stranger)
    assert stranger.session_state.services.list_conversations() == []
    guess = visit("x" * 43)                                    # a made-up link sees nothing either
    assert guess.session_state.services.list_conversations() == []

    back = visit(alice_token)                                  # the same link returns to the same conversations
    assert "how many orders are there?" in page_html(back)
    assert "Anyone who has this address can read them" in page_html(back)


def test_banner_when_conversations_cannot_be_saved(llm, monkeypatch, tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setenv("ASKDATA_DATA_DIR", str(blocker / "data"))
    at = ask(start(), "how many orders are there?")
    page = page_html(at)
    assert "Conversations aren't being saved." in page and "Nothing is being saved." in page
    assert 'class="metric"' in page                            # the app still answers


def test_settings_show_where_things_are_saved_and_create_a_workspace(llm):
    at = ask(start(), "how many orders are there?")
    page = page_html(at)
    assert "askdata_meta.db" in page and "treats everyone using it as one person" in page
    next(t for t in at.text_input if t.label == "New workspace name").set_value("Client work")
    button(at, "Create workspace").click().run()
    assert not at.exception
    services = at.session_state.services
    assert services.workspace.name == "Client work" and services.conversation.history == []
    assert "What do you want to know?" in page_html(at)
    assert any(s.label == "Current workspace" for s in at.selectbox)


def test_restoring_sample_data_asks_before_acting(llm):
    at = start()
    services = at.session_state.services
    services.registry.sample.commit_write("UPDATE orders SET status = 'On hold' WHERE order_id = 7")
    button(at, "Restore sample data").click().run()
    assert any("undoes every change" in w.value for w in at.warning)
    conn = sqlite3.connect(services.registry.sample.path)
    assert conn.execute("SELECT status FROM orders WHERE order_id = 7").fetchone()[0] == "On hold"   # not yet
    button(at, "Restore").click().run()
    assert not at.exception
    assert conn.execute("SELECT status FROM orders WHERE order_id = 7").fetchone()[0] != "On hold"
    conn.close()


# --------------------------------------------------------------------------- #
# Chat operations in the interface
# --------------------------------------------------------------------------- #
from fakes import SequenceLLM, reply  # noqa: E402
from ui.chat_view import GENERATED_NOTE, safety_checks, turn_details  # noqa: E402
from ui.components import MAIN_TABS  # noqa: E402

COUNT = reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n", explanation="Total orders.")
BY_STATUS = reply(sql="SELECT status, COUNT(*) AS n FROM orders GROUP BY status", chart="bar", x="status", y="n",
                  explanation="Orders by status.")
BY_CHANNEL = reply(sql="SELECT channel, COUNT(*) AS n FROM orders GROUP BY channel", chart="bar", x="channel", y="n",
                   explanation="Orders by channel.")


def use_sequence(monkeypatch, *replies) -> SequenceLLM:
    fake = SequenceLLM(*replies)
    monkeypatch.setattr("agent.llm.build_client", lambda get_secret: fake)
    return fake


def conversation_buttons(at: AppTest):
    return [b for b in at.button if b.key and b.key.startswith("conv_")]


def text_input(at: AppTest, label: str):
    return next(t for t in at.text_input if t.label == label)


def history(at: AppTest):
    return at.session_state.services.conversation.history


def test_each_answer_shows_its_details_and_safety_checks(llm):
    at = ask(start(), "Which products had the highest return rate last month?")
    page = page_html(at)
    assert 'class="details"' in page and "A database query" in page and "Sample store" in page
    assert "Ran on a connection that cannot change data" in page and "Reads no column marked sensitive" in page
    assert GENERATED_NOTE in page
    assert 'class="conv-title"' in page and "Rankings" in page          # title and category in the header

    result = history(at)[0]
    labels = dict(turn_details(result))
    assert labels["Outcome"] == "Answered" and labels["Intent"] == "query" and labels["Rows"] == "10"
    assert labels["Shown as"] == "a bar chart" and "ms" in labels["Query time"] and labels["Model"] == "fake · keyword"

    ask(at, "who are our best customers?")
    assert dict(turn_details(history(at)[1]))["Answered from"].startswith("The model's reply only")
    assert safety_checks(history(at)[1]) == []


def test_safety_checks_describe_changes_blocks_and_sensitive_reads():
    answer = result("answer", sql="SELECT name FROM customers", statement="SELECT",
                    sensitive_columns=["customers.name"])
    assert any("customers.name" in line and "not saved" in line for line in safety_checks(answer))
    pending = result("needs_confirmation", sql="UPDATE t SET a = 1 WHERE b = 2", statement="UPDATE")
    assert "Nothing changes until you confirm" in safety_checks(pending)
    applied = result("committed", sql="UPDATE t SET a = 1 WHERE b = 2", statement="UPDATE")
    assert any("row count re-checked" in line for line in safety_checks(applied))
    blocked = result("blocked", sql="DROP TABLE t", error="DROP is not allowed.")
    assert safety_checks(blocked) == ["Stopped before running: DROP is not allowed."]


def test_regenerate_and_try_again_buttons(monkeypatch):
    from agent.llm import LLMError
    use_sequence(monkeypatch, BY_STATUS, BY_CHANNEL, LLMError("The provider is overloaded."), COUNT)
    at = ask(start(), "orders by what?")
    assert "Orders by status." in page_html(at)
    button(at, "Regenerate").click().run()
    assert not at.exception
    page = page_html(at)
    assert "Orders by channel." in page and "Orders by status." not in page and len(history(at)) == 1

    ask(at, "and the total?")
    assert "Couldn't answer that." in page_html(at)
    button(at, "Try again").click().run()
    assert not at.exception and "Total orders." in page_html(at)
    assert [r.status for r in history(at)] == ["answer", "answer"]
    assert [b.label for b in at.button].count("Regenerate") == 1          # only on the latest answer


def test_edit_question_replaces_later_turns(monkeypatch):
    use_sequence(monkeypatch, COUNT, BY_STATUS, BY_CHANNEL)
    at = ask(start(), "how many orders?")
    ask(at, "orders by status")
    first = history(at)[0]
    assert "removes the 1 turn after it" in page_html(at)
    next(t for t in at.text_area if t.key == f"edit_text_{first.request_id}").set_value("orders by channel instead")
    next(b for b in at.button if b.key == f"edit_go_{first.request_id}").click().run()
    assert not at.exception
    assert [r.question for r in history(at)] == ["orders by channel instead"]
    page = page_html(at)
    assert "Orders by channel." in page and "how many orders?" not in page
    assert conversation_buttons(at)[0].label == "orders by channel instead"   # the automatic title followed


def test_manage_menu_renames_tags_pins_duplicates_archives_and_deletes(llm):
    at = ask(start(), "how many orders are there?")
    services = at.session_state.services
    record = services.conversation.conversation

    text_input(at, "Title").set_value("Weekly order count")
    next(m for m in at.multiselect if m.label == "Tags").set_value(["weekly", "ops"])
    button(at, "Save details").click().run()
    assert not at.exception
    assert services.conversation.title == "Weekly order count" and "Weekly order count" in page_html(at)
    assert sorted(t.name for t in services.tags_for(record.id)) == ["ops", "weekly"]

    button(at, "Pin").click().run()
    assert services.storage.conversations.get(services.workspace.id, record.id).pinned
    assert conversation_buttons(at)[0].label == "📌 Weekly order count" and button(at, "Unpin") is not None

    button(at, "Duplicate").click().run()
    assert not at.exception and services.conversation.title == "Copy of Weekly order count"
    assert len(services.list_conversations()) == 2
    copy_id = services.conversation.conversation.id

    button(at, "Archive").click().run()
    assert [c.id for c in services.list_conversations()] == [record.id]
    assert services.conversation.history == []                           # the archived one was closed

    services.open_conversation(copy_id)
    at.run()
    button(at, "Unarchive").click().run()
    assert len(services.list_conversations()) == 2

    button(at, "Delete conversation").click().run()
    assert any("permanently deletes" in w.value for w in at.warning) and len(services.list_conversations()) == 2
    button(at, "Yes, delete it").click().run()
    assert not at.exception and [c.id for c in services.list_conversations()] == [record.id]


def test_export_buttons_offer_three_formats(llm):
    at = ask(start(), "how many orders are there?")
    downloads = [d.proto.label for d in at.get("download_button")]
    for label in ("JSON", "Markdown", "Printable HTML"):
        assert label in downloads
    services = at.session_state.services
    events = list(services.audit.read()["event"])
    assert "export_conversation" not in events                           # preparing the files isn't an export


def test_search_and_filters_narrow_the_conversation_list(llm):
    at = start()
    services = at.session_state.services
    for question in ("how many orders are there?", "Which products had the highest return rate last month?"):
        services.new_conversation()
        services.conversation.ask(question)
    at.run()
    assert len(conversation_buttons(at)) == 2 and "**Recent**" in [m.value for m in at.markdown]

    text_input(at, "Search conversations").set_value("return rate").run()
    assert [b.label for b in conversation_buttons(at)] == ["Which products had the highest return rate last month?"]
    assert "**Results**" in [m.value for m in at.markdown]
    text_input(at, "Search conversations").set_value("zebra").run()
    assert conversation_buttons(at) == [] and "No conversations match." in page_html(at)
    text_input(at, "Search conversations").set_value("").run()

    category_filter = lambda: next(s for s in at.selectbox if s.key == "filter_category")   # noqa: E731
    category_filter().set_value("Totals and counts").run()
    assert [b.label for b in conversation_buttons(at)] == ["how many orders are there?"]
    category_filter().set_value("Any").run()

    services.archive_conversation(services.list_conversations()[0].id)
    at.run()
    assert len(conversation_buttons(at)) == 1
    toggle(at, "Show archived conversations").set_value(True).run()
    assert len(conversation_buttons(at)) == 1 and "**Archived**" in [m.value for m in at.markdown]


def test_saving_a_chart_puts_it_in_the_saved_tab(llm):
    at = start()
    assert "Nothing saved yet" in page_html(at)
    ask(at, "Which products had the highest return rate last month?")
    turn = history(at)[0]
    next(r for r in at.radio if r.key == f"save_kind_{turn.request_id}").set_value("Chart")
    next(t for t in at.text_input if t.key == f"save_name_{turn.request_id}").set_value("Return rate chart")
    next(b for b in at.button if b.key == f"save_go_{turn.request_id}").click().run()
    assert not at.exception
    services = at.session_state.services
    assert [(i.kind, i.title) for i in services.saved_items()] == [("chart", "Return rate chart")]
    page = page_html(at)
    assert "Return rate chart" in page and "Nothing saved yet" not in page

    services.new_conversation()
    at.run()
    button(at, "Open conversation").click().run()                        # jumps back to the conversation it came from
    assert not at.exception and history(at)[0].request_id == turn.request_id
    button(at, "Remove").click().run()
    assert services.saved_items() == [] and "Nothing saved yet" in page_html(at)


def test_related_conversations_are_offered_under_the_chat(llm):
    at = ask(start(), "Which products had the highest return rate last month?")
    services = at.session_state.services
    first_id = services.conversation.conversation.id
    button(at, "New conversation").click().run()
    ask(at, "What is the return rate by category?")
    assert "Related conversations" in page_html(at)
    next(b for b in at.button if b.key == f"related_{first_id}").click().run()
    assert not at.exception and services.conversation.conversation.id == first_id


# --------------------------------------------------------------------------- #
# Memory in the interface
# --------------------------------------------------------------------------- #
def memory_of(at: AppTest):
    return at.session_state.services.memory


def text_area(at: AppTest, label: str):
    return next(t for t in at.text_area if t.label == label)


def test_memory_tab_adds_edits_and_deletes(llm):
    at = start()
    assert "Nothing yet." in page_html(at) and toggle(at, "Use long-term memory").value is True
    text_area(at, "What should AskData remember?").set_value("Revenue always excludes cancelled orders")
    button(at, "Remember this").click().run()
    assert not at.exception
    item = memory_of(at).list()[0]
    page = page_html(at)
    assert "Revenue always excludes cancelled orders" in page and "From a memory you added on" in page
    assert "applies to Sample store" in page and "Nothing yet." not in page

    next(t for t in at.text_area if t.key == f"memory_text_{item.id}").set_value("Revenue excludes cancelled and test orders")
    next(x for x in at.checkbox if x.key == f"memory_pin_{item.id}").set_value(True)
    next(b for b in at.button if b.key == f"memory_save_{item.id}").click().run()
    assert not at.exception
    edited = memory_of(at).get(item.id)
    assert edited.content == "Revenue excludes cancelled and test orders" and edited.pinned
    assert "📌 Revenue excludes cancelled and test orders" in page_html(at)

    next(b for b in at.button if b.key == f"ask_memory_delete_{item.id}").click().run()
    assert any("Delete this memory?" in w.value for w in at.warning) and memory_of(at).get(item.id) is not None
    button(at, "Delete it").click().run()
    assert not at.exception and memory_of(at).list() == [] and "Nothing yet." in page_html(at)


def test_remember_in_chat_then_see_it_used_in_the_next_answer(llm):
    at = ask(start(), "Remember that revenue always excludes cancelled orders")
    page = page_html(at)
    assert "Remembered." in page and "Saved to memory" in page and "revenue always excludes cancelled orders" in page
    assert len(llm.calls) == 0 and "Regenerate" not in [b.label for b in at.button]

    ask(at, "how many orders are there?")
    page = page_html(at)
    assert "A database query and remembered context" in page
    assert "Based on a previous conversation from" in page and "revenue always excludes cancelled orders" in page
    assert "What the user asked you to remember" in llm.calls[-1][0]


def test_suggestion_is_offered_under_the_answer_and_needs_a_click(llm):
    at = ask(start(), "how many orders are there? from now on always leave out cancelled ones")
    assert "Remember this for next time?" in page_html(at)
    assert memory_of(at).list() == [] and len(memory_of(at).list(status="suggested")) == 1
    button(at, "Remember").click().run()
    assert not at.exception and len(memory_of(at).list()) == 1
    assert "Remember this for next time?" not in page_html(at)

    ask(at, "how many orders are there? in future never show test accounts")
    button(at, "No thanks").click().run()
    assert len(memory_of(at).list()) == 1 and len(memory_of(at).list(status="rejected")) == 1
    assert "Remember this for next time?" not in page_html(at)


def test_suggestions_can_be_approved_or_rejected_in_the_memory_tab(llm):
    at = start()
    memory_of(at).suggest_from_question("From now on show money in whole dollars")
    memory_of(at).suggest_from_question("When I say whales I mean customers with more than 20 orders")
    at.run()
    assert "Nothing here is used until you approve it." in page_html(at)
    first, second = memory_of(at).list(status="suggested")
    next(b for b in at.button if b.key == f"memory_approve_{first.id}").click().run()
    next(b for b in at.button if b.key == f"memory_reject_{second.id}").click().run()
    assert not at.exception
    assert [m.id for m in memory_of(at).list()] == [first.id] and memory_of(at).list(status="suggested") == []


def test_turning_memory_off_and_clearing_it(llm):
    at = start()
    memory_of(at).add("Always exclude cancelled orders")
    memory_of(at).add("The fiscal year starts in July", "note")
    at.run()
    toggle(at, "Use long-term memory").set_value(False).run()
    assert memory_of(at).enabled is False and "Long-term memory is off for this workspace." in page_html(at)
    ask(at, "how many orders are there?")
    assert "What the user asked you to remember" not in llm.calls[-1][0]
    assert "Always exclude cancelled orders" in page_html(at)             # still listed, just not used

    button(at, "Forget everything in this workspace").click().run()
    assert any("permanently deletes all 2 memories" in w.value for w in at.warning) and len(memory_of(at).list()) == 2
    button(at, "Forget everything").click().run()
    assert not at.exception and memory_of(at).list(status=None) == []
    assert any("Forgot 2 memories." in s.value for s in at.success)


def test_forgetting_one_conversations_memory(llm):
    at = ask(start(), "Remember that the fiscal year starts in July")
    memory_of(at).add("Always round money to whole dollars")             # not from this conversation
    at.run()
    button(at, "Forget this conversation's memory").click().run()
    assert any("questions and answers stay" in w.value for w in at.warning)
    button(at, "Forget it").click().run()
    assert not at.exception
    assert [m.content for m in memory_of(at).list()] == ["Always round money to whole dollars"]
    assert len(history(at)) == 1


def test_download_of_all_conversations_is_prepared_on_request(llm):
    at = ask(start(), "how many orders are there?")
    assert "Download all conversations (JSON)" not in [d.proto.label for d in at.get("download_button")]
    button(at, "Prepare a download of all conversations").click().run()
    assert "Download all conversations (JSON)" in [d.proto.label for d in at.get("download_button")]


# --------------------------------------------------------------------------- #
# Visitors' own API keys
# --------------------------------------------------------------------------- #
VISITOR_KEY = "gsk_visitorSecretKey0123456789abcdef"


@pytest.fixture
def key_recorder(monkeypatch):
    """build_client stand-in: no key in the app's settings fails as for real; a visitor's key gets a fake model."""
    seen = []

    def fake_build(get_secret):
        from agent.llm import LLMError
        key = get_secret("GROQ_API_KEY")
        if not key:
            raise LLMError("No API key found.")
        seen.append({"provider": get_secret("LLM_PROVIDER"), "key": key, "model": get_secret("GROQ_MODEL")})
        fake = KeywordLLM(STORE_ANSWERS)
        fake.model = "Groq · visitor"
        return fake
    monkeypatch.setattr("agent.llm.build_client", fake_build)
    return seen


def enter_key(at, form_key="ask", provider="groq", key=VISITOR_KEY, model="", url="", name=""):
    next(r for r in at.radio if r.key == f"own_key_provider_{form_key}").set_value(provider).run()
    if provider == "custom":
        next(t for t in at.text_input if t.key == f"own_key_url_{form_key}").set_value(url)
        next(t for t in at.text_input if t.key == f"own_key_name_{form_key}").set_value(name)
    next(t for t in at.text_input if t.key == f"own_key_value_{form_key}").set_value(key)
    next(t for t in at.text_input if t.key == f"own_key_model_{form_key}").set_value(model)
    next(b for b in at.button if b.label == "Use this key").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_visitor_key_enables_the_app_and_is_never_stored(key_recorder, tmp_path):
    at = enter_key(start())
    assert key_recorder == [{"provider": "groq", "key": VISITOR_KEY, "model": None}]
    services = at.session_state.services
    assert services.ready and services.key_source == "yours" and at.chat_input[0].disabled is False
    assert "One step before you start." not in page_html(at)
    ask(at, "how many orders are there?")
    assert 'class="metric"' in page_html(at)

    # Nowhere on disk, in the log, or on the page
    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            assert VISITOR_KEY.encode() not in path.read_bytes(), path
    assert VISITOR_KEY not in (tmp_path / "audit.jsonl").read_text()
    assert VISITOR_KEY not in page_html(at)
    assert all(t.value != VISITOR_KEY for t in at.text_input)          # the form was cleared after submitting

    # A new session (another visitor, or the same one after closing the page) does not have it
    assert start().session_state.services.ready is False


def test_bad_key_is_refused_with_a_message(key_recorder):
    at = enter_key(start(), key="not a key")
    assert any("doesn't look like an API key" in e.value for e in at.error)
    assert key_recorder == [] and at.session_state.services.ready is False


def test_own_key_can_replace_and_then_return_to_the_apps_key(monkeypatch, key_recorder):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_serverKey0123456789abcdefgh")
    at = ask(start(), "how many orders are there?")
    services = at.session_state.services
    assert services.key_source == "server" and "this app's key" in page_html(at)
    enter_key(at, form_key="settings", model="openai/gpt-oss-20b")
    assert services.key_source == "yours" and key_recorder[-1]["model"] == "openai/gpt-oss-20b"
    ask(at, "how many orders are there?")
    assert services.conversation.history[-1].model == "Groq · visitor"   # the open conversation switched over
    button(at, "Stop using my key").click().run()
    assert services.key_source == "server" and services.llm is services.server_llm



def test_provider_is_a_fixed_choice_with_an_enter_your_own_option(key_recorder, monkeypatch):
    at = start()
    choice = next(r for r in at.radio if r.key == "own_key_provider_ask")
    assert choice.options == ["Groq", "Gemini", "OpenRouter", "OpenAI", "Enter your own"]
    assert not any(s.key == "own_key_provider_ask" for s in at.selectbox)     # nothing to type into
    assert not any(t.key == "own_key_url_ask" for t in at.text_input)          # address only for "Enter your own"

    built = []

    def fake_custom(api_key, base_url, model, name="", **kwargs):
        from agent.llm import check_public_https_url
        url = check_public_https_url(base_url)
        built.append((api_key, url, model, name))
        fake = KeywordLLM(STORE_ANSWERS)
        fake.model = f"{name or 'custom'} · {model}"
        return fake
    monkeypatch.setattr("agent.llm.build_custom_client", fake_custom)
    monkeypatch.setattr("agent.llm.check_public_https_url",
                        lambda url: url.rstrip("/") if url.startswith("https://api.") else
                        (_ for _ in ()).throw(__import__("agent.llm", fromlist=["LLMError"]).LLMError(
                            "That address points to a private or local network, which isn't allowed.")))

    enter_key(at, provider="custom", url="https://169.254.169.254/latest", model="m1", name="Sneaky")
    assert any("private or local network" in e.value for e in at.error) and not at.session_state.services.ready

    enter_key(at, provider="custom", url="https://api.together.xyz/v1/", model="meta-llama/Llama-3.3-70B",
              name="Together")
    assert built == [(VISITOR_KEY, "https://api.together.xyz/v1", "meta-llama/Llama-3.3-70B", "Together")]
    services = at.session_state.services
    assert services.ready and services.key_source == "yours" and services.model_label.startswith("Together")
    ask(at, "how many orders are there?")
    assert 'class="metric"' in page_html(at)


def test_custom_providers_can_be_switched_off(key_recorder, monkeypatch):
    monkeypatch.setenv("ASKDATA_ALLOW_CUSTOM_PROVIDERS", "off")
    at = start()
    assert "Enter your own" not in next(r for r in at.radio if r.key == "own_key_provider_ask").options


def test_open_on_a_saved_item_goes_to_its_conversation_in_the_ask_tab(llm):
    at = ask(start(), "Which products had the highest return rate last month?")
    services = at.session_state.services
    turn = history(at)[0]
    services.save_item(turn.request_id, "chart", "Return rates")
    with pytest.raises(Exception, match="already in the Saved tab"):
        services.save_item(turn.request_id, "chart", "Return rates again")   # no duplicates
    services.save_item(turn.request_id, "query")                              # a different kind is fine
    assert len(services.saved_items()) == 2

    services.new_conversation()
    at.session_state[MAIN_TABS] = "Saved"
    at.run()
    assert services.conversation.history == []
    assert any(e.label == "Show the saved chart" for e in at.expander)
    assert len(at.get("plotly_chart")) >= 1                                   # the preview draws the chart
    next(b for b in at.button if b.label == "Open conversation").click().run()
    assert not at.exception
    assert at.session_state[MAIN_TABS] == "Ask"
    assert services.conversation.history[0].request_id == turn.request_id
