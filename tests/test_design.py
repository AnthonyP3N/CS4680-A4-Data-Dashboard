"""Phases 6 and 7: design tokens, themes, result cards, view switching, explanations and the new layout."""

from __future__ import annotations

import io
import json
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from agent.charts import VIEWS, ChartSpec, apply_view, available_views, build_figure
from agent.explain import describe_query, explain_result
from agent.llm import PROVIDERS
from agent.pipeline import AgentResult
from fakes import STORE_ANSWERS, KeywordLLM, SequenceLLM, open_app, reply
from ui import theme
from ui.context_view import time_ago

APP = str(Path(__file__).resolve().parent.parent / "app.py")
BY_STATUS = reply(sql="SELECT status, COUNT(*) AS order_count FROM orders WHERE channel = 'Web' GROUP BY status "
                      "ORDER BY order_count DESC", chart="bar", x="status", y="order_count",
                  explanation="Web orders by status.")


# --------------------------------------------------------------------------- #
# Tokens, generated files and contrast
# --------------------------------------------------------------------------- #
def test_streamlit_config_is_generated_from_the_tokens():
    on_disk = theme.CONFIG_PATH.read_text()
    assert on_disk == theme.streamlit_config_text(), "run `python -m ui.theme` after changing ui/theme.py"
    config = tomllib.loads(on_disk)
    for name in theme.THEMES:
        section = config["theme"][name]
        palette = theme.PALETTES[name]
        assert section["primaryColor"] == palette["accent"] and section["backgroundColor"] == palette["canvas"]
        assert section["textColor"] == palette["ink"] and section["chartCategoricalColors"] == theme.CHART_PALETTES[name]
    assert config["theme"]["font"].startswith("Figtree:https://fonts.googleapis.com/")
    assert "base" not in config["theme"]                     # follows the viewer's system setting by default


def test_design_tokens_file_for_figma_matches_the_tokens():
    on_disk = json.loads(theme.TOKENS_PATH.read_text())
    assert on_disk == theme.design_tokens(), "run `python -m ui.theme` after changing ui/theme.py"
    assert set(on_disk) == {"core", "light", "dark"}
    assert on_disk["light"]["color"]["highlight"] == {"$type": "color", "$value": "#FFD84D"}
    assert on_disk["core"]["font-size"]["md"]["$value"] == "1rem"
    assert set(on_disk["light"]["color"]) == set(on_disk["dark"]["color"])        # every role exists in both themes


@pytest.mark.parametrize("name", theme.THEMES)
def test_every_colour_pair_meets_wcag_contrast(name):
    palette = theme.PALETTES[name]
    failures = [f"{fg} on {bg}: {theme.contrast(palette[fg], palette[bg]):.2f} < {need}"
                for fg, bg, need in theme.CONTRAST_PAIRS
                if theme.contrast(palette[fg], palette[bg]) < need]
    assert failures == []


def test_contrast_formula_matches_known_values():
    assert theme.contrast("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert theme.contrast("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)
    assert theme.contrast("#777777", "#FFFFFF") == pytest.approx(4.48, abs=0.01)


def test_chart_colours_follow_the_theme():
    df = pd.DataFrame({"status": ["a", "b"], "n": [1, 2]})
    spec = ChartSpec("bar", "status", "n")
    assert build_figure(df, spec, theme.CHART_PALETTES["dark"]).data[0].marker.color == theme.CHART_PALETTES["dark"][0]
    assert build_figure(df, spec).data[0].marker.color == theme.CHART_PALETTES["light"][0]
    from agent import charts
    assert charts.PALETTE == theme.CHART_PALETTES["light"]      # the agent layer's default matches the light theme


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def test_available_views_depend_on_the_result():
    one_number = pd.DataFrame({"n": [42]})
    assert available_views(one_number) == ["metric", "table", "none"]
    categories = pd.DataFrame({"status": ["a", "b", "c"], "n": [3, 2, 1]})
    assert available_views(categories) == ["table", "bar", "line", "area", "pie", "none"]
    two_measures = pd.DataFrame({"price": [1.0, 2.0, 3.0], "units": [3, 2, 1], "name": ["x", "y", "z"]})
    assert "scatter" in available_views(two_measures)
    many = pd.DataFrame({"day": [f"d{i}" for i in range(20)], "n": range(20)})
    assert "pie" not in available_views(many)                 # too many slices
    negative = pd.DataFrame({"account": ["a", "b"], "balance": [-5.0, 3.0]})
    assert "pie" not in available_views(negative)
    text_only = pd.DataFrame({"name": ["a", "b"]})
    assert available_views(text_only) == ["table", "none"]
    assert available_views(None) == ["table", "none"] and set(available_views(two_measures)) <= set(VIEWS)


def test_every_offered_view_draws_or_is_deliberately_not_a_chart():
    df = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "revenue": [10.0, 12.0, 9.0],
                       "orders": [5, 7, 4]})
    for view in available_views(df):
        spec = apply_view(df, ChartSpec("bar", "month", "revenue"), view)
        assert spec.type == view, (view, spec.adjusted)
        figure = build_figure(df, spec)
        assert (figure is None) == (view in ("table", "metric", "none"))
    area = build_figure(df, apply_view(df, None, "area"))
    assert area.data[0].type == "scatter" and (area.data[0].stackgroup or area.data[0].fill)   # filled under the line


def test_an_impossible_view_falls_back_to_a_table_with_a_note():
    text_only = pd.DataFrame({"name": ["a", "b"], "city": ["x", "y"]})
    spec = apply_view(text_only, None, "bar")
    assert spec.type == "table" and "Showing a table instead" in spec.adjusted
    assert apply_view(text_only, None, "hologram").type == "table"
    negative = pd.DataFrame({"m": ["a", "b", "c"], "v": [1.0, -2.0, 3.0]})
    assert apply_view(negative, None, "area").type == "line"


def test_changing_the_view_is_saved_with_the_answer(tmp_path, small_sample):
    app = open_app(tmp_path, SequenceLLM(BY_STATUS))
    turn = app.conversation.ask("web orders by status")
    assert turn.chart.type == "bar"
    app.conversation.set_view(turn.request_id, "pie")
    assert turn.chart.type == "pie" and turn.figure.data[0].type == "pie"
    reloaded = open_app(tmp_path, SequenceLLM(BY_STATUS)).conversation.history[0]
    assert reloaded.chart.type == "pie"
    app.conversation.set_view(turn.request_id, "none")
    assert open_app(tmp_path, SequenceLLM(BY_STATUS)).conversation.history[0].chart.type == "none"
    assert "change_view" in list(app.audit.read()["event"])


# --------------------------------------------------------------------------- #
# Explaining a result
# --------------------------------------------------------------------------- #
def test_query_facts_and_explanation_come_from_the_sql():
    sql = ("SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue, COUNT(*) AS lines FROM order_items oi "
           "JOIN products p ON p.product_id = oi.product_id JOIN orders o ON o.order_id = oi.order_id "
           "WHERE o.status <> 'Cancelled' AND o.channel = 'Web' GROUP BY p.category HAVING SUM(oi.quantity) > 10 "
           "ORDER BY revenue DESC LIMIT 5")
    facts = describe_query(sql)
    assert facts.tables == ["order_items", "products", "orders"] and facts.joins == 2
    assert facts.measures == ["revenue = the total of oi.quantity * oi.unit_price", "lines = the number of rows"]
    assert facts.groups == ["p.category"] and facts.filters == ["o.status <> 'Cancelled'", "o.channel = 'Web'"]
    assert facts.group_filters == ["SUM(oi.quantity) > 10"] and facts.order == ["revenue, highest first"]
    assert facts.limit == 5

    result = AgentResult(request_id="r", question="q", status="answer", sql=sql, row_count=4,
                         memories=[{"content": "Always exclude cancelled orders"}])
    text = " ".join(explain_result(result))
    for expected in ("joining 3 tables", "separately for each p.category", "o.channel = 'Web'",
                     "keeps only groups", "highest first", "at most 5 rows", "had 4 rows",
                     "Always exclude cancelled orders"):
        assert expected in text
    assert explain_result(AgentResult(request_id="r", question="q", status="clarify")) == \
        ["No query was run for this answer."]
    assert describe_query("not sql at all ((") is None and describe_query("DELETE FROM t WHERE a = 1") is None


def test_explanation_reports_business_rules_without_claiming_they_were_applied():
    from data.metadata import SemanticConfig
    result = AgentResult(request_id="r", question="q", status="answer", sql="SELECT COUNT(*) AS n FROM orders")
    text = " ".join(explain_result(result, semantics=SemanticConfig(default_filters=["status <> 'Cancelled'"])))
    assert "were given to the AI" in text and "doesn't filter out any rows" in text


# --------------------------------------------------------------------------- #
# Staged progress
# --------------------------------------------------------------------------- #
def test_progress_reports_each_stage_including_self_correction(tmp_path, small_sample):
    app = open_app(tmp_path, KeywordLLM({
        "your previous response": reply(sql="SELECT name FROM products"),
        "list products": reply(sql="SELECT product_name FROM products"),
        "on hold": STORE_ANSWERS["on hold"],
        "how many": reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n"),
    }))
    stages = []
    app.conversation.ask("list products", progress=stages.append)
    assert stages == ["Reading your question", "Checking the query is safe", "Running the query",
                      "Fixing a problem with the query", "Checking the query is safe", "Running the query"]
    stages.clear()
    app.conversation.ask("put order 7 on hold", allow_writes=True, progress=stages.append)
    assert stages[-1] == "Trying the change without saving it"

    def broken(stage):
        raise RuntimeError("the screen went away")
    assert app.conversation.ask("how many orders?", progress=broken).status == "answer"   # display errors never break it


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider["key"], raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ASKDATA_LOG_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASKDATA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ASKDATA_AUTH", "local")


@pytest.fixture
def llm(monkeypatch) -> KeywordLLM:
    fake = KeywordLLM({**STORE_ANSWERS, "web orders": BY_STATUS})
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


def page(at: AppTest) -> str:
    import html
    return html.unescape("\n".join(m.value for m in at.markdown))


def by_key(widgets, key):
    return next(w for w in widgets if w.key == key)


def test_empty_state_offers_starter_questions(llm):
    at = start()
    starters = [b for b in at.button if b.key and b.key.startswith("starter_")]
    assert len(starters) == 4 and starters[0].label == "Which products had the highest return rate last month?"
    starters[1].click().run()
    assert not at.exception and at.session_state.services.conversation.history[0].question == starters[1].label


def test_result_card_has_summary_meta_views_actions_and_source(llm):
    at = ask(start(), "web orders by status")
    text = page(at)
    turn = at.session_state.services.conversation.history[0]
    assert "Web orders by status." in text and 'class="meta"' in text and "Sample store ·" in text
    assert "Source: A query on Sample store (orders)." in text
    switch = by_key(at.segmented_control, f"view_{turn.request_id}")
    assert switch.value == "bar" and switch.options == ["Table", "Bar", "Line", "Area", "Pie", "No chart"]
    popovers = [p.proto.popover.label for p in at.get("popover")]
    for label in ("Save", "Ask a follow-up", "Explain this result", "Change filters"):
        assert label in popovers
    assert "Download CSV" in [d.proto.label for d in at.get("download_button")]
    assert "It only includes rows where channel = 'Web'." in text            # the explanation is in the card
    assert "Current filters: channel = 'Web'" in text


def test_switching_views_redraws_and_is_remembered(llm):
    at = ask(start(), "web orders by status")
    turn = at.session_state.services.conversation.history[0]
    by_key(at.segmented_control, f"view_{turn.request_id}").set_value("table").run()
    assert not at.exception and turn.chart.type == "table" and len(at.get("plotly_chart")) == 0
    by_key(at.segmented_control, f"view_{turn.request_id}").set_value("pie").run()
    assert len(at.get("plotly_chart")) == 1 and turn.chart.type == "pie"
    by_key(at.segmented_control, f"view_{turn.request_id}").set_value("none").run()
    assert "Chart hidden." in page(at)
    assert start().session_state.services.conversation.history[0].chart.type == "none"   # still so after a reload


def test_follow_up_and_change_filters_ask_in_the_same_conversation(llm):
    at = ask(start(), "web orders by status")
    turn = at.session_state.services.conversation.history[0]
    by_key(at.text_input, f"filters_text_{turn.request_id}").set_value("only the last 6 months")
    by_key(at.button, f"filters_go_{turn.request_id}").click().run()
    assert not at.exception
    history = at.session_state.services.conversation.history
    assert history[-1].question == 'Same as "web orders by status", but only the last 6 months'
    assert ("user", "web orders by status") in llm.calls[-1][1]                   # with the earlier turn as context

    by_key(at.text_input, f"followup_text_{history[-1].request_id}").set_value("how many orders are there?")
    by_key(at.button, f"followup_go_{history[-1].request_id}").click().run()
    assert at.session_state.services.conversation.history[-1].question == "how many orders are there?"


def test_context_panel_shows_source_safety_query_rules_and_memory(llm):
    at = start()
    services = at.session_state.services
    config = services.source.semantics
    config.default_filters = ["orders.status != 'Cancelled'"]
    services.update_semantics(config)
    services.memory.add("Web means the website channel, not the app", "definition")
    ask(at, "web orders by status")
    text = page(at)
    for expected in ("Data source", "Sample store", "5 tables · sqlite · structure read", "Safety",
                     "Read-only. Questions can't change the data.", "up to 1,000 rows and 8 seconds",
                     "Latest query", "Tables: orders", "Filters: channel = 'Web'", "Business rules",
                     "orders.status != 'Cancelled'", "Metrics", "Revenue / sales", "Remembered for this answer",
                     "Web means the website channel", "Based on a memory you added on"):
        assert expected in text, expected
    toggle = next(t for t in at.toggle if t.label == "Allow data changes")
    toggle.set_value(True).run()
    assert "Changes allowed. Each one is previewed" in page(at)
    by_key(at.button, "refresh_structure").click().run()
    assert not at.exception and "refresh_schema" in list(services.audit.read()["event"])


def test_context_panel_can_be_hidden(llm):
    at = ask(start(), "web orders by status")
    next(t for t in at.toggle if t.label == "Show context panel").set_value(False).run()
    assert not at.exception and "Read-only. Questions can't change the data." not in page(at)
    assert "Web orders by status." in page(at)


def test_csv_attached_in_the_question_box_becomes_a_table_and_is_asked_about(llm, monkeypatch):
    class Attached(io.BytesIO):
        name = "pets.csv"

    class Submission:
        text = "how many pets of each species?"
        files = [Attached(b"species,age\ncat,3\ndog,5\ncat,9\n")]

    import ui.chat_view as chat_view
    at = start()
    original = chat_view._read_submission

    def read(submitted):
        text, files = original(submitted)
        return (Submission.text, Submission.files) if text == "ATTACH" else (text, files)
    monkeypatch.setattr(chat_view, "_read_submission", read)
    llm.answers["pets"] = reply(sql="SELECT species, COUNT(*) AS n FROM pets GROUP BY species", chart="bar",
                                x="species", y="n")
    ask(at, "ATTACH")
    services = at.session_state.services
    assert services.source.info.name == "Uploaded files" and "pets" in services.source.schema().table_names
    assert services.conversation.history[0].question == "how many pets of each species?"
    assert services.conversation.history[0].status == "answer"


def test_keyboard_shortcuts_are_wired(llm):
    at = ask(start(), "web orders by status")
    assert by_key(at.button, "regen_" + at.session_state.services.conversation.history[0].request_id).proto.shortcut
    new = next(b for b in at.button if b.label == "New conversation")
    assert new.proto.shortcut
    assert "Start a new conversation" in page(at)


def test_dark_theme_stylesheet_is_used_when_the_viewer_is_in_dark_mode(llm, monkeypatch):
    monkeypatch.setattr(theme, "current_theme", lambda: "dark")
    at = ask(start(), "web orders by status")
    css = next(m.value for m in at.markdown if m.value.startswith("<style>"))
    assert f"--canvas:{theme.PALETTES['dark']['canvas']};" in css
    turn = at.session_state.services.conversation.history[0]
    chart = at.get("plotly_chart")[0]
    assert theme.CHART_PALETTES["dark"][0] in chart.proto.spec


def test_states_have_accessible_roles(llm):
    from agent.llm import LLMError
    llm.fail_with = "The provider is overloaded."
    at = ask(start(), "how many orders?")
    text = page(at)
    assert 'role="alert"' in text and 'aria-label="What AskData did"' in text and 'aria-label="Your question"' in text
    llm.fail_with = None
    ask(at, "how many orders are there?")
    assert 'role="group" aria-label="Total orders.' in page(at) or 'class="metric" role="group"' in page(at)


def test_time_ago_wording():
    now = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    assert time_ago(None) == "not read yet"
    assert time_ago(now - timedelta(seconds=20), now) == "just now"
    assert time_ago(now - timedelta(minutes=1), now) == "1 minute ago"
    assert time_ago(now - timedelta(hours=5), now) == "5 hours ago"
    assert time_ago(now - timedelta(days=3), now) == "3 days ago"


def test_layout_rules_for_narrow_screens_and_the_open_conversation(llm):
    css = theme.stylesheet()
    assert ':has(.st-key-context_panel){ flex-wrap:wrap; }' in css            # panel moves below the chat
    assert ".st-key-conv_open button:disabled" in css and "[data-testid=\"stBaseButton-primary\"]" in css
    at = ask(start(), "web orders by status")
    current = at.session_state.services.conversation.conversation.id
    assert by_key(at.button, f"conv_{current}").disabled                         # marked as open, not clickable
