"""Scripted stand-ins for an LLM, shared by the offline tests."""

from __future__ import annotations

import json

from agent.llm import LLMError, LLMResponse, parse_agent_json


def reply(intent="query", sql="", chart="table", x="", y="", explanation="ok", **extra) -> str:
    """One model response in the agent's JSON format."""
    data = {"reasoning": "r", "intent": intent, "sql": sql, "explanation": explanation, "chart_type": chart,
            "x": x, "y": y, "color": "", "chart_title": "", "follow_ups": ["Next question A", "Next question B"]}
    data.update(extra)
    return json.dumps(data)


class KeywordLLM:
    """Answers by matching a keyword in the latest user message. Records every call."""

    model = "fake · keyword"

    def __init__(self, answers: dict[str, str], default: str | None = None, fail_with: str | None = None):
        self.answers = answers
        self.default = default or reply(intent="unsupported", explanation="That isn't in this data.")
        self.fail_with = fail_with
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def generate(self, system, turns):
        self.calls.append((system, turns))
        if self.fail_with:
            raise LLMError(self.fail_with)
        last = turns[-1][1].lower()
        text = next((v for k, v in self.answers.items() if k in last), self.default)
        return LLMResponse(data=parse_agent_json(text), raw_text=text, model=self.model, attempts=1, latency_ms=5)


STORE_ANSWERS = {
    "return rate": reply(
        sql="SELECT p.name, ROUND(100.0*COUNT(r.return_id)/COUNT(oi.order_item_id),1) AS return_rate_pct "
            "FROM order_items oi JOIN products p ON p.product_id=oi.product_id "
            "LEFT JOIN returns r ON r.order_item_id=oi.order_item_id GROUP BY p.name "
            "ORDER BY return_rate_pct DESC LIMIT 10",
        chart="bar", x="name", y="return_rate_pct", explanation="Products with the highest return rate."),
    "how many orders": reply(sql="SELECT COUNT(*) AS n FROM orders", chart="metric", y="n",
                             explanation="Total number of orders."),
    "best customers": reply(intent="clarify", explanation="By total spend or by number of orders?"),
    # 'On hold' is not a status the sample data ever contains, so tests can tell a real change from none
    "on hold": reply(intent="modify", sql="UPDATE orders SET status = 'On hold' WHERE order_id = 7",
                     explanation="This will change the status of order #7 to On hold."),
    "drop": reply(intent="modify", sql="DROP TABLE orders", explanation="Drop the orders table."),
    # Reads a column the sample store marks as sensitive (customers.name)
    "top customers": reply(
        sql="SELECT c.name, COUNT(*) AS order_count FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
            "GROUP BY c.name ORDER BY order_count DESC LIMIT 5",
        chart="bar", x="name", y="order_count", explanation="The five customers with the most orders."),
    "every order": reply(sql="SELECT order_id, status FROM orders", chart="table", explanation="All orders."),
    "pets": reply(sql="SELECT species, COUNT(*) AS n FROM pets GROUP BY species", chart="bar", x="species", y="n"),
}


class SequenceLLM:
    """Gives its replies in order (the last one repeats). An Exception in the list is raised."""

    model = "fake · sequence"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    def generate(self, system, turns):
        self.calls.append((system, turns))
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, BaseException):
            raise item
        return LLMResponse(data=parse_agent_json(item), raw_text=item, model=self.model, attempts=1, latency_ms=5)


def open_app(tmp_path, llm=None, owner="local", **settings):
    """Build the services the way the app does on start-up, against the data folder under tmp_path."""
    from agent.audit import AuditLog
    from services import build_services
    for name in ("ASKDATA_RATE_PER_MINUTE", "ASKDATA_RATE_PER_DAY", "ASKDATA_RATE_PER_CLIENT_MINUTE",
                 "ASKDATA_RATE_GLOBAL_PER_MINUTE"):
        settings.setdefault(name, 0)       # limits are off unless a test turns them on
    return build_services(lambda name, default=None: settings.get(name, default), owner_id=owner,
                          llm=llm or KeywordLLM(STORE_ANSWERS), data_dir=tmp_path / "data",
                          audit=AuditLog("s", tmp_path / "audit.jsonl"))
