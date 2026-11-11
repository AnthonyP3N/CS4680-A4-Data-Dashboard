"""Everything specific to the built-in sample dataset lives here, and nowhere else.

This is also a complete example of a semantic configuration: copy the shape of
`config()` when describing your own data.
"""

from __future__ import annotations

from .metadata import FewShot, Metric, SemanticConfig

NAME = "Sample store"

DESCRIPTION = ("The sample store has products, customers, orders and returns from the start of last "
               "year until today.")

EXAMPLE_QUESTIONS = [
    "Which products had the highest return rate last month?",
    "Show monthly revenue for the last 12 months",
    "What share of orders came from each sales channel this year?",
    "Who are our top 10 customers by total spend?",
    "Why do customers return Footwear? Break down by reason",
    "What was our average order value this month?",
]

EXAMPLE_CHANGES = [
    "Cancel order 2999",
    "Raise the price of Classic Cotton Tee by 10%",
    "Delete returns with reason 'Changed mind' from before this year",
]

# Worked examples. "{today}" is replaced with the current date when the prompt is built.
_FEW_SHOT = [
    FewShot("How many orders did we get this month?", {
        "reasoning": "Count orders whose order_date falls in the current month, excluding cancelled ones.",
        "intent": "query",
        "sql": "SELECT COUNT(*) AS orders_this_month FROM orders "
               "WHERE order_date >= date('{today}', 'start of month') AND status != 'Cancelled';",
        "explanation": "Number of non-cancelled orders placed so far this month.",
        "chart_type": "metric", "x": "", "y": "orders_this_month", "color": "",
        "chart_title": "Orders this month",
        "follow_ups": ["How does that compare to last month?", "Which channel brought the most orders?"],
    }),
    FewShot("Monthly revenue by category for this year", {
        "reasoning": "Join order_items to orders and products, filter to this year and non-cancelled "
                     "orders, group by month and category, sum quantity * unit_price.",
        "intent": "query",
        "sql": "SELECT strftime('%Y-%m', o.order_date) AS month, p.category, "
               "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue\n"
               "FROM order_items oi\nJOIN orders o ON o.order_id = oi.order_id\n"
               "JOIN products p ON p.product_id = oi.product_id\n"
               "WHERE o.status != 'Cancelled' AND o.order_date >= date('{today}', 'start of year')\n"
               "GROUP BY month, p.category\nORDER BY month, p.category;",
        "explanation": "Revenue for each product category, month by month, for the current year.",
        "chart_type": "line", "x": "month", "y": "revenue", "color": "category",
        "chart_title": "Monthly revenue by category",
        "follow_ups": ["Which category grew the fastest?", "Show profit instead of revenue"],
    }),
    FewShot("Who are our best customers?", {
        "reasoning": "'Best' could mean total spend, number of orders, or profit; these give different lists.",
        "intent": "clarify",
        "sql": "",
        "explanation": "Do you mean the customers who spent the most in total, who placed the most orders, "
                       "or who are most profitable? And over all time or a specific period?",
        "chart_type": "table", "x": "", "y": "", "color": "", "chart_title": "",
        "follow_ups": ["Top 10 customers by total spend", "Top 10 customers by number of orders"],
    }),
    FewShot("Mark order 1042 as cancelled", {
        "reasoning": "The user explicitly asks to change one order's status; target it by primary key.",
        "intent": "modify",
        "sql": "UPDATE orders SET status = 'Cancelled' WHERE order_id = 1042;",
        "explanation": "This will change the status of order #1042 to Cancelled.",
        "chart_type": "table", "x": "", "y": "", "color": "", "chart_title": "",
        "follow_ups": ["Show all cancelled orders this month", "What was in order 1042?"],
    }),
    FewShot("What's the weather in Seattle?", {
        "reasoning": "Weather is not in the database.",
        "intent": "unsupported",
        "sql": "",
        "explanation": "This data covers the store's products, customers, orders and returns, so it can't "
                       "tell you the weather. It can answer questions like the ones below.",
        "chart_type": "table", "x": "", "y": "", "color": "", "chart_title": "",
        "follow_ups": ["Revenue from Seattle customers this year", "Top products in Washington state"],
    }),
]


def config() -> SemanticConfig:
    """A fresh copy of the sample store's semantic configuration."""
    return SemanticConfig(
        description=DESCRIPTION,
        glossary={
            "Order date vs return date": "an order's date is orders.order_date; a return's date is "
                                         "returns.return_date. For 'return rate last month', use items from "
                                         "orders placed last month.",
            "Cancelled orders": "orders with status = 'Cancelled' never count towards revenue, profit or AOV.",
        },
        metrics=[
            Metric("Revenue / sales", "SUM(order_items.quantity * order_items.unit_price)",
                   "excluding orders with status = 'Cancelled'"),
            Metric("Profit", "revenue - SUM(order_items.quantity * products.cost)"),
            Metric("Return rate", "number of returned order_items / number of order_items sold",
                   "per product or category, as a percentage rounded to 1 decimal; join returns to "
                   "order_items on order_item_id"),
            Metric("AOV (average order value)", "revenue / number of distinct non-cancelled orders"),
        ],
        table_descriptions={
            "order_items": "one row per product line within an order",
            "returns": "one row per returned order item",
        },
        column_descriptions={
            "order_items.unit_price": "price actually paid per unit, after discounts",
            "products.price": "current list price in USD",
            "products.cost": "unit cost in USD",
            "returns.refund_amount": "USD refunded",
        },
        synonyms={"sales": "revenue", "clients": "customers", "buyers": "customers", "items": "order_items",
                  "refunds": "returns"},
        date_fields=["orders.order_date", "returns.return_date", "customers.signup_date"],
        sensitive_columns=["customers.name", "customers.email"],
        joins=[
            "orders.customer_id = customers.customer_id",
            "order_items.order_id = orders.order_id",
            "order_items.product_id = products.product_id",
            "returns.order_item_id = order_items.order_item_id",
        ],
        preferred_charts={"revenue over time": "line", "share by channel": "pie"},
        example_questions=list(EXAMPLE_QUESTIONS),
        example_changes=list(EXAMPLE_CHANGES),
        few_shot=[FewShot(f.question, dict(f.answer)) for f in _FEW_SHOT],
    )
