"""Charts built from the shapes real models actually return."""

from __future__ import annotations

import pandas as pd

from agent.charts import ChartSpec, build_figure, category_labels, resolve_spec


def spec_for(df, **data):
    return resolve_spec(df, {"chart_type": data.pop("chart_type", "bar"), **data})


def test_numeric_ids_on_a_bar_chart_are_labels_not_a_number_line():
    """The bug from a live run: customer ids were spaced out on a 0-400 scale."""
    df = pd.DataFrame({"customer_id": [131, 37, 220, 84, 299], "city": ["Austin", "Denver", "Miami", "Reno", "Boise"],
                       "total_spend": [4001.0, 3975.5, 3960.0, 3240.0, 3100.0]})
    fig = build_figure(df, spec_for(df, x="customer_id", y="total_spend"))
    assert fig.layout.xaxis.type == "category"
    assert list(fig.data[0].x) == ["#131 · Austin", "#37 · Denver", "#220 · Miami", "#84 · Reno", "#299 · Boise"]
    assert list(fig.layout.xaxis.categoryarray) == list(fig.data[0].x)       # the query's ranking order is kept


def test_id_labels_without_a_text_column():
    df = pd.DataFrame({"product_id": [7, 3], "units": [50, 40]})
    assert list(category_labels(df, ChartSpec("bar", "product_id", "units"))) == ["#7", "#3"]


def test_year_bars_and_lines_get_one_tick_per_year():
    df = pd.DataFrame({"year": [2024, 2025, 2026], "revenue": [10.0, 12.5, 14.0]})
    bars = build_figure(df, spec_for(df, x="year", y="revenue"))
    assert bars.layout.xaxis.type == "category" and list(bars.data[0].x) == ["2024", "2025", "2026"]
    line = build_figure(df, spec_for(df, chart_type="line", x="year", y="revenue"))
    assert line.layout.xaxis.type == "category" and list(line.data[0].x) == ["2024", "2025", "2026"]


def test_long_names_go_horizontal_with_the_first_row_on_top():
    df = pd.DataFrame({"name": ["Trailblazer Running Shoes", "Summit Hiking Boots", "Flex Yoga Leggings"],
                       "return_rate_pct": [22.1, 18.0, 15.5]})
    fig = build_figure(df, spec_for(df, x="name", y="return_rate_pct"))
    assert fig.data[0].orientation == "h"
    assert list(fig.layout.yaxis.categoryarray)[-1] == "Trailblazer Running Shoes"   # top of the chart


def test_months_stay_on_a_time_axis_and_categories_keep_their_text():
    months = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "n": [3, 5, 4]})
    fig = build_figure(months, spec_for(months, chart_type="line", x="month", y="n"))
    assert list(fig.data[0].x) == ["2026-01", "2026-02", "2026-03"]
    status = pd.DataFrame({"status": ["Delivered", "Shipped"], "n": [200, 30]})
    fig = build_figure(status, spec_for(status, x="status", y="n"))
    assert list(fig.data[0].x) == ["Delivered", "Shipped"] and fig.layout.xaxis.type == "category"


def test_pie_slices_from_numeric_codes_are_labels():
    df = pd.DataFrame({"region_id": [1, 2, 3], "orders": [10, 20, 5]})
    fig = build_figure(df, spec_for(df, chart_type="pie", x="region_id", y="orders"))
    assert list(fig.data[0].labels) == ["#1", "#2", "#3"]
