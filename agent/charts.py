"""Turn the model's chart spec + a result DataFrame into a Plotly figure.

The spec comes from the LLM, so it is checked against the real result columns.
If the model named a column that does not exist, or picked a chart that does
not fit the data, we fall back to a sensible chart chosen from the data shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = ["#3B4CCA", "#0E9F8E", "#D98200", "#D64545", "#7C5CD6", "#2E8B57", "#C2418F", "#596273"]   # = ui.theme.CHART_PALETTES["light"] (tested)


# How a result can be shown. "table", "metric" and "none" aren't charts.
VIEWS = ("metric", "table", "bar", "line", "area", "pie", "scatter", "none")
NON_CHART_VIEWS = ("table", "metric", "none")
VIEW_LABELS = {"metric": "Number", "table": "Table", "bar": "Bar", "line": "Line", "area": "Area", "pie": "Pie",
               "scatter": "Scatter", "none": "No chart"}


@dataclass
class ChartSpec:
    type: str
    x: str = ""
    y: str = ""
    color: str = ""
    title: str = ""
    adjusted: str = ""   # note when we had to change the model's choice


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.lower().endswith("_id")]


def _looks_temporal(series: pd.Series) -> bool:
    s = series.astype(str).head(20)
    return bool(len(s)) and s.str.match(r"^\d{4}(-\d{2})?(-\d{2})?$").all()


def _find(df: pd.DataFrame, name: str) -> str:
    """Case-insensitive column lookup; returns '' if absent."""
    if not name:
        return ""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return ""


def resolve_spec(df: pd.DataFrame, data: dict) -> ChartSpec:
    spec = ChartSpec(type=data.get("chart_type") or "table", x=_find(df, data.get("x", "")),
                     y=_find(df, data.get("y", "")), color=_find(df, data.get("color", "")),
                     title=data.get("chart_title", ""))
    nums = _numeric_cols(df)
    others = [c for c in df.columns if c not in nums]

    if df.empty:
        spec.type = "table"
        return spec

    # Single value -> metric
    if df.shape == (1, 1) or (len(df) == 1 and spec.type == "metric"):
        spec.type = "metric"
        spec.y = spec.y or df.columns[-1]
        return spec

    if spec.type in ("table", "metric"):
        if spec.type == "metric":
            spec.type, spec.adjusted = "table", "Showing a table because the result has several rows."
        return spec

    # Fill in or repair axes
    original = (spec.x, spec.y)
    if not spec.y or not pd.api.types.is_numeric_dtype(df[spec.y]):
        spec.y = nums[-1] if nums else ""
    if not spec.x or spec.x == spec.y:
        candidates = others or [c for c in df.columns if c != spec.y]
        spec.x = candidates[0] if candidates else ""
    if not spec.x or not spec.y:
        spec.type, spec.adjusted = "table", "Showing a table because there is nothing numeric to plot."
        return spec
    if (spec.x, spec.y) != original and any(original):
        spec.adjusted = "Adjusted the chart axes to match the result columns."

    if spec.type == "pie" and (len(df) > 8 or (df[spec.y] < 0).any()):
        spec.type, spec.adjusted = "bar", "Switched to a bar chart: too many slices for a pie."
    if spec.type == "scatter" and not pd.api.types.is_numeric_dtype(df[spec.x]):
        # A scatter compares two measures: use another numeric column for x when there is one
        other = next((c for c in nums if c != spec.y), None)
        if other is not None:
            spec.x, spec.color = other, spec.color or (others[0] if others and len(df) <= 12 else "")
            spec.adjusted = "Plotted the two numeric columns against each other."
        else:
            spec.type = "bar"
    if spec.type == "bar" and _looks_temporal(df[spec.x]) and len(df) > 12 and not spec.color:
        spec.type = "line"
    if spec.type == "area" and (df[spec.y] < 0).any():
        spec.type, spec.adjusted = "line", "Switched to a line chart: an area chart can't show negative values."
    if spec.color and spec.color in (spec.x, spec.y):
        spec.color = ""
    return spec


def _is_id_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "id" or lowered.endswith("_id")


def category_labels(df: pd.DataFrame, spec: ChartSpec) -> pd.Series:
    """Text labels for the x column of a bar or pie chart.

    Bars and slices name things; they don't sit on a number line. So every value
    becomes text (a numeric customer_id must not be spaced out like a measurement).
    An id on its own says little, so it is shown as "#37", followed by the first
    short text column of the result when there is one ("#37 · Seattle").
    """
    values = df[spec.x]
    if not _is_id_name(spec.x):
        return values.map(lambda v: "" if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer()
                                                           else str(v)))
    extra = next((c for c in df.columns
                  if c not in (spec.x, spec.y, spec.color) and not pd.api.types.is_numeric_dtype(df[c])
                  and not _looks_temporal(df[c]) and df[c].astype(str).str.len().max() <= 30), None)
    labels = values.map(lambda v: f"#{int(v)}" if isinstance(v, (int, float)) and not pd.isna(v)
                        and float(v).is_integer() else f"#{v}")
    if extra:
        labels = labels + " · " + df[extra].astype(str)
    return labels


def build_figure(df: pd.DataFrame, spec: ChartSpec, palette: list[str] | None = None) -> go.Figure | None:
    """A Plotly figure for `spec`, or None for views that aren't charts (table, metric, none).

    palette: series colours for the viewer's theme (ui.theme.CHART_PALETTES); defaults to the light set.
    """
    if spec.type in NON_CHART_VIEWS or df.empty:
        return None
    kw = dict(title=spec.title or None, color_discrete_sequence=palette or PALETTE)
    color = spec.color or None
    plot_df = df.copy()
    temporal = bool(spec.x) and _looks_temporal(plot_df[spec.x])
    if temporal:
        plot_df[spec.x] = plot_df[spec.x].astype(str)
    if spec.type in ("bar", "pie") and not temporal:
        plot_df[spec.x] = category_labels(plot_df, spec)
    # Keep the query's order (rankings arrive sorted); Plotly would otherwise reorder categories
    order = list(dict.fromkeys(plot_df[spec.x].tolist())) if spec.x else []

    if spec.type == "bar":
        horizontal = not temporal and plot_df[spec.x].astype(str).str.len().max() > 14
        if horizontal:
            fig = px.bar(plot_df, x=spec.y, y=spec.x, color=color, orientation="h", **kw)
            # First row at the top, as in the table
            fig.update_yaxes(type="category", categoryorder="array", categoryarray=list(reversed(order)))
        else:
            fig = px.bar(plot_df, x=spec.x, y=spec.y, color=color, **kw)
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=order)
    elif spec.type == "line":
        if pd.api.types.is_integer_dtype(df[spec.x]) and df[spec.x].nunique() <= 40:
            # Whole numbers such as years or month numbers: one tick per value, no 2024.5
            plot_df = df.copy().sort_values(spec.x)
            plot_df[spec.x] = plot_df[spec.x].astype(str)
            fig = px.line(plot_df, x=spec.x, y=spec.y, color=color, markers=True, **kw)
            fig.update_xaxes(type="category")
        else:
            fig = px.line(plot_df, x=spec.x, y=spec.y, color=color, markers=True, **kw)
    elif spec.type == "area":
        if pd.api.types.is_integer_dtype(df[spec.x]) and df[spec.x].nunique() <= 40:
            plot_df = df.copy().sort_values(spec.x)
            plot_df[spec.x] = plot_df[spec.x].astype(str)
        fig = px.area(plot_df, x=spec.x, y=spec.y, color=color, **kw)
        if pd.api.types.is_integer_dtype(df[spec.x]):
            fig.update_xaxes(type="category")
    elif spec.type == "pie":
        fig = px.pie(plot_df, names=spec.x, values=spec.y, hole=0.45, **kw)
    elif spec.type == "scatter":
        hover = [c for c in plot_df.columns if c not in (spec.x, spec.y)][:3] or None
        fig = px.scatter(plot_df, x=spec.x, y=spec.y, color=color, hover_data=hover, **kw)
    else:
        return None

    fig.update_layout(
        margin=dict(l=10, r=10, t=50 if spec.title else 20, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
        legend_title_text="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=12),
    )
    def label(c: str) -> str:
        return c.replace("_", " ").capitalize()

    horizontal = spec.type == "bar" and fig.data and getattr(fig.data[0], "orientation", None) == "h"
    if horizontal:
        # Category names label themselves; the value axis gets the measure name
        fig.update_yaxes(title_text=None, showgrid=False)
        fig.update_xaxes(title_text=label(spec.y), gridcolor="rgba(128,128,128,0.15)")
    elif spec.type != "pie":
        fig.update_xaxes(title_text=label(spec.x), showgrid=False)
        fig.update_yaxes(title_text=label(spec.y), gridcolor="rgba(128,128,128,0.15)")
    if spec.type == "bar":
        fig.update_traces(marker_line_width=0)
    return fig


# --------------------------------------------------------------------------- #
# Switching views
# --------------------------------------------------------------------------- #
def available_views(df: pd.DataFrame | None) -> list[str]:
    """The views that make sense for this result, in the order they are offered."""
    if df is None or df.empty:
        return ["table", "none"]
    nums = _numeric_cols(df)
    views = []
    if len(df) == 1 and nums:
        views.append("metric")
    views.append("table")
    if nums and df.shape[1] >= 2 and len(df) >= 2:
        views += ["bar", "line", "area"]
        y = nums[-1]
        if len(df) <= 8 and (df[y] >= 0).all():
            views.append("pie")
        if len(nums) >= 2:
            views.append("scatter")
    views.append("none")
    return views


def apply_view(df: pd.DataFrame, current: ChartSpec | None, view: str) -> ChartSpec:
    """Show `df` as `view`, keeping the current axes and title where they still fit.

    A view that can't be drawn for this data falls back to a table, with a note saying so.
    """
    if view not in VIEWS:
        return ChartSpec(type="table", adjusted=f"Unknown view '{view}', showing a table instead.")
    if view in ("table", "none"):
        base = current or ChartSpec(type=view)
        return ChartSpec(type=view, x=base.x, y=base.y, color=base.color, title=base.title)
    base = current or ChartSpec(type=view)
    spec = resolve_spec(df, {"chart_type": view, "x": base.x, "y": base.y, "color": base.color,
                             "chart_title": base.title})
    if spec.type != view and (base.x or base.y):
        # The current axes don't suit this view (a scatter needs two numbers): let it choose its own
        retry = resolve_spec(df, {"chart_type": view, "chart_title": base.title})
        if retry.type == view:
            spec = retry
    if spec.type != view:
        spec.adjusted = (spec.adjusted or f"This result can't be shown as a {VIEW_LABELS[view].lower()} chart.") \
            + (" Showing a table instead." if spec.type == "table" else "")
    return spec
