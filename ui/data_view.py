"""Browsing the active data source, adding CSV files, and editing its business context."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.adapters import DataSourceError
from agent import masking as masking_rules
from data.metadata import MASKING_MODES, Metric, SemanticConfig, all_columns
from services import AppServices

from . import components as c

CHART_TYPES = ["bar", "line", "area", "pie", "scatter", "metric", "table"]
MASKING_LABELS = {
    "partial": "Partly hidden (j•••@example.com)",
    "full": "Fully hidden (••••)",
    "off": "Shown as stored",
}


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def render_upload(services: AppServices) -> None:
    """CSV uploader. Each file becomes a table in the "Uploaded files" data source."""
    flash = st.session_state.pop("_upload_message", None)
    if flash:
        st.success(flash)

    uploaded = st.file_uploader("Add your own CSV", type=["csv"],
                                help="Becomes a table in its own data source, separate from the sample. "
                                     "You can also attach a CSV in the question box.")
    if uploaded is None:
        return
    fingerprint = (uploaded.name, uploaded.size)
    if st.session_state.get("_last_upload") == fingerprint:
        return   # Streamlit re-sends the file on every run; import it once
    try:
        imported = services.import_csv(uploaded, uploaded.name)
    except DataSourceError as error:
        st.error(str(error))
        return
    st.session_state._last_upload = fingerprint
    message = f"Added table '{imported.table}' ({imported.rows:,} rows). Ask a question about it."
    if imported.date_columns:
        message += f" Dates in {', '.join(imported.date_columns)} were standardised to YYYY-MM-DD."
    if imported.truncated:
        message += " The file was longer than the row limit, so only the first rows were loaded."
    st.session_state._upload_message = message
    st.rerun()   # redraw with the new data source selected


# --------------------------------------------------------------------------- #
# Data tab
# --------------------------------------------------------------------------- #
def render(services: AppServices) -> None:
    """The Data tab: about the source, every table with a preview, and the business context."""
    source = services.source
    try:
        schema = source.schema()
    except DataSourceError as error:
        c.notice(str(error), "err", lead="Couldn't open the data.")
        return

    render_upload(services)
    info = source.info
    access = "can be changed (with confirmation)" if info.writable else "read-only"
    st.markdown(f"**{info.name}** has **{len(schema.tables)} tables**. "
                "These are exactly what the AI sees when writing queries.")
    c.small(f"Type: {info.type} · SQL dialect: {info.dialect} · Access: {access} · "
            f"Limits: {info.limits.max_rows:,} rows and {info.limits.query_timeout_s:g}s per query")

    semantics = source.semantics
    sensitive = semantics.sensitive_set()
    for name, table in schema.tables.items():
        with st.expander(f"{name}  ·  {table.row_count:,} rows"):
            c.dataframe(pd.DataFrame([{
                "column": column.name, "type": column.type,
                "key": "primary" if column.pk else (f"→ {column.fk}" if column.fk else ""),
                "values": ("hidden from the AI (sensitive)" if f"{name}.{column.name}".lower() in sensitive
                           else ", ".join(column.distinct_values)),
            } for column in table.columns]))
            c.dataframe(masking_rules.protect_table(source.preview(name, limit=50), name, sensitive,
                                                    semantics.masking), height=260)
    with st.expander("Exact schema text sent to the model"):
        st.code(services.schema_text(), language="sql")

    _render_data_actions(services)
    st.divider()
    _render_business_context(services)


def _confirmed(key: str, button: str, warning: str, confirm: str) -> bool:
    return c.confirm_button(key, button, warning, confirm)


def _render_data_actions(services: AppServices) -> None:
    info = services.source.info
    if info.is_sample:
        if _confirmed(f"restore_{info.id}", "Restore sample data",
                      "This undoes every change made to the sample data and its business context. "
                      "Your conversations are kept.", "Restore"):
            services.restore_sample()
            st.rerun()
    elif info.type == "csv":
        tables, conversations = services.uploads_summary()
        if _confirmed(f"remove_{info.id}", "Remove uploaded files",
                      f"This permanently deletes {tables} uploaded table{'s' if tables != 1 else ''} and "
                      f"{conversations} conversation{'s' if conversations != 1 else ''} about them.", "Remove"):
            services.remove_uploads()
            st.session_state.pop("_last_upload", None)
            st.rerun()


# --------------------------------------------------------------------------- #
# Business context editor
# --------------------------------------------------------------------------- #
def _pairs(mapping: dict[str, str], first: str, second: str) -> pd.DataFrame:
    return pd.DataFrame([{first: k, second: v} for k, v in mapping.items()], columns=[first, second])


def _to_mapping(frame: pd.DataFrame, first: str, second: str) -> dict[str, str]:
    out = {}
    for row in frame.to_dict("records"):
        key, value = str(row.get(first) or "").strip(), str(row.get(second) or "").strip()
        if key and value and key != "nan" and value != "nan":
            out[key] = value
    return out


def _lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _editor(frame: pd.DataFrame, key: str, **column_config) -> pd.DataFrame:
    try:
        return st.data_editor(frame, num_rows="dynamic", hide_index=True, width="stretch", key=key,
                              column_config=column_config or None)
    except TypeError:  # older Streamlit
        return st.data_editor(frame, num_rows="dynamic", hide_index=True, use_container_width=True, key=key,
                              column_config=column_config or None)


def _render_business_context(services: AppServices) -> None:
    source = services.source
    config = source.semantics
    columns = sorted(all_columns(source.schema()))
    tables = sorted(source.schema().table_names)
    key = source.info.id   # separate widget state per data source

    st.markdown("**Business context**")
    c.small("Tell the AI what this data means: your terms, how your metrics are calculated, and which "
            "columns are private. It is sent with every question about this data source.")

    for warning in st.session_state.pop("_context_warnings", []):
        st.warning(warning)
    if st.session_state.pop("_context_saved", False):
        st.success("Business context saved.")

    with st.form(f"context_{key}"):
        description = st.text_area("What this data is", value=config.description, height=70,
                                   help="One or two sentences. Shown to the AI and on the empty chat screen.")

        st.markdown("Glossary")
        glossary = _editor(_pairs(config.glossary, "term", "meaning"), f"glossary_{key}")

        st.markdown("Metrics")
        metrics = _editor(pd.DataFrame([{"name": m.name, "definition": m.definition, "notes": m.description}
                                        for m in config.metrics], columns=["name", "definition", "notes"]),
                          f"metrics_{key}")

        st.markdown("Table and column descriptions")
        descriptions = _editor(
            _pairs({**config.table_descriptions, **config.column_descriptions}, "table or table.column", "description"),
            f"descriptions_{key}")

        st.markdown("Words users say")
        synonyms = _editor(_pairs(config.synonyms, "word", "means"), f"synonyms_{key}")

        left, right = st.columns(2)
        sensitive = left.multiselect(
            "Sensitive columns", columns, default=[s for s in config.sensitive_columns if s.lower() in columns],
            help="Their values are never sent to the AI, it is told to avoid selecting them, answers that use "
                 "them aren't saved, and their values are hidden in results as set below.")
        date_fields = right.multiselect(
            "Date fields", columns, default=[d for d in config.date_fields if d.lower() in columns])
        masking_mode = st.radio(
            "Sensitive values in results", MASKING_MODES, horizontal=True,
            index=MASKING_MODES.index(config.masking) if config.masking in MASKING_MODES else 0,
            format_func=MASKING_LABELS.get, key=f"masking_{key}",
            help="Applies to tables, charts, downloads and the data preview above. Anything computed from a "
                 "sensitive column is hidden too; counts are not.")

        left, right = st.columns(2)
        default_filters = left.text_area("Default filters (one per line)", "\n".join(config.default_filters), height=90,
                                         help="Applied unless the question says otherwise, e.g. status != 'Cancelled'")
        ownership = right.text_area("Data ownership rules (one per line)", "\n".join(config.ownership_rules), height=90,
                                    help="Always applied, e.g. Only include rows where region = 'West'")
        left, right = st.columns(2)
        joins = left.text_area("Join relationships (one per line)", "\n".join(config.joins), height=90)
        questions = right.text_area("Starter questions (one per line)", "\n".join(config.example_questions), height=90)

        st.markdown("Preferred charts")
        charts = _editor(_pairs(config.preferred_charts, "kind of question", "chart"), f"charts_{key}",
                         chart=st.column_config.SelectboxColumn("chart", options=CHART_TYPES))

        c.small("Default filters and ownership rules guide the AI's queries. They are not access controls: "
                "the database itself does not enforce them.")
        saved = st.form_submit_button("Save business context", type="primary")

    if saved:
        described = _to_mapping(descriptions, "table or table.column", "description")
        new_config = SemanticConfig.from_dict({
            "description": description,
            "glossary": _to_mapping(glossary, "term", "meaning"),
            "metrics": [{"name": m.name, "definition": m.definition, "description": m.description}
                        for m in _metrics_from(metrics)],
            "table_descriptions": {k: v for k, v in described.items() if k.lower() in tables},
            "column_descriptions": {k: v for k, v in described.items() if k.lower() not in tables},
            "synonyms": _to_mapping(synonyms, "word", "means"),
            "sensitive_columns": sensitive, "date_fields": date_fields, "masking": masking_mode,
            "default_filters": _lines(default_filters), "ownership_rules": _lines(ownership),
            "joins": _lines(joins), "example_questions": _lines(questions),
            "preferred_charts": _to_mapping(charts, "kind of question", "chart"),
            "example_changes": config.example_changes,
            "few_shot": [{"question": f.question, "answer": f.answer} for f in config.few_shot],
        })
        st.session_state._context_warnings = services.update_semantics(new_config)
        st.session_state._context_saved = True
        st.rerun()

    left, right = st.columns(2)
    left.download_button("Download as JSON", config.to_json().encode(), mime="application/json",
                         file_name=f"{source.info.name.lower().replace(' ', '_')}_context.json")
    with right.popover("Load from JSON"):
        loaded = st.file_uploader("Business context file", type=["json"], key=f"context_file_{key}")
        if loaded is not None and st.button("Replace the current business context", key=f"context_load_{key}"):
            try:
                new_config = SemanticConfig.from_json(loaded.getvalue().decode("utf-8", errors="replace"))
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state._context_warnings = services.update_semantics(new_config)
                st.session_state._context_saved = True
                st.rerun()


def _metrics_from(frame: pd.DataFrame) -> list[Metric]:
    out = []
    for row in frame.to_dict("records"):
        name, definition = str(row.get("name") or "").strip(), str(row.get("definition") or "").strip()
        notes = str(row.get("notes") or "").strip()
        if name and definition and "nan" not in (name, definition):
            out.append(Metric(name, definition, "" if notes == "nan" else notes))
    return out
