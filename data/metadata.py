"""Semantic configuration: what a data source *means*, as opposed to how it is structured.

The schema says there is a column `orders.status`. The semantic configuration
says "revenue excludes cancelled orders". Each data source carries its own, so
nothing about any particular industry is built into the agent.

A configuration can be written by hand, loaded from JSON, or suggested
automatically from the schema (`suggest_config`) and then edited.

Two honest limits, stated here because they matter for security:
- Default filters and ownership rules are *guidance to the model*. They are not
  access control; the database enforces nothing on their behalf.
- Sensitive columns are kept out of the prompt (their values are never sent to
  the LLM). Masking them in query results is a separate, later safeguard.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.schema import SchemaInfo

MAX_TEXT = 500          # longest single entry accepted, so a config can't flood the prompt
MAX_ITEMS = 100

# Column-name patterns that usually hold personal or secret data
_SENSITIVE_PATTERNS = (
    r"e_?mail", r"phone", r"mobile", r"\bssn\b", r"social_?sec", r"passport", r"\bdob\b", r"birth",
    r"address", r"street", r"password", r"passwd", r"secret", r"token", r"api_?key", r"card_?num",
    r"credit_?card", r"\biban\b", r"national_?id", r"licen[cs]e", r"ip_?addr", r"first_?name",
    r"last_?name", r"full_?name", r"surname",
)
_PERSON_TABLES = ("customer", "employee", "patient", "user", "student", "member", "client", "contact",
                  "staff", "person", "people", "applicant", "teacher", "doctor")
_DATE_NAME = re.compile(r"(date|_at$|_on$|time|timestamp|^day$|^month$|^year$)", re.I)
_DATE_VALUE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T]\d{2}:\d{2}(:\d{2})?)?")
_NUMERIC_TYPES = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")


@dataclass
class Metric:
    name: str
    definition: str            # SQL expression or precise wording, e.g. "SUM(quantity * unit_price)"
    description: str = ""


@dataclass
class FewShot:
    """A worked example specific to this data source: a question and the ideal JSON answer."""
    question: str
    answer: dict[str, Any]


# How values of sensitive columns appear in results: partly hidden, fully hidden, or shown as stored
MASKING_MODES = ("partial", "full", "off")
DEFAULT_MASKING = "partial"


@dataclass
class SemanticConfig:
    description: str = ""                                              # one or two sentences about the data
    glossary: dict[str, str] = field(default_factory=dict)             # business term -> meaning
    table_descriptions: dict[str, str] = field(default_factory=dict)   # table -> meaning
    column_descriptions: dict[str, str] = field(default_factory=dict)  # "table.column" -> meaning
    synonyms: dict[str, str] = field(default_factory=dict)             # word users say -> table or table.column
    metrics: list[Metric] = field(default_factory=list)
    date_fields: list[str] = field(default_factory=list)               # "table.column"
    sensitive_columns: list[str] = field(default_factory=list)         # "table.column"
    masking: str = DEFAULT_MASKING                                     # one of MASKING_MODES
    default_filters: list[str] = field(default_factory=list)           # applied unless the user says otherwise
    preferred_charts: dict[str, str] = field(default_factory=dict)     # kind of question -> chart type
    joins: list[str] = field(default_factory=list)                     # "a.x = b.y"
    ownership_rules: list[str] = field(default_factory=list)           # e.g. "Only include region = 'West'"
    example_questions: list[str] = field(default_factory=list)
    example_changes: list[str] = field(default_factory=list)
    few_shot: list[FewShot] = field(default_factory=list)

    # -- lookups ----------------------------------------------------------- #
    def sensitive_set(self) -> set[str]:
        return {c.lower() for c in self.sensitive_columns}

    def is_empty(self) -> bool:
        values = asdict(self)
        values.pop("masking")            # a setting, not content
        return not any(values.values())

    # -- prompt ------------------------------------------------------------ #
    def to_prompt(self) -> str:
        """The business-context section of the system prompt. Empty string when nothing is defined.

        Table and column descriptions are not repeated here: they are written
        next to the columns in the schema section.
        """
        sections: list[str] = []

        def add(title: str, lines: list[str]) -> None:
            if lines:
                sections.append(f"### {title}\n" + "\n".join(f"- {line}" for line in lines))

        if self.description:
            sections.append(self.description.strip())
        add("Glossary", [f"{term}: {meaning}" for term, meaning in self.glossary.items()])
        add("Metrics (use these exact definitions)",
            [f"{m.name} = {m.definition}" + (f"  ({m.description})" if m.description else "") for m in self.metrics])
        add("Words users may use", [f'"{word}" means {target}' for word, target in self.synonyms.items()])
        add("Date fields", list(self.date_fields))
        add("Join relationships", list(self.joins))
        add("Default filters (apply unless the user asks otherwise, and say so in the explanation)",
            list(self.default_filters))
        add("Data ownership rules (always apply)", list(self.ownership_rules))
        add("Preferred charts", [f"{kind}: {chart}" for kind, chart in self.preferred_charts.items()])
        if self.sensitive_columns:
            add("Sensitive columns",
                [", ".join(self.sensitive_columns),
                 "Their values are hidden from you. Do not select them unless the user explicitly asks for "
                 "that column; prefer counts and other aggregates."])
        return "\n\n".join(sections)

    # -- (de)serialisation ------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SemanticConfig":
        """Build from untrusted JSON: unknown keys are ignored, sizes are capped, types are coerced."""
        if not isinstance(raw, dict):
            raise ValueError("A semantic configuration must be a JSON object.")

        def text(value: Any) -> str:
            return str(value).strip()[:MAX_TEXT]

        def str_dict(key: str) -> dict[str, str]:
            value = raw.get(key) or {}
            if not isinstance(value, dict):
                raise ValueError(f"'{key}' must be an object of text values.")
            return {text(k): text(v) for k, v in list(value.items())[:MAX_ITEMS] if text(k) and text(v)}

        def str_list(key: str) -> list[str]:
            value = raw.get(key) or []
            if not isinstance(value, list):
                raise ValueError(f"'{key}' must be a list.")
            return [text(v) for v in value[:MAX_ITEMS] if text(v)]

        metrics = []
        for m in (raw.get("metrics") or [])[:MAX_ITEMS]:
            if isinstance(m, dict) and text(m.get("name", "")) and text(m.get("definition", "")):
                metrics.append(Metric(text(m["name"]), text(m["definition"]), text(m.get("description", ""))))
        few_shot = []
        for f in (raw.get("few_shot") or [])[:10]:
            if isinstance(f, dict) and text(f.get("question", "")) and isinstance(f.get("answer"), dict):
                few_shot.append(FewShot(text(f["question"]), f["answer"]))

        return cls(
            description=text(raw.get("description", "")),
            glossary=str_dict("glossary"), table_descriptions=str_dict("table_descriptions"),
            column_descriptions=str_dict("column_descriptions"), synonyms=str_dict("synonyms"),
            metrics=metrics, date_fields=str_list("date_fields"),
            sensitive_columns=str_list("sensitive_columns"), default_filters=str_list("default_filters"),
            masking=(str(raw.get("masking") or DEFAULT_MASKING).strip().lower()
                     if str(raw.get("masking") or DEFAULT_MASKING).strip().lower() in MASKING_MODES
                     else DEFAULT_MASKING),
            preferred_charts=str_dict("preferred_charts"), joins=str_list("joins"),
            ownership_rules=str_list("ownership_rules"), example_questions=str_list("example_questions"),
            example_changes=str_list("example_changes"), few_shot=few_shot,
        )

    @classmethod
    def from_json(cls, text: str) -> "SemanticConfig":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"That file isn't valid JSON ({e.msg}).") from None

    # -- checking ---------------------------------------------------------- #
    def problems(self, schema: "SchemaInfo") -> list[str]:
        """References to tables or columns that don't exist in the schema."""
        known = all_columns(schema)
        issues = []
        for label, refs in (("Column description", self.column_descriptions), ("Date field", self.date_fields),
                            ("Sensitive column", self.sensitive_columns)):
            for ref in refs:
                if ref.lower() not in known:
                    issues.append(f"{label} '{ref}' doesn't match any table.column in this data.")
        for table in self.table_descriptions:
            if table.lower() not in schema.table_names:
                issues.append(f"Table description '{table}' doesn't match any table in this data.")
        return issues


# --------------------------------------------------------------------------- #
# Suggestions from the schema
# --------------------------------------------------------------------------- #
def all_columns(schema: "SchemaInfo") -> set[str]:
    return {f"{t.name}.{c.name}".lower() for t in schema.tables.values() for c in t.columns}


def _is_numeric(column) -> bool:
    return any(k in (column.type or "").upper() for k in _NUMERIC_TYPES)


def _is_id(column) -> bool:
    name = column.name.lower()
    return column.pk or column.fk is not None or name == "id" or name.endswith("_id")


def looks_sensitive(table: str, column: str) -> bool:
    name = column.lower()
    if any(re.search(p, name) for p in _SENSITIVE_PATTERNS):
        return True
    return name == "name" and any(p in table.lower() for p in _PERSON_TABLES)


def _date_columns(table) -> list[str]:
    found = []
    for i, column in enumerate(table.columns):
        samples = [row[i] for row in table.sample_rows if i < len(row) and row[i] is not None]
        typed = any(k in (column.type or "").upper() for k in ("DATE", "TIME"))
        valued = bool(samples) and all(isinstance(v, str) and _DATE_VALUE.match(v) for v in samples)
        if typed or (valued and (_DATE_NAME.search(column.name) or len(samples) >= 2)):
            found.append(column.name)
    return found


def _label(name: str) -> str:
    return name.replace("_", " ")


def suggest_config(schema: "SchemaInfo", description: str = "") -> SemanticConfig:
    """A starting configuration worked out from the schema alone: date fields, joins,
    likely-sensitive columns and starter questions. Meant to be reviewed and edited."""
    config = SemanticConfig(description=description)
    per_table_questions: list[list[str]] = []
    changes: list[str] = []

    for table in schema.tables.values():
        dates = _date_columns(table)
        config.date_fields += [f"{table.name}.{d}" for d in dates]
        for column in table.columns:
            if column.fk:
                target_table, target_column = column.fk.split(".", 1)
                config.joins.append(f"{table.name}.{column.name} = {target_table}.{target_column}")
            if looks_sensitive(table.name, column.name):
                config.sensitive_columns.append(f"{table.name}.{column.name}")

        sensitive = config.sensitive_set()
        usable = [c for c in table.columns if f"{table.name}.{c.name}".lower() not in sensitive]
        # A column whose values are all different is an identifier or label, not a category
        categories = [c for c in usable if c.distinct_values and len(c.distinct_values) < table.row_count]
        measures = [c for c in usable if _is_numeric(c) and not _is_id(c)]
        noun = _label(table.name)

        questions = [f"How many {noun} are there?"]
        if categories:
            questions.append(f"How many {noun} are there for each {_label(categories[0].name)}?")
        if measures and categories:
            questions.append(f"What is the average {_label(measures[0].name)} by {_label(categories[0].name)}?")
        elif measures:
            questions.append(f"What are the highest and lowest {_label(measures[0].name)} in {noun}?")
        open_dates = [d for d in dates if f"{table.name}.{d}".lower() not in sensitive]
        if open_dates:
            questions.append(f"How many {noun} were there each month, by {_label(open_dates[0])}?")
        per_table_questions.append(questions[1:] + questions[:1])   # the most informative ones first

        key = next((c for c in table.columns if c.pk), None)
        if key and categories and table.sample_rows:
            key_value = table.sample_rows[0][table.columns.index(key)]
            changes.append(f"In {noun}, set {_label(categories[0].name)} to '{categories[0].distinct_values[-1]}' "
                           f"where {_label(key.name)} is {key_value}")

    # Take questions from each table in turn, so every table is represented
    while len(config.example_questions) < 6 and any(per_table_questions):
        for questions in per_table_questions:
            if questions and len(config.example_questions) < 6:
                config.example_questions.append(questions.pop(0))
    config.example_changes = changes[:2]
    return config


def merge_suggestions(current: SemanticConfig, suggested: SemanticConfig) -> SemanticConfig:
    """Add newly suggested structural facts (after a schema change) without overwriting the user's edits."""
    for attr in ("date_fields", "joins", "sensitive_columns", "example_questions", "example_changes"):
        existing = getattr(current, attr)
        lowered = {e.lower() for e in existing}
        existing += [s for s in getattr(suggested, attr) if s.lower() not in lowered]
    current.example_questions = current.example_questions[:8]
    if not current.description:
        current.description = suggested.description
    return current
