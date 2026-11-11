"""Where an answer came from: the query that produced it and any remembered context it used.

A citation is a small record that can be stored with a message and turned into
a sentence for the person reading, such as
"Based on a previous conversation from November 8".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, tzinfo
from typing import Any

from storage.models import MemoryItem


@dataclass
class Citation:
    kind: str                        # "memory" or "query"
    origin: str = ""                 # memory: "conversation", "saved_analysis" or "user"; query: ""
    date: str = ""                   # UTC timestamp of the thing cited
    memory_id: str = ""
    conversation_id: str | None = None
    message_id: str | None = None
    source_name: str = ""            # query: the data source
    tables: tuple[str, ...] = ()     # query: tables read

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tables"] = list(self.tables)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Citation":
        return cls(kind=str(raw.get("kind", "")), origin=str(raw.get("origin", "")), date=str(raw.get("date", "")),
                   memory_id=str(raw.get("memory_id", "")), conversation_id=raw.get("conversation_id"),
                   message_id=raw.get("message_id"), source_name=str(raw.get("source_name", "")),
                   tables=tuple(str(t) for t in raw.get("tables") or ()))


def format_day(timestamp: str, zone: tzinfo | None = None, today: datetime | None = None) -> str:
    """ "November 8", or "November 8, 2025" when it isn't this year. The day is the reader's local day."""
    try:
        moment = datetime.fromisoformat(timestamp).astimezone(zone)
    except (TypeError, ValueError):
        return "an earlier date"
    now = (today or datetime.now(moment.tzinfo)).astimezone(moment.tzinfo)
    text = f"{moment.strftime('%B')} {moment.day}"
    return text if moment.year == now.year else f"{text}, {moment.year}"


def memory_citation(item: MemoryItem) -> Citation:
    if item.memory_type == "saved_analysis":
        origin = "saved_analysis"
    elif item.conversation_id or item.source_message_id:
        origin = "conversation"
    else:
        origin = "user"
    return Citation(kind="memory", origin=origin, date=item.created_at, memory_id=item.id,
                    conversation_id=item.conversation_id, message_id=item.source_message_id)


def query_citation(source_name: str, tables: list[str], date: str = "") -> Citation:
    return Citation(kind="query", source_name=source_name, tables=tuple(tables), date=date)


def describe(citation: Citation, zone: tzinfo | None = None, today: datetime | None = None) -> str:
    """The sentence shown to the reader."""
    if citation.kind == "query":
        tables = ", ".join(citation.tables)
        where = f" ({tables})" if tables else ""
        return f"A query on {citation.source_name or 'the data'}{where}"
    day = format_day(citation.date, zone, today)
    if citation.origin == "saved_analysis":
        return f"Based on an analysis you saved on {day}"
    if citation.origin == "conversation":
        return f"Based on a previous conversation from {day}"
    return f"Based on a memory you added on {day}"
