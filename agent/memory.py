"""Memory: what the agent carries from earlier into the next answer.

Three layers, from shortest-lived to longest:

1. Short-term memory   The last few successful turns of the open conversation,
                       sent word for word (ShortTermMemory).
2. Conversation summary  When a conversation grows past the short-term window,
                       the older turns are condensed into a few lines: what was
                       asked, which measures and groupings were used, which
                       filters applied, what changed, and what is still open
                       (summarize_turns).
3. Long-term memory    Things remembered for the whole workspace, across
                       conversations: preferences, definitions, default filters,
                       preferred date ranges and charts, saved analyses
                       (MemoryService).

Rules that hold throughout:

- Long-term memories are made only from the user's own words: "remember that
  ...", the Memory screen, or a suggestion the user approves. Result rows are
  never turned into memory, and text that looks like personal data or a secret
  is never suggested.
- Nothing here calls the model. Summaries and suggestions are rule-based, so
  they are free, immediate, repeatable and work offline.
- Every lookup is scoped to one workspace, and a memory tied to one data source
  is never used for another.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

import sqlglot
from sqlglot import exp

from data.adapters import redact
from storage.models import MemoryItem

from .citations import Citation, memory_citation

if TYPE_CHECKING:
    from storage import Storage

    from .audit import AuditLog
    from .pipeline import AgentResult

# --------------------------------------------------------------------------- #
# 1. Short-term memory
# --------------------------------------------------------------------------- #
# Turns that carry meaning for a follow-up question. Failed, blocked and
# cancelled turns are left out so they can't mislead the next answer.
CONTEXT_STATUSES = ("answer", "clarify", "committed")
DEFAULT_TURNS = 3


class ShortTermMemory:
    """The last few successful turns of the current conversation."""

    def __init__(self, max_turns: int = DEFAULT_TURNS):
        self.max_turns = max_turns

    def usable(self, history: Sequence["AgentResult"]) -> list["AgentResult"]:
        return [h for h in history if h.status in CONTEXT_STATUSES]

    def select(self, history: Sequence["AgentResult"]) -> list["AgentResult"]:
        usable = self.usable(history)
        return usable[-self.max_turns:] if self.max_turns > 0 else []

    def older(self, history: Sequence["AgentResult"]) -> list["AgentResult"]:
        """The usable turns that no longer fit in the short-term window (what a summary must cover)."""
        usable = self.usable(history)
        return usable[:-self.max_turns] if self.max_turns > 0 else usable


# --------------------------------------------------------------------------- #
# Shared text helpers
# --------------------------------------------------------------------------- #
# Words too common to say what a question is about
STOPWORDS = frozenset("""a an and are as at be been by did do does each for from had has have how i in into is it its
last many me much my of on or our per show that the their them there these this those to us was we were what when
where which who why will with you your please can could would should also just than then""".split())


def estimate_tokens(text: str) -> int:
    """A rough token count (about four characters per token), good enough for budgeting."""
    return math.ceil(len(text) / 4) if text else 0


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def keywords(text: str) -> set[str]:
    """The meaningful words of a piece of text, lower-cased and roughly stemmed."""
    return {_stem(w) for w in re.findall(r"[^\W_]+", (text or "").lower()) if w not in STOPWORDS and len(w) > 1}


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _first_sentence(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    match = re.match(r"(.+?[.!?])(\s|$)", flat)
    return _clip(match.group(1) if match else flat, limit)


# --------------------------------------------------------------------------- #
# 2. Conversation summary
# --------------------------------------------------------------------------- #
SUMMARY_CHARS = 1200


def _query_facts(sql: str, dialect: str) -> str:
    """Measures, groupings and filters of a query, in a few words. Empty if it can't be read."""
    try:
        node = sqlglot.parse_one(sql, read=dialect)
    except Exception:  # sqlglot raises several types; a summary must never fail
        return ""
    if not isinstance(node, exp.Select):
        return ""
    facts = []
    measures = [e.alias_or_name for e in node.expressions if e.find(exp.AggFunc) and e.alias_or_name]
    if measures:
        facts.append("measures: " + ", ".join(measures[:4]))
    group = node.args.get("group")
    if group is not None and group.expressions:
        facts.append("grouped by: " + ", ".join(_clip(g.sql(dialect=dialect), 30) for g in group.expressions[:3]))
    where = node.args.get("where")
    if where is not None:
        facts.append("filter: " + _clip(where.this.sql(dialect=dialect), 110))
    return "; ".join(facts)


def summarize_turns(turns: Sequence["AgentResult"], *, dialect: str = "sqlite", max_chars: int = SUMMARY_CHARS) -> str:
    """Condense older turns into a short, factual summary.

    One line per turn: the question, the gist of the answer, and the measures,
    groupings and filters its query used. Applied data changes and questions the
    assistant asked back are called out, since they shape what comes next. When
    the summary would be too long, the oldest turns are dropped and counted.
    """
    lines = []
    for turn in turns:
        question = _clip(turn.question, 90)
        if turn.status == "answer":
            line = f'- Asked "{question}": {_first_sentence(turn.explanation, 120)}'
            facts = _query_facts(turn.sql, dialect) if turn.sql else ""
            if facts:
                line += f" [{facts}]"
        elif turn.status == "committed":
            line = f'- Changed data after "{question}": {_first_sentence(turn.explanation, 120)}'
        elif turn.status == "clarify":
            line = f'- "{question}" was unclear, so the assistant asked: {_first_sentence(turn.explanation, 120)}'
        else:
            continue
        lines.append(line)
    if not lines:
        return ""

    kept: list[str] = []
    used = 0
    for line in reversed(lines):                 # keep the most recent of the older turns
        if kept and used + len(line) + 1 > max_chars:
            break
        kept.append(_clip(line, max_chars))
        used += len(line) + 1
    kept.reverse()
    dropped = len(lines) - len(kept)
    header = f"Summary of {len(lines)} earlier turn{'s' if len(lines) != 1 else ''} in this conversation"
    if dropped:
        header += f" (the oldest {dropped} left out for length)"
    return header + ":\n" + "\n".join(kept)


# --------------------------------------------------------------------------- #
# 3. Long-term memory
# --------------------------------------------------------------------------- #
MEMORY_TYPES = {
    "preference": "Preference",
    "filter": "Default filter",
    "date_range": "Preferred date range",
    "chart": "Preferred chart",
    "definition": "Business definition",
    "metric": "Metric",
    "terminology": "Terminology",
    "saved_analysis": "Saved analysis",
    "note": "Note",
}
# Standing instructions apply to every question about their data source.
# The other types are only brought in when the question is about the same thing.
STANDING_TYPES = frozenset({"preference", "filter", "date_range", "chart"})
MAX_CONTENT = 500
MEMORY_SETTING = "long_term_memory"

# "Remember that ...", "Please note: ...", "Keep in mind that ...". The "that" or colon is required so
# an ordinary question that happens to start with "note" or "remember" is not swallowed.
REMEMBER_COMMAND = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note|keep in mind|don'?t forget)(?:\s+that\b|\s*[:,])\s*(?P<body>.{3,})$",
    re.I | re.S)
_STANDING = re.compile(r"\b(always|by default|from now on|going forward|in future|in the future|never|every time|"
                       r"unless i say otherwise)\b", re.I)
_DEFINITION = re.compile(
    r"(?:\bwhen i say\s+[\"'“]?(?P<term1>[\w][\w \-/]{1,40}?)[\"'”]?,?\s+i mean\s+(?P<meaning1>.{3,}))|"
    r"(?:^[\"'“]?(?P<term2>[\w][\w \-/]{1,40}?)[\"'”]?\s+(?:means|is defined as|refers to)\s+(?P<meaning2>.{3,}))",
    re.I | re.S)
_CHART_WORDS = re.compile(r"\b(chart|charts|graph|graphs|plot|pie|bar|line|table|tables|scatter)\b", re.I)
_PREFERS = re.compile(r"\bi (?:prefer|like|want|would like)\b", re.I)
_FILTER_WORDS = re.compile(r"\b(exclude|excluding|ignore|only include|only show|leave out|filter out|without)\b", re.I)
_DATE_RANGES = ("last 12 months", "last 6 months", "last 3 months", "last 90 days", "last 30 days", "last 7 days",
                "year to date", "this year", "last year", "this quarter", "last quarter", "this month", "last month")
_PERSONAL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\b\d[\d \-]{6,}\d\b")
HABIT_THRESHOLD = 3


def standing_clause(text: str) -> str:
    """The part of a question that is the standing instruction, without the question around it.

    "Which products sell best? From now on always show the top 10" -> "From now on always show the top 10"
    "Show revenue by month, and always exclude cancelled orders"   -> "Always exclude cancelled orders"
    """
    flat = " ".join(text.split())
    for sentence in re.split(r"(?<=[.?!;])\s+", flat):
        trigger = _STANDING.search(sentence) or _PREFERS.search(sentence)
        if not trigger:
            continue
        before = sentence[: trigger.start()]
        cut = max(before.rfind(", and "), before.rfind(", but "), before.rfind(", "), before.rfind("; "))
        if cut >= 0:
            clause = sentence[cut:].lstrip(",; ")
            clause = re.sub(r"^(and|but)\s+", "", clause, flags=re.I)
        else:
            clause = sentence
        clause = clause.strip().rstrip("?.!;")
        return clause[:1].upper() + clause[1:] if clause else flat.rstrip("?")
    return flat.rstrip("?")


def looks_private(text: str) -> bool:
    """True for text containing an email address, a long number, or something shaped like a secret."""
    return bool(_PERSONAL.search(text)) or redact(text) != text


def classify(text: str) -> str:
    """The memory type that best fits a statement."""
    lowered = text.lower()
    if _DEFINITION.search(text):
        return "definition"
    if _CHART_WORDS.search(text) and (_PREFERS.search(text) or _STANDING.search(text)):
        return "chart"
    if any(phrase in lowered for phrase in _DATE_RANGES) and (_STANDING.search(text) or "default" in lowered):
        return "date_range"
    if _FILTER_WORDS.search(text):
        return "filter"
    if _STANDING.search(text) or _PREFERS.search(text):
        return "preference"
    return "note"


@dataclass
class RetrievedMemory:
    item: MemoryItem
    score: float
    citation: Citation

    def to_dict(self) -> dict:
        return {"id": self.item.id, "type": self.item.memory_type, "content": self.item.content,
                "citation": self.citation.to_dict()}


class MemoryActionError(Exception):
    """A memory action couldn't be carried out (bad input, or not found in this workspace)."""


class MemoryService:
    """Long-term memory for one workspace."""

    def __init__(self, storage: "Storage", workspace_id: str, owner_id: str = "local",
                 audit: "AuditLog | None" = None):
        self.storage = storage
        self.workspace_id = workspace_id
        self.owner_id = owner_id
        self.audit = audit

    def _log(self, event: str, **details) -> None:
        if self.audit is not None:
            self.audit.log(event, **details)     # ids and counts only: memory text is never written to the log

    # -- on / off ---------------------------------------------------------- #
    @property
    def enabled(self) -> bool:
        workspace = self.storage.workspaces.get(self.owner_id, self.workspace_id)
        return bool(workspace.settings.get(MEMORY_SETTING, True)) if workspace else False

    def set_enabled(self, enabled: bool) -> None:
        """Turn long-term memory on or off. Existing memories are kept, just not used or added to."""
        self.storage.workspaces.update_settings(self.owner_id, self.workspace_id, **{MEMORY_SETTING: bool(enabled)})
        self._log("memory_enabled" if enabled else "memory_disabled")

    # -- adding ------------------------------------------------------------ #
    @staticmethod
    def _clean(content: str) -> str:
        text = _clip(str(redact(content or "")), MAX_CONTENT)
        if len(text) < 3:
            raise MemoryActionError("There is nothing to remember in that.")
        return text

    def _duplicate(self, content: str) -> MemoryItem | None:
        wanted = " ".join(content.lower().split()).rstrip(".")
        for item in self.storage.memories.list(self.workspace_id):
            if " ".join(item.content.lower().split()).rstrip(".") == wanted:
                return item
        return None

    def add(self, content: str, memory_type: str | None = None, *, data_source_id: str | None = None,
            conversation_id: str | None = None, source_message_id: str | None = None, pinned: bool = False,
            importance: float = 0.7) -> MemoryItem:
        """Remember something the user stated or typed. It is active straight away."""
        text = self._clean(content)
        memory_type = memory_type or classify(text)
        if memory_type not in MEMORY_TYPES:
            raise MemoryActionError(f"Unknown kind of memory '{memory_type}'.")
        existing = self._duplicate(text)
        if existing is not None:                       # saying it again approves it, never duplicates it
            self.storage.memories.update(self.workspace_id, existing.id, status="active", user_approved=True)
            return self.storage.memories.get(self.workspace_id, existing.id)
        try:
            item = self.storage.memories.add(self.workspace_id, MemoryItem(
                workspace_id=self.workspace_id, memory_type=memory_type, content=text,
                data_source_id=data_source_id, conversation_id=conversation_id,
                source_message_id=source_message_id, importance=importance, confidence=1.0,
                user_approved=True, status="active", pinned=pinned))
        except LookupError as error:
            raise MemoryActionError(str(error)) from None
        self._log("memory_added", memory=item.id, memory_type=memory_type)
        return item

    # -- suggestions --------------------------------------------------------- #
    def _suggest(self, content: str, memory_type: str, confidence: float, *, data_source_id: str | None,
                 conversation_id: str | None, source_message_id: str | None) -> MemoryItem | None:
        text = _clip(str(redact(content)), MAX_CONTENT)
        if len(text) < 8 or looks_private(text) or self._duplicate(text) is not None:
            return None                                # includes ones the user already rejected
        try:
            item = self.storage.memories.add(self.workspace_id, MemoryItem(
                workspace_id=self.workspace_id, memory_type=memory_type, content=text,
                data_source_id=data_source_id, conversation_id=conversation_id,
                source_message_id=source_message_id, importance=0.5, confidence=confidence,
                user_approved=False, status="suggested"))
        except LookupError:
            return None
        self._log("memory_suggested", memory=item.id, memory_type=memory_type)
        return item

    def suggest_from_question(self, question: str, *, data_source_id: str | None = None,
                              conversation_id: str | None = None,
                              source_message_id: str | None = None) -> list[MemoryItem]:
        """Offer to remember standing instructions and definitions found in a question.

        Suggestions do nothing until the user approves them. They are made only
        from the question's own words, never from results.
        """
        if not self.enabled or not question or looks_private(question):
            return []
        scope = dict(data_source_id=data_source_id, conversation_id=conversation_id,
                     source_message_id=source_message_id)
        text = " ".join(question.split())
        found: list[MemoryItem | None] = []

        definition = _DEFINITION.search(text)
        if definition:
            term = (definition.group("term1") or definition.group("term2") or "").strip()
            meaning = (definition.group("meaning1") or definition.group("meaning2") or "").strip().rstrip("?.")
            if term and meaning:
                found.append(self._suggest(f'"{term}" means {meaning}', "definition", 0.8, **scope))
        elif _STANDING.search(text) or (_PREFERS.search(text) and _CHART_WORDS.search(text)):
            clause = standing_clause(text)
            found.append(self._suggest(clause, classify(clause), 0.7, **scope))

        if data_source_id:
            found.append(self._suggest_habit(data_source_id, conversation_id))
        return [item for item in found if item is not None]

    def _suggest_habit(self, data_source_id: str, conversation_id: str | None) -> MemoryItem | None:
        """Notice a time range the user keeps asking for, and offer to make it the default."""
        questions = [q.lower() for q in self.storage.memories.recent_questions(self.workspace_id, data_source_id)]
        existing = self.storage.memories.list(self.workspace_id, for_data_source=data_source_id)
        if any(m.memory_type == "date_range" for m in existing):
            return None                                # one default range is enough; don't nag
        for phrase in _DATE_RANGES:
            if sum(1 for q in questions if phrase in q) >= HABIT_THRESHOLD:
                return self._suggest(f"When no time range is given, use {phrase}", "date_range", 0.6,
                                     data_source_id=data_source_id, conversation_id=conversation_id,
                                     source_message_id=None)
        return None

    def approve(self, memory_id: str) -> bool:
        changed = self.storage.memories.update(self.workspace_id, memory_id, status="active", user_approved=True)
        if changed:
            self._log("memory_approved", memory=memory_id)
        return changed

    def reject(self, memory_id: str) -> bool:
        """Decline a suggestion. It is kept, marked rejected, so the same thing isn't suggested again."""
        changed = self.storage.memories.update(self.workspace_id, memory_id, status="rejected", user_approved=False)
        if changed:
            self._log("memory_rejected", memory=memory_id)
        return changed

    # -- viewing and editing ------------------------------------------------- #
    def list(self, status: str | None = "active") -> list[MemoryItem]:
        return self.storage.memories.list(self.workspace_id, status=status)

    def get(self, memory_id: str) -> MemoryItem | None:
        return self.storage.memories.get(self.workspace_id, memory_id)

    def update(self, memory_id: str, *, content: str | None = None, memory_type: str | None = None,
               pinned: bool | None = None, data_source_id: str | None = "", importance: float | None = None) -> bool:
        """Edit a memory. Pass data_source_id=None to make it apply to all data ("" leaves it unchanged)."""
        changes: dict = {}
        if content is not None:
            changes["content"] = self._clean(content)
        if memory_type is not None:
            if memory_type not in MEMORY_TYPES:
                raise MemoryActionError(f"Unknown kind of memory '{memory_type}'.")
            changes["memory_type"] = memory_type
        if pinned is not None:
            changes["pinned"] = pinned
        if importance is not None:
            changes["importance"] = max(0.0, min(1.0, float(importance)))
        if data_source_id != "":
            changes["data_source_id"] = data_source_id
        if not changes:
            return False
        try:
            changed = self.storage.memories.update(self.workspace_id, memory_id, **changes)
        except LookupError as error:
            raise MemoryActionError(str(error)) from None
        if changed:
            self._log("memory_edited", memory=memory_id, fields=sorted(changes))
        return changed

    def delete(self, memory_id: str) -> bool:
        deleted = self.storage.memories.delete(self.workspace_id, memory_id)
        if deleted:
            self._log("memory_deleted", memory=memory_id)
        return deleted

    def clear_conversation(self, conversation_id: str) -> int:
        """Forget everything remembered from one conversation, and its rolling summary."""
        removed = self.storage.memories.delete_for_conversation(self.workspace_id, conversation_id)
        self.storage.conversations.update(self.workspace_id, conversation_id, summary="")
        self._log("memory_cleared_conversation", conversation=conversation_id, removed=removed)
        return removed

    def clear_all(self) -> int:
        """Forget every long-term memory in this workspace, suggestions and rejections included."""
        removed = self.storage.memories.delete_all(self.workspace_id)
        self._log("memory_cleared_all", removed=removed)
        return removed

    # -- retrieval ----------------------------------------------------------- #
    def retrieve(self, question: str, data_source_id: str, *, conversation_id: str | None = None,
                 token_budget: int = 500, limit: int = 8, now: datetime | None = None) -> list[RetrievedMemory]:
        """The memories worth putting in front of the model for this question, best first.

        Considered: active memories that apply to this data source (its own, plus
        workspace-wide ones). Included: pinned memories, standing instructions,
        and others that share meaningful words with the question. Ranked by word
        overlap, the full-text index, where the memory came from, how recent it
        is, its importance and whether it is pinned; then cut to the token budget.
        """
        if not self.enabled or token_budget <= 0:
            return []
        candidates = self.storage.memories.list(self.workspace_id, status="active", for_data_source=data_source_id)
        if not candidates:
            return []
        question_words = keywords(question)
        indexed = self.storage.memories.search(self.workspace_id, sorted(question_words))
        moment = now or datetime.now(timezone.utc)

        scored: list[RetrievedMemory] = []
        for item in candidates:
            overlap = len(question_words & keywords(item.content))
            in_index = item.id in indexed
            standing = item.memory_type in STANDING_TYPES
            if not (item.pinned or standing or overlap or in_index):
                continue                               # nothing connects it to this question
            score = 2.0 * overlap / max(1, min(len(question_words), 4))
            if in_index:
                score += 1.0 / (1.0 + indexed[item.id])
            if standing:
                score += 1.5
            if item.pinned:
                score += 5.0
            if conversation_id and item.conversation_id == conversation_id:
                score += 1.0
            score += item.importance
            try:
                age_days = (moment - datetime.fromisoformat(item.updated_at)).total_seconds() / 86400
                score += max(0.0, 1.0 - age_days / 90)
            except (TypeError, ValueError):
                pass
            scored.append(RetrievedMemory(item=item, score=round(score, 3), citation=memory_citation(item)))

        scored.sort(key=lambda m: (-m.score, m.item.created_at))
        chosen: list[RetrievedMemory] = []
        spent = 0
        for memory in scored[: max(limit, 0)]:
            cost = estimate_tokens(memory.item.content) + 6
            if spent + cost > token_budget:
                continue                               # a shorter, lower-ranked one may still fit
            chosen.append(memory)
            spent += cost
        stamp = moment.isoformat(timespec="milliseconds")
        for memory in chosen:
            self.storage.memories.update(self.workspace_id, memory.item.id, last_used_at=stamp)
        return chosen
