"""Assembles what is sent to the model for one question.

    system instruction = role + today's date
                       + the active data source's schema (sensitive values hidden)
                       + that data source's business context (glossary, metrics, ...)
                       + what the user asked to be remembered     (long-term memory, if relevant)
                       + a summary of older turns                  (when the conversation is long)
                       + dialect notes + rules
    turns              = worked examples, then the most recent turns, then the new question

The schema, business context and rules are always sent in full. The three kinds
of memory share a token budget (ContextBudget), so a long history or a large
memory store can never crowd the prompt: long-term memories are chosen by
relevance and cut to their share, the summary is clipped to its share, and
recent turns are dropped oldest-first if the rest runs out. Nothing is ever
sent from another workspace or about another data source.

Keeping this in one place means there is exactly one answer to "what did the
model see?", which matters for debugging and for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from . import prompts
from .memory import MemoryService, RetrievedMemory, ShortTermMemory, estimate_tokens

if TYPE_CHECKING:
    from data.adapters import DataSourceAdapter

    from .pipeline import AgentResult

DEFAULT_CONTEXT_TOKENS = 2000


@dataclass
class ContextBudget:
    """How many tokens memory may use in one prompt, and how they are shared out.

    total_tokens covers long-term memories, the conversation summary and the
    recent turns together. It does not include the schema, business context or
    rules, which are always sent.
    """
    total_tokens: int = DEFAULT_CONTEXT_TOKENS
    memory_share: float = 0.30          # at most this fraction for long-term memories
    summary_share: float = 0.25         # at most this fraction for the conversation summary

    @property
    def memory_tokens(self) -> int:
        return int(self.total_tokens * self.memory_share)

    @property
    def summary_tokens(self) -> int:
        return int(self.total_tokens * self.summary_share)


@dataclass
class PromptContext:
    system: str
    turns: list[tuple[str, str]]                      # (role, text) with role "user" or "model"
    remembered: int = 0                               # how many recent turns were included
    memories: list[RetrievedMemory] = field(default_factory=list)   # long-term memories included
    summary_used: bool = False
    tokens: dict[str, int] = field(default_factory=dict)            # estimated tokens per kind of memory


class ContextBuilder:
    def __init__(self, memory: ShortTermMemory | None = None, long_term: MemoryService | None = None,
                 budget: ContextBudget | None = None):
        self.memory = memory or ShortTermMemory()
        self.long_term = long_term
        self.budget = budget or ContextBudget()

    def schema_text(self, source: "DataSourceAdapter") -> str:
        """The schema exactly as the model sees it (also shown in the Data tab)."""
        semantics = source.semantics
        return source.schema().to_prompt(sensitive=semantics.sensitive_set(),
                                         table_notes=semantics.table_descriptions,
                                         column_notes=semantics.column_descriptions)

    def build(self, question: str, history: Sequence["AgentResult"], source: "DataSourceAdapter", *,
              summary: str = "", conversation_id: str | None = None) -> PromptContext:
        info, semantics = source.info, source.semantics
        budget = self.budget

        # Long-term memories: relevant ones for this data source, within their share
        memories: list[RetrievedMemory] = []
        if self.long_term is not None:
            memories = self.long_term.retrieve(question, info.id, conversation_id=conversation_id,
                                               token_budget=budget.memory_tokens)
        memory_text = "\n".join(f"- {m.item.content}" for m in memories)

        # Conversation summary: clipped to its share (it is built newest-last, so the oldest lines go first)
        summary_text = _fit(summary, budget.summary_tokens)

        # Recent turns: about this data source only, newest kept first, within what is left
        relevant = [h for h in history if not h.source_id or h.source_id == info.id]
        remaining = budget.total_tokens - estimate_tokens(memory_text) - estimate_tokens(summary_text)
        recent: list["AgentResult"] = []
        spent = 0
        for turn in reversed(self.memory.select(relevant)):
            cost = estimate_tokens(turn.question) + estimate_tokens(turn.history_json())
            if recent and spent + cost > remaining:
                break                                 # the latest turn is always kept; older ones only if they fit
            recent.append(turn)
            spent += cost
        recent.reverse()

        system = prompts.build_system_instruction(
            self.schema_text(source), source_name=info.name, business_context=semantics.to_prompt(),
            dialect=info.dialect, writable=info.writable, memory_text=memory_text, summary_text=summary_text)

        turns: list[tuple[str, str]] = []
        for q, a in prompts.few_shot_turns(source.schema(), semantics, writable=info.writable):
            turns += [("user", q), ("model", a)]
        for turn in recent:
            turns += [("user", turn.question), ("model", turn.history_json())]
        turns.append(("user", question))
        return PromptContext(
            system=system, turns=turns, remembered=len(recent), memories=memories,
            summary_used=bool(summary_text),
            tokens={"memories": estimate_tokens(memory_text), "summary": estimate_tokens(summary_text),
                    "recent_turns": spent, "budget": budget.total_tokens})


def _fit(text: str, max_tokens: int) -> str:
    """Trim a summary to a token allowance by dropping its oldest lines (the header line is kept)."""
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    header, *lines = text.splitlines()
    while lines and estimate_tokens("\n".join([header] + lines)) > max_tokens:
        lines.pop(0)
    return "\n".join([header] + lines) if lines else ""
