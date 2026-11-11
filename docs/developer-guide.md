# AskData developer guide

How the main parts work and how to extend them. For the visual side (tokens,
components, themes, adding a chart type) see [design-system.md](design-system.md);
for the safety rules, see [security.md](security.md).

## Layers

```
ui/        Streamlit views. No business logic.
services.py  One AppServices per browser session: wires everything below together.
agent/     The agent: context, model clients, validation, execution, memory, charts, masking, limits.
data/      Data sources behind one interface (DataSourceAdapter) and their business context.
storage/   The metadata database: workspaces, conversations, messages, saved items, memories.
```

Each layer calls only the one below it. Nothing below `ui/` imports Streamlit, so
everything except the views can be tested without a browser.

## What happens when someone asks a question

1. **The view** (`ui/chat_view._render_composer`) passes the text to
   `ConversationService.ask` (`agent/conversation.py`), with a callback that updates
   the progress label.
2. **The conversation service** adds the conversation's recent turns and its rolling
   summary, and calls `AskDataAgent.ask` (`agent/pipeline.py`).
3. **The agent**
   1. rejects empty or over-long questions;
   2. asks the **request gate** whether a question limit has been reached
      (`agent/ratelimit.py`); if so it stops here and nothing reaches the model;
   3. builds the prompt with `ContextBuilder` (next section);
   4. calls the model (`agent/llm.py`), which returns JSON matching `RESPONSE_SCHEMA`;
   5. stops for `clarify` and `unsupported` answers;
   6. validates the SQL (`agent/validator.py`). A policy refusal ends the turn; a
      correctable problem (unknown column, syntax) is sent back to the model with the
      error, up to two more times;
   7. for a change, runs a dry run and waits for confirmation;
   8. for a read, runs it on a read-only connection, masks sensitive values
      (`agent/masking.py`) and picks a chart (`agent/charts.resolve_spec`).
4. **The conversation service** saves the turn (`to_messages`), offers memory
   suggestions from the question's words, and returns the result to the view.

Every step writes to the activity log with the same request id.

## Prompt context assembly

`agent/context.py` builds everything the model sees, in one place, so the answer to
"what did the model see?" is always one function.

```
system instruction
  role and task                                   always
  today's date                                    always
  data source name and schema                     always (sensitive values replaced)
  business context                                always (glossary, metrics, filters, ...)
  remembered preferences and definitions          within the memory share of the budget
  summary of older turns                          within the summary share
  SQL dialect rules and the numbered rules        always
turns
  worked examples                                 always (the source's own, or generated from its schema)
  recent turns of this conversation               what is left of the budget, newest first
  the new question
```

The budget (`ContextBudget`, `ASKDATA_CONTEXT_TOKENS`, default 2,000 tokens at about
four characters a token) covers only the three kinds of memory. Long-term memory may
use 30% of it and the summary 25%; recent turns get what's left. The latest turn is
always kept. The schema, business context and rules are never cut, because a
prompt without them gives wrong SQL.

Only turns about the active data source are included, and only memories from the
current workspace. Result rows are never sent: a previous turn is represented by its
intent, SQL and explanation (`AgentResult.history_json`).

The prompt text itself lives in `agent/prompts.py`, with each prompt-engineering
technique numbered in its docstring.

To change what the model sees, change `ContextBuilder.build` or `prompts.py` and run
`tests/test_memory.py` and `tests/test_services.py`; several tests read the exact
prompt a fake model received.

## Memory retrieval

Long-term memories (`agent/memory.MemoryService`) come only from the user's own
words: "remember that ...", approved suggestions, or the Memory tab. They are never
made from results, and anything that looks like personal data or a secret is
refused (`looks_private`).

For each question, `MemoryService.retrieve` scores the workspace's active memories
that apply to the data source:

| Signal | Score |
|---|---|
| Words shared with the question (stop words removed) | up to +2.0, relative to the question's length |
| Matched by the full-text index | up to +1.0 |
| A standing instruction (preference, default filter, date range, chart) | +1.5 |
| Pinned | +5.0 |
| From this conversation | +1.0 |
| Its importance (0 to 1) | + importance |
| Recently updated | up to +1.0, fading over 90 days |

A memory with no shared words, no index match, and that is neither standing nor
pinned is skipped. The rest are sorted by score and added until the memory share of
the budget is used; a shorter, lower-ranked memory can still fit after a longer one
didn't. Each included memory carries a citation ("Based on a memory you added on
November 8"), shown under the answer and in the context panel.

## Conversation categorisation

A new conversation gets a category from its first question
(`agent/conversation.categorize`). Rules are checked in order and the first match
wins:

| Category | Matches (examples) |
|---|---|
| Data changes | the model's intent was `modify` |
| Trends over time | over time, trend, monthly, by month, growth, year over year |
| Rankings | top, highest, lowest, best, worst, most, rank |
| Breakdowns | break down, share of, distribution, for each, by, per |
| Totals and counts | how many, how much, count, total, average, number of |
| Lookups | show, list, which, who, find, what is |
| General | anything else |

The rules are plain keyword lists in `CATEGORY_RULES`, so they cost nothing and give
the same answer every time. To add a category, add a `(name, keywords)` pair in the
position it should take priority, and add a case to `tests/test_chat.py`. Users can
change a conversation's category from its **Manage** menu.

## Adding a data source adapter

The agent talks only to `DataSourceAdapter` (`data/adapters.py`). To support another
database:

1. **Implement the interface** in a new file under `data/`:

   ```python
   class PostgresAdapter(DataSourceAdapter):
       def __init__(self, url: str, *, name: str, semantics=None):
           self.info = DataSourceInfo(name=name, type="postgres", dialect="postgres",
                                      connection={"url": url}, writable=False)
           self.semantics = semantics or SemanticConfig()

       def schema(self, refresh=False) -> SchemaInfo: ...       # tables, columns, keys, sample rows
       def preview(self, table, limit=50) -> pd.DataFrame: ...  # only for tables in schema()
       def run_read(self, validated) -> QueryResult: ...        # on a read-only connection or role
   ```

   `dry_run_write` and `commit_write` are optional; leave them out for read-only
   sources.

2. **Keep the guarantees the agent relies on:**
   * reads must run on a connection or role that cannot write (a read-only
     transaction or a database role with SELECT only);
   * enforce `self.info.limits`: stop queries after `query_timeout_s` (a statement
     timeout) and fetch at most `max_rows`;
   * `preview` must accept only table names that exist in `schema()`;
   * a write's dry run must roll back, and `commit_write` must roll back if the row
     count differs from `expected_rows`;
   * never put passwords in `info.connection` (it's stored and shown). Read them
     from secrets when connecting; `redact()` is a backstop, not the plan.

3. **Set the dialect.** `info.dialect` must be a sqlglot dialect name (`postgres`,
   `mysql`, `duckdb`, ...). The validator parses with it, and the prompt adds
   dialect-specific notes from `prompts.DIALECT_RULES`; add an entry there if the
   model needs reminders for this database.

4. **Register it:** `register_adapter_type("postgres", PostgresAdapter)` (in
   `data/registry.py`), then create it with `create_adapter("postgres", url=..., name=...)`
   and add it to a workspace with `registry.add(...)`.

5. **Restoring after a restart.** `DataSourceRegistry.load` currently rebuilds
   file-based SQLite sources. A server-based source needs a branch there that calls
   `create_adapter` with credentials taken from secrets, since they're never stored.

6. **Test it** like `tests/test_datasources.py` does for SQLite: the dialect's
   dangerous statements are refused, limits hold, and the prompt contains no
   assumptions from the sample store.

`suggest_config(schema)` in `data/metadata.py` proposes a starting business context
(date fields, joins, likely-sensitive columns, starter questions) for any schema.

## Theming and charts

See [design-system.md](design-system.md): changing tokens, regenerating
`config.toml` and the Figma tokens, adding a component, and adding a chart type.

## Storage and migrations

`storage/migrations.py` holds numbered migrations. To change the structure, append a
new `Migration` with the next number; never edit an applied one. Before upgrading,
the database is backed up next to itself, and a failed migration rolls back.
Repositories in `storage/repositories.py` take a `workspace_id` on every call; keep
it that way, since that is what keeps workspaces apart.

## Running the tests

```bash
python -m pytest            # about 400 tests, all offline, under a minute
python -m pytest tests/test_security.py -v
```

Question limits are switched off in tests by the root `conftest.py`; the security
tests switch them on explicitly. The fake models in `tests/fakes.py` answer by
keyword, so tests can say exactly which SQL the "model" produced.
