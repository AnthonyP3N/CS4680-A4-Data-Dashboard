# AskData design system

AskData's look comes from one file, `ui/theme.py`. The colours, type, spacing and
radii live there as tokens. Every other place that needs them is generated from
it or reads it at run time:

| Where | How it gets the tokens |
|---|---|
| Streamlit's own widgets (buttons, inputs, tabs, tables) | `.streamlit/config.toml`, generated: `[theme.light]` and `[theme.dark]` |
| AskData's own pieces (question bubble, step trail, notices, context panel) | CSS variables injected by `theme.inject()` for the viewer's theme |
| Charts | `theme.CHART_PALETTES[theme]`, passed to `agent.charts.build_figure` |
| Figma | `docs/design-tokens.json` (W3C design-token format), generated |

After changing any token, regenerate the two files and run the tests:

```bash
python -m ui.theme          # rewrites .streamlit/config.toml and docs/design-tokens.json
python -m pytest tests/test_design.py
```

The tests fail if either generated file is out of date, if any text and
background pair falls below its WCAG contrast target in either theme, or if the
stylesheet contains something Streamlit's markdown would print as text.

## Identity

AskData is a tool for reading numbers, so the interface stays quiet: a cool
neutral canvas, one indigo accent for things you can act on, and one signature,
the yellow highlighter stroke under the wordmark. The highlighter is used
nowhere else, so it stays recognisable. Answers are set larger than the
interface around them, because the answer is what people came for.

Type is Figtree for everything people read, and JetBrains Mono only for SQL.

## Tokens

### Colour roles

Each role exists in both themes with the same name. Use the role, never a hex value.

| Role | Used for | Light | Dark |
|---|---|---|---|
| `canvas` | Page background | `#F7F8FA` | `#10141C` |
| `surface` | Result cards, notices, the context panel | `#FFFFFF` | `#171C26` |
| `surface-sunken` | Sidebar, inputs, code | `#EDEFF4` | `#1F2532` |
| `ink` | Main text | `#1B2333` | `#E7EAF1` |
| `ink-muted` | Secondary text, metadata | `#596273` | `#A6AEC0` |
| `ink-faint` | Separators and placeholders only (not for reading) | `#7A8294` | `#7C8598` |
| `line` | Borders and rules | `#DCE0E8` | `#2E3545` |
| `chip` | Step-trail pills, the open conversation | `#E9ECF4` | `#252C3A` |
| `accent` | Links, focus rings, primary buttons | `#3B4CCA` | `#8E9CFF` |
| `on-accent` | Text on `accent`, `success` and `danger` | `#FFFFFF` | `#10141C` |
| `highlight` | The wordmark's highlighter stroke | `#FFD84D` | `#8A6A00` |
| `success` | Passed steps, "read-only" | `#0B7A6E` | `#3CC8B4` |
| `attention` | Self-corrected steps, "changes allowed" | `#9A5B00` | `#E5A445` |
| `danger` | Blocked steps, errors, destructive confirmations | `#B93A0B` | `#FF8A5B` |
| `danger-surface` | Background of error notices and confirmations | `#FFF4EC` | `#2B1C16` |
| `bubble` / `on-bubble` | The person's question | `#1B2333` / `#FFFFFF` | `#E7EAF1` / `#10141C` |

Contrast targets (checked by `test_every_colour_pair_meets_wcag_contrast`): 4.5:1 for
text people read (`ink`, `ink-muted`, `accent` and `danger` on `canvas` and `surface`,
`on-accent` on `accent`), 3:1 for large text, icons and status dots.

### Chart colours

Eight series colours per theme, in the order they're used. The first is the
theme's accent, so a single-series chart matches the rest of the interface.

### Type scale

A 1.2 ratio above a 1rem base: `md` 1rem, `lg` 1.2rem, `xl` 1.44rem, `2xl` 1.728rem,
`3xl` 2.074rem, `display` 2.986rem (the big number in a metric answer). Below the base the
steps are smaller so small print stays readable: `sm` 0.917rem, `xs` 0.833rem. Weights: regular 400, medium 500, semibold 600, bold 700, heavy 800.
Running text is capped at 72 characters a line (`MEASURE`).

### Space, radius, elevation

Spacing steps `1`–`7` (0.25rem to 3rem). Radii: `sm` 0.5rem (buttons), `md` 0.625rem
(inputs, cards), `lg` 1rem (question bubble), `pill` (step trail). Elevation: `flat`, `raised`
(a hairline shadow) and `overlay` (popovers); most surfaces use a border instead of a shadow.

## Using the tokens in Figma

1. Install the **Tokens Studio for Figma** plugin.
2. In the plugin, choose **Load from file** and pick `docs/design-tokens.json`.
3. The file has three sets: `core` (type, space, radius, elevation) and `light` and
   `dark` (colour roles and chart colours). Turn on `core` plus one theme.
4. Use **Create styles** to turn them into Figma colour and text styles.

The tokens flow one way: from code to Figma. To change a colour, change it in
`ui/theme.py`, run `python -m ui.theme`, and reload the file in the plugin.

Suggested page structure for the Figma file, matching the app:

| Figma page | Contents |
|---|---|
| Foundations | Colour roles (both themes), type scale, spacing, radii |
| Components | Each entry in the component list below, in its states |
| Ask | Empty state, an answer, a clarifying question, a change awaiting confirmation, an error |
| Other tabs | Saved, Memory, Data, Activity log, Settings |
| Responsive | Desktop (1440), laptop with sidebar (1024), phone (390) |

## Components

All reusable pieces are functions in `ui/components.py`. They escape every piece of
text they're given, so they're safe to use with model output and uploaded data.

| Component | Function | Notes |
|---|---|---|
| Wordmark | `wordmark(tagline)` | The highlighter stroke |
| Conversation header | `conversation_title(title, subtitle)` | |
| Question bubble | `question_bubble(text)` | `aria-label="Your question"` |
| Step trail | `step_trail(steps, seconds)` | A list (`role="list"`); each step passed, corrected or blocked |
| Answer text | `answer_text(text)` | The answer's summary sentence, set larger than the interface |
| Metadata line | `meta(text)` | Source, local time, rows, query time |
| Source line | `source_line(text)` | The citation at the foot of a result card |
| Metric | `metric(value, label)` | The big-number view; `role="group"` with the label as its name |
| Notice | `notice(text, kind, lead)` | `kind` is `ok`, `ask`, `warn` or `err`; errors use `role="alert"`, the rest `role="status"` |
| Empty state | `empty_state(title, body)` | |
| Confirmation | `confirm_panel(title, paragraphs)` and `confirm_button(...)` | Two-step confirm for destructive actions |
| Details list | `detail_list(pairs)` | Key–value pairs in "How I got this" |
| Panel heading / value | `panel_heading(text)`, `panel_value(text, status)` | The context panel; `status` adds a coloured dot |
| Table | `dataframe(df, height)` | |
| Chart | `plotly(fig, key)` | A unique `key` is required |

Larger pieces are composed from these in the views: the result card
(`chat_view._render_result_card`), the context panel (`context_view.render`) and the
sidebar (`conversation_view.render_sidebar`).

### Adding a component

1. Add a function to `ui/components.py` that builds its markup from escaped text
   (`esc()`) and gives it one class name.
2. Add its rules to `_RULES` in `ui/theme.py` using only `var(--token)` values.
   Don't add CSS comments with blank lines in them; they're stripped, but keep rules compact.
3. Give it an accessible role or label if it carries meaning beyond its text.
4. Add a test in `tests/test_ui.py` or `tests/test_design.py` that renders it.

## How themes are chosen

Streamlit follows the viewer's system setting, and the viewer can switch it from
the app menu (⋮ › Settings). `theme.current_theme()` reads the choice from
`st.context.theme`, and `theme.inject()` adds the matching CSS variables. Charts
are drawn with the same theme's series colours. There is no AskData-specific theme
switch, because a second switch next to Streamlit's would disagree with it.

## Layout

| Width | Layout |
|---|---|
| Wider than 1180px | Sidebar, chat, and the context panel on the right |
| 641–1180px | Sidebar and chat; the context panel moves below the chat |
| 640px and narrower | The sidebar starts closed; everything is one column with smaller margins |

The context panel can be hidden from the sidebar (**Show context panel**).

## Accessibility

* Every text and background pair meets WCAG AA contrast in both themes (tested).
* Keyboard focus is always visible: a 2px `accent` outline on `:focus-visible`.
* Motion is turned off for people who ask their system for reduced motion.
* Status is never shown by colour alone: steps have a text label and an icon shape,
  safety status has words next to its dot.
* Shortcuts: Alt+N starts a new conversation, Alt+G regenerates the latest answer.

## Adding a chart type

Charts live in `agent/charts.py`, which has no Streamlit code.

1. Add the name to `VIEWS` and a button label to `VIEW_LABELS`.
2. In `resolve_spec`, add any rules for when the data can't be drawn this way and
   what to show instead (for example, `area` becomes `line` when values are negative).
3. In `available_views`, add the conditions under which it's offered.
4. In `build_figure`, draw it with Plotly Express, passing `color_discrete_sequence=palette`
   so it follows the theme.
5. Add the name to the `chart_type` enum in `RESPONSE_SCHEMA` (`agent/prompts.py`) if the
   model should be able to choose it, and mention when to use it in the system prompt.
6. Add cases to `tests/test_charts.py` and `tests/test_design.py`
   (`test_every_offered_view_draws_or_is_deliberately_not_a_chart` covers drawing).

Any view that can't be drawn for a result falls back to a table with a note saying why.
