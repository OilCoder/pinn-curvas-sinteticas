---
paths:
  - "todo/**/*.md"
---

<!-- description: Define format and update rules for project plan files in todo/ -->

# Plan Format

Every plan lives in `todo/` as a Markdown file.
This rule defines the format; the skill `/plan-writing` handles the procedure.

## Plan location

- General project plan: `todo/PLAN.md`
- Phase-specific plans (if needed): `todo/phase_NN_<name>.md`
- If `todo/` does not exist, create it.

## File format

Use this structure, nothing more:

```markdown
# <Project or Phase Name>

## Goal
One sentence. What does this plan accomplish?

## Stack (only in PLAN.md)
Simple table: Layer | Technology

## Structure (only in PLAN.md)
Folder tree showing key paths and their purpose.

## Phases

### Phase N — <Name>
- [ ] Task description (file or module it targets)
- [ ] Task description
- [x] Task completed (YYYY-MM-DD)
- ~~Discarded task description~~ (discarded YYYY-MM-DD: short reason)

### Phase N+1 — <Name>
- [ ] ...

## Conventions
Short bullet list of naming rules or constraints relevant to this plan.
```

## Writing rules

- Use plain Markdown only. No HTML, no frontmatter, no badges.
- Tasks use `- [ ]` checkboxes. One task = one action.
- Each task should name the file or module it targets.
- No sub-tasks, no nested checkboxes. Keep it flat.
- No status tables, no emoji columns, no progress bars.
- Avoid vague tasks like "improve X" or "refactor Y". Be specific.
- Phases must be independent — a phase should not depend on assumptions from another phase unless explicitly stated.

## Update rules

Three task states:

| State | Marker | Format |
|---|---|---|
| Pending | `- [ ]` | `- [ ] Task description` |
| Completed | `- [x]` | `- [x] Task description (YYYY-MM-DD)` |
| Discarded | `~~strikethrough~~`, no checkbox | `- ~~Task description~~ (discarded YYYY-MM-DD: reason)` |

- Mark completed tasks as `- [x]` immediately after finishing them.
- **Discarded tasks**: when a task becomes obsolete (project pivoted, scope cut,
  approach abandoned, replaced by another task), do **not** delete it. Convert
  it to the discarded form: drop the checkbox, wrap the description in `~~...~~`
  strikethrough, and append `(discarded YYYY-MM-DD: reason)` in plain text.
- Reasons for discarding must be specific. Examples:
  - `(discarded 2026-04-25: scope creep, moved to Phase 5)`
  - `(discarded 2026-04-30: pivoted from JWT to Auth0 SDK)`
  - `(discarded 2026-05-02: experiment failed, see bitacora-2026-05-02.md)`
- The bitácora's `Errors` section captures the **detail** of the discard. The
  PLAN.md preserves the **public record** that the option was considered and dropped.
- Do not delete tasks, ever — completed, discarded, or pending.
- Do not add new tasks to a phase without user approval.
- If a phase is fully completed (all tasks `[x]` or discarded), add `(COMPLETED)` to the phase title.
- Never rewrite or reformat existing content — only update task states and phase titles.

## Cross-references

- See `plan-writing/SKILL.md` for the procedure to create and update plans.
- See `phase-executor/SKILL.md` for automated phase execution.
- See `bitacora/SKILL.md` for session logging that feeds into plan updates.
