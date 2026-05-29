# Customization Guide

How to adapt the base rules, skills, agents, and hooks to a new project.

## Recommended path: install as plugin

```
/plugin marketplace add OilCoder/claude-project-base
/plugin install claude-project-base
/setup     # answers 2 questions, creates the minimum
```

Updates: `/plugin update claude-project-base` pulls newer versions across all your projects.

## Manual path (fallback)

1. Copy `.claude/` contents to the new project and rename `CLAUDE-TEMPLATE.md` → `CLAUDE.md` at the project root.
2. Run `/setup` (or follow its checklist manually).
3. `/setup` already customizes `project-guidelines.md`, `code-style.md` naming, `doc-enforcement.md` paths, and the linter hook per stack. You only fill in the project-specific parts (verification commands, tech constraints).

## Folder philosophy: minimum + organic growth

`/setup` creates only 4 folders:

```
.claude/         ← rules, skills, agents, hooks
todo/            ← plans, bitácoras
documentation/   ← code docs (target of /document)
docs/            ← reserved for GitHub Pages
```

Everything else (`src/`, `pipeline/`, `tests/`, `data/`, `models/`, `experiments/`) is created by you, when the project genuinely needs it. The base supports any of those layouts via path-scoped rules — `doc-enforcement.md` already covers `src/`, `lib/`, `app/`, `pipeline/`.

## Greenfield vs existing project

| Scenario | First step |
|---|---|
| Greenfield (no CLAUDE.md, no AGENTS.md) | Run `/setup` directly |
| Existing repo with conventions but no Claude files | Run **`/init` first**, then `/setup` to layer the base on top |
| Project with existing `AGENTS.md` (Cursor, Aider, etc.) | Run `/setup` and let it import `AGENTS.md` via `@AGENTS.md` in `CLAUDE.md` |
| Project with existing `CLAUDE.md` | Run `/setup` in merge mode (it will ask before overwriting) |

## When to use a rule, a skill, an agent, or a hook

The base operates on four layers. Match the mechanism to the goal:

| You want to... | Use a... | Why |
|---|---|---|
| Guide Claude's decisions while writing code | **Rule** (`.claude/rules/*.md`) | Loads as advisory context. Influences choices but doesn't enforce. |
| Run a multi-step procedure on demand | **Skill** (`.claude/skills/*/SKILL.md`) | Loads only when invoked, no context cost. Reusable workflow. |
| Review or design in isolation | **Agent** (`.claude/agents/*.md`) | Runs in a fresh context window. Doesn't pollute the main conversation. |
| Guarantee something happens every time | **Hook** (`.claude/settings.json`) | Deterministic. Fires on tool events regardless of Claude's state. |
| Block dangerous commands | **Hook** (PreToolUse) | Only hooks can block tool calls. |
| Auto-format on save | **Hook** (PostToolUse) | Linters are faster and more reliable than asking Claude to follow style rules. |
| Get a second opinion on a diff | **Agent** (`code-reviewer`) | Fresh context = no bias toward code Claude just wrote. |
| Bootstrap a recurring workflow | **Skill** | Skills compose; rules don't. |
| Combine several skills at a milestone | **Composite skill** (`/checkpoint`) | Skills can orchestrate other skills' procedures inline. |
| Pre-fetch shell data for a skill | **`` !`command` `` injection** | Output is rendered into the skill content before Claude sees it; saves turns. |

The rule of thumb from the official Claude Code docs:
> *"Hooks run scripts automatically at specific points in Claude's workflow. Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action happens."*

## Rules: what is generic vs what to customize

### code-style.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Function structure | Single responsibility, helpers | Configure linter limits (`ruff C901`, `eslint max-lines-per-function`) via hooks |
| Minimalism | No boilerplate, no premature abstractions | — |
| Naming | `snake_case`, descriptive names | Change to `camelCase` for JS/TS |
| Comments and style | Step/Substep + emojis, no trivial comments | Comment language (English/Spanish) |
| Imports | Logical grouping, only used ones | Add framework conventions |
| Scope discipline | Do not solve more than requested | — |

### file-naming.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| General conventions | `snake_case`, descriptive, no spaces | File name language |
| Execution order | `NN_` numeric prefix or `sNN[x]_` | Choose project pattern |
| Output files | Dedicated output folder concept | Exact pattern (`{id}_{type}.png`, etc.) |
| Test/Debug files | `test_<module>_<case>`, `dbg_<slug>` | — |
| Documentation | `NN_<slug>.md` in `documentation/` | — (always `documentation/`, never `docs/`) |

### code-change.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Edit scope | Minimal block, no refactor | — |
| Structural integrity | Preserve order, format, separators | — |
| Multi-file changes | Dependency order, read first | — |
| Output format | Only modified code | — |
| Forbidden | No debug code, no new deps | Add project-specific prohibitions |

### logging-policy.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Print usage | Temporary in dev, clean before commit | Message language |
| Logging usage | Module-scoped loggers, no debug noise | — |
| Progress output | Visible progress concept | Tools (`tqdm`, MRST indicators) |
| Cleanup | Isolate in `debug/`, disposable scaffolding | — |
| Exceptions | Notebooks and CLI allowed | — |

### doc-enforcement.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Docstring required | Public functions, private if nontrivial | — |
| Module header | Module docstring mandatory | — |
| Docstring structure | Args, Returns, Raises | Format (Google Style, NumPy, JSDoc) |
| Enforcement scope | Scope concept | Define files (`src/`, `pipeline/`) |

### docs-style.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Required sections | Title, Workflow, I/O, Math, Code Ref | Add/remove sections |
| Style | Concise, current code, no TODOs | — |
| `documentation/` vs `docs/` | Fixed split: code docs in `documentation/`, GH Pages in `docs/` | — |
| Bilingual structure | Subfolders by language under `documentation/` | Languages used |

### plan-format.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| File format | Goal/Stack/Phases structure | — |
| Writing rules | Flat checkboxes, specific, not vague | — |
| Update rules | Mark `[x]` with date, never delete | — |

### verification.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Verification is mandatory | Verification gate before declaring done | — |
| What counts as verification | Real command exits 0 / screenshot / sample run | — |
| Where verification commands come from | Read from `project-guidelines.md` Tech constraints | Set actual commands in `project-guidelines.md` (`test`, `type-check`, `lint`, `format`) |
| When verification fails | Treat failure as part of the task, address root cause | — |
| Forbidden | No `--no-verify`, no `@skip` to make suite green | Add project-specific bypasses to forbid |

### delegation.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Decision matrix | Where each task type runs | — |
| Heuristic | "Do workers need to talk to each other?" → team | — |
| Cost awareness | Token cost ladder: main → subagent → team | — |
| Agent Teams are experimental | Opt-in via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` | Add stack-specific delegation patterns if any |

### memory-policy.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| Two systems | Bitácora (human, narrative) vs MEMORY.md (Claude, factual) | — |
| What goes where | Clear boundary, examples per system | Bitácora language (Spanish default) |
| Forbidden | No narrative in MEMORY.md | — |
| Maintenance | `/consolidate-memory` for MEMORY.md, Cowork for bitácora | — |

### commit-style.md

| Section | Generic (do not touch) | Customize |
|---|---|---|
| 7 prefixes | `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf` | — |
| Format | `<type>(<scope>): <subject>` + body | — |
| Subject rules | Imperative, ≤72 chars, lowercase, no period | — |
| Decision tree | Pick prefix based on dominant change | — |

### project-guidelines.md

| Section | Action |
|---|---|
| Rules index | Update if rules were removed |
| Skills index | Update if skills were removed |
| Validation modes | Choose initial mode (`suggest`/`warn`/`strict`) |
| Project structure | Replace with actual folder tree |
| Tech constraints | Replace with actual constraints |
| Policies | Add if the project uses immutable principles |

## Skills: which to keep

| Skill | When to keep | When to remove |
|---|---|---|
| checkpoint | Always — closes the milestone loop | Never |
| bug-fix | Always — TDD bug fix workflow | Never |
| bitacora | Always — bridge with Cowork/Obsidian | Never |
| plan-writing | Always — any project needs a plan | Never |
| phase-executor | Projects with defined phases | Very small or exploratory projects |
| test | Projects with code | Documentation-only projects |
| investigate | Projects with code | Documentation-only projects |
| document | Projects needing technical docs | Simple projects |
| doc-enforce | Projects with many functions | Short scripts |
| setup | Only in the base — do not copy to projects | — |

## Agents: which to keep

| Agent | When to keep | When to remove |
|---|---|---|
| code-reviewer | Always — pre-commit fresh-context review | Never |
| security-reviewer | Projects with auth, network, user input, persistence | Pure-internal scripts with no security surface |
| architect | Projects with non-trivial features | Pure scripts / one-off projects |
| implementer | Projects where you delegate self-contained tasks autonomously | Small projects where you write everything yourself |

## Greenfield vs existing project

| Scenario | First step |
|---|---|
| Greenfield (no CLAUDE.md, no AGENTS.md) | Run `/setup` directly |
| Existing repo with conventions but no Claude files | Run **`/init` first**, then `/setup` to layer the base on top |
| Project with existing `AGENTS.md` (Cursor, Aider, etc.) | Run `/setup` and let it import `AGENTS.md` via `@AGENTS.md` in `CLAUDE.md` |
| Project with existing `CLAUDE.md` | Run `/setup` in merge mode (it will ask before overwriting) |

The official Claude Code docs document `@AGENTS.md` as the way to share a
single instructions source across multiple AI coding tools without duplication.

## Example: Python ML project

1. **code-style**: comment language = English; line limit enforced by `ruff C901` via hook
2. **file-naming**: `NN_` pattern, outputs = `outputs/{experiment}_{metric}.png`
3. **logging-policy**: progress = `tqdm`, language = English
4. **doc-enforcement**: format = Google Style, scope = `src/`
5. **docs-style**: location = `docs/`, no bilingual
6. **project-guidelines**: mode = `warn`, actual structure, GPU/CUDA constraints
7. **Skills**: all except setup

## Example: static web project

1. **code-style**: naming = `camelCase` for JS, `snake_case` for files
2. **file-naming**: no numeric prefix, fixed names (`index.html`, `style.css`)
3. **logging-policy**: only `console.log` allowed in dev
4. **doc-enforcement**: format = JSDoc, scope = `js/`
5. **docs-style**: location = `docs/`, simplified sections (no Math)
6. **project-guidelines**: mode = `suggest`, browser compatibility constraints
7. **Skills**: bitacora, plan-writing, document. No test/investigate/doc-enforce.
