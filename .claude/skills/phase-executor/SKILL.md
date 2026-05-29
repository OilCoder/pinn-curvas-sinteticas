---
name: phase-executor
description: >
  Read and execute a phase from the project plan in order.
  Use when the user says "execute the phase", "work on phase N",
  "implement phase N", "run phase", or references a phase by name or number.
argument-hint: "[phase number or name]"
allowed-tools: Read Write Edit Bash Bash(date:*) Bash(git:*) Grep Glob
---

# Phase Executor

Read `PLAN.md`, present an execution plan, wait for approval, and execute
the phase's tasks in order, updating checkboxes as each one is completed.

## Pre-rendered context

- **Date**: !`date +%Y-%m-%d`
- **Branch**: !`git branch --show-current 2>/dev/null || echo "(not a git repo)"`
- **PLAN.md**:
```!
[ -f todo/PLAN.md ] && cat todo/PLAN.md || echo "(no PLAN.md found — run /plan-writing first)"
```
- **Project guidelines (verification commands and tech constraints)**:
```!
[ -f .claude/rules/project-guidelines.md ] && sed -n '/## Verification commands/,/^## /p' .claude/rules/project-guidelines.md || echo "(no project-guidelines.md found)"
```

## Before writing any code

1. The PLAN.md is already pre-rendered above. Identify the requested phase from `$ARGUMENTS` and extract its task list.
2. Present a short plan to the user:
   - Files to create or modify
   - Order of execution
   - Any ambiguity that needs user input
3. **Wait for explicit user approval** before proceeding.

## Execution rules

### Scope

- Only create or modify files listed in the phase tasks.
- Never touch files from other phases.
- If a required file from a previous phase is missing, stop and inform the user before continuing.

### Order

- Execute tasks in the order they appear in `PLAN.md`.
- **Skip discarded tasks** (those in `~~strikethrough~~` form) — they are part of
  the historical record but are not active work.
- Complete each task fully before moving to the next.
- Mark each task as `- [x]` in `PLAN.md` immediately after completing it.
- If a task becomes obsolete during execution (e.g., a previous task made it
  redundant), do **not** silently skip — instead, propose discarding it to the
  user and mark it per `plan-format.md` (`~~task~~ (discarded YYYY-MM-DD: reason)`).

### Code

- Follow all rules in `.claude/rules/`.
- Apply the project's style, naming, and logging conventions.
- Respect `doc-enforcement`: docstrings on all public functions.

### Conventions

- Project-specific conventions are respected per `project-guidelines.md`.
- Do not assume conventions — read them from the guidelines file.

## Before completing the phase — verification gate

Per `verification.md`, a phase must not be marked `(COMPLETED)` until its
result has been verified. Before declaring the phase done:

1. Read `project-guidelines.md` to find the project's verification commands
   (under **Tech constraints** or referenced from `package.json` /
   `pyproject.toml`).
2. Run the relevant subset for the type of change:
   - Code changes → test command (`pytest`, `npm test`, etc.)
   - Type-annotated code → type checker (`mypy`, `tsc --noEmit`)
   - Always → linter and formatter on the changed files
3. If any verification fails:
   - Do **not** mark the phase complete.
   - Address the root cause, not the symptom.
   - Re-run the verification.
4. If no verification command exists for this type of change, say so
   explicitly in the report instead of claiming verified.

## After completing the phase

1. Mark the phase title as `(COMPLETED)` in `PLAN.md` — only after the
   verification gate above has passed.
2. Report to the user:
   - Files created
   - Functions implemented
   - Decisions made during execution
   - Verification commands run and their outcome
3. Flag anything that needs user review before starting the next phase.
4. If the `/bitacora` skill is available, suggest logging the session.
