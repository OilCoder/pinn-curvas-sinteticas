<!-- description: Require Claude to provide and use verification before declaring a task complete -->

# Verification

A task is not complete until its result has been verified.
This rule operationalizes the official Claude Code guidance:

> *"Give Claude a way to verify its work. This is the single highest-leverage thing you can do."*
> — [Claude Code best practices](https://code.claude.com/docs/en/best-practices)

## Verification is mandatory

Before declaring any task as completed (or marking a phase as `(COMPLETED)` per `plan-format.md`),
the change must be verified through at least one of the following channels:

| Type of change | Required verification |
|---|---|
| New function or class | A test that exercises it, run successfully |
| Bug fix | A regression test that reproduces the bug, then passes |
| Refactor | Existing test suite runs successfully; no behavioral diff |
| Type changes | Type-check passes (`mypy`, `tsc --noEmit`, etc.) |
| Style/format | Linter and formatter run cleanly on the changed files |
| UI changes | Screenshot or visual diff (e.g., Claude in Chrome) |
| Data pipeline | Run on a sample input; output matches expected shape |
| Documentation | Read-through pass; commands/snippets actually execute |

If no verification channel exists for the type of change, **say so explicitly** in the response
rather than claiming the task is done.

## What counts as verification

- **Pass**: a real command (`pytest`, `npm test`, `tsc`, `ruff check`, `cargo test`, etc.)
  exited with success after the change.
- **Pass**: a script or notebook produced expected output that the user can inspect.
- **Pass**: a screenshot or browser run confirms the UI change matches the spec.
- **Not enough**: "the code looks correct".
- **Not enough**: "the diff compiles in my head".
- **Not enough**: "I changed it the way you asked".

## Where verification commands come from

Each project should declare its verification commands in `project-guidelines.md` under
**Tech constraints** or in `package.json` / `pyproject.toml` scripts. Typical entries:

```text
test:        pytest -q  |  npm test  |  cargo test
type-check:  mypy src/  |  tsc --noEmit
lint:        ruff check  |  npx eslint .
format:      ruff format  |  npx prettier --check .
```

Skills that conclude work (`/phase-executor`, `/test`, `/bitacora`) must read these commands
and run them before reporting a task as complete.

## When verification fails

- Treat failure as part of the task, not a separate problem.
- Do **not** mark the task as complete.
- Do **not** suppress, skip, or weaken assertions to make verification pass.
- Address the **root cause**, not the symptom — per the official guidance:
  > *"Address root causes, not symptoms."*
- If the failure is outside the scope of the current task, stop and report it to the user
  before continuing.

## Forbidden

- Disabling tests with `@skip`, `xfail`, `it.skip`, `@pytest.mark.skip`, or equivalents
  to make a suite green, unless the user explicitly asks for it.
- Bypassing pre-commit or pre-push hooks (`--no-verify`, `--no-gpg-sign`).
- Marking a phase or task as `(COMPLETED)` while a verification command is failing.

## Cross-references

- See `plan-format.md` for how to mark tasks complete.
- See `phase-executor/SKILL.md` for verification at end of phase.
- See `test/SKILL.md` for writing verification tests.
- See `code-change.md` for behavior on multi-file changes.
