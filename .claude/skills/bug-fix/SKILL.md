---
name: bug-fix
description: >
  TDD workflow for fixing a bug: reproduce, write failing test, fix, confirm,
  log learning. Use when the user says "fix this bug", "this is broken",
  "X doesn't work as expected", "regression in Y".
argument-hint: "[bug description or issue number]"
allowed-tools: Read Write Edit Bash(git log:*) Bash(git diff:*) Bash(git status:*) Bash(pytest:*) Bash(npm test:*) Bash(date:*) Grep Glob
---

# Bug Fix

Fix a bug **with regression preserved**: reproduce → failing test → fix → test passes
→ bitácora entry. This is distinct from `/investigate` (which only explores) and
from `/simplify` (which refactors without changing behavior).

## Pre-rendered context

- **Date**: !`date +%Y-%m-%d`
- **Branch**: !`git branch --show-current 2>/dev/null || echo "(not a git repo)"`
- **Working tree**: !`git status --short 2>/dev/null`
- **Last 5 commits**:
```!
git log --oneline -5 2>/dev/null
```
- **Existing tests directory layout**:
```!
ls -1 tests/ 2>/dev/null | head -n20 || echo "(no tests/ folder yet)"
```

## When to use vs not use

| Use `/bug-fix` when... | Use a different skill when... |
|---|---|
| You have a bug and want it fixed permanently | You're still exploring (use `/investigate`) |
| The fix should preserve a regression test | You want to refactor without behavior change (use `/simplify`) |
| Behavior is wrong (output, error, crash) | Code is ugly but works (use `/simplify`) |
| You want a `fix:` commit at the end | You're just adding features (use `/phase-executor`) |

## Procedure

### 1. Frame the bug

If `$ARGUMENTS` is provided, use it as the bug description.
Otherwise, ask the user:
- What's the observed behavior?
- What's the expected behavior?
- Is there a stack trace, error message, or log line?
- When did it start? (commit, deployment, environment change)

State the framing back to the user before proceeding.

### 2. Reproduce

Before writing any code, **reproduce the bug**:

- Identify the smallest input or sequence that triggers it.
- Run the exact reproduction in the project's environment.
- Confirm with the user that what you reproduced is the same bug.

If you cannot reproduce, **stop and report**. Do not write a fix for a bug you
cannot trigger.

### 3. Write a failing test (regression)

Write a test that:
- Encodes the **expected** behavior (not the broken one).
- Fails today because of the bug.
- Lives at `tests/test_<module>_<bug-slug>.py` (per `file-naming.md`).

Run the test and **confirm it fails for the right reason** (not a syntax error
or wrong import). The failure message should describe the bug.

### 4. Diagnose root cause

Read the affected module(s). Identify the line(s) that produce the wrong behavior.
Per `verification.md`, address **root cause, not symptom**:

- Symptom-fix example (wrong): catch the exception and ignore it.
- Root-cause fix example (right): handle the malformed input that produced the exception.

If the root cause is outside the scope of the current request (e.g., a library
bug, an API contract change), stop and report — do not patch around it without
the user knowing.

### 5. Implement the fix

Apply the project's rules:
- `code-style.md` for layout and step/substep markers
- `code-change.md` for scope discipline (don't refactor unrelated code)
- `logging-policy.md` for prints/loggers
- `doc-enforcement.md` for docstring updates if the contract changed

Keep the fix minimal. The smaller the diff, the easier the review and the
smaller the regression surface.

### 6. Confirm the fix

Run **all** of:
- The regression test from step 3 → must now pass.
- The full test suite for the affected module → must still pass.
- The linter and type checker on the changed files.

If any of these fail, the fix is incomplete. Per `verification.md`, do not
claim done while verification is failing.

### 7. Bitácora entry

Append a bitácora entry to `todo/bitacora-YYYY-MM-DD.md` per `bitacora/SKILL.md`:

- Technical changes: file:line of the fix, file of the regression test
- Design decisions: why this fix vs alternatives considered
- Learnings: what about the codebase / domain made this bug possible — this
  is the part you'll want in your knowledge base later

### 8. Compose commit

Per `commit-style.md`, the commit type is **always `fix:`** for bug fixes.

```
fix(<scope>): <subject describing what's now correct>

<body: 2-4 lines on root cause and why this fix>
```

Example:
```
fix(loader): handle LAS files with malformed ~Other section

lasio crashed when ~Other appeared before ~Curve. The parser
assumed strict section order. Added section re-ordering pass
before parse. Regression in tests/test_loader_malformed_other.py.
```

Stage specific files (the fix file + the test file). Do not `git add -A`.

### 9. Hand off

Report to the user:
- Bug description (one line)
- Root cause (one line)
- File(s) changed
- Test file added
- All verification commands run + outcome
- Suggested next: `/checkpoint` to commit, push, and update PLAN.md if relevant

## Rules

- **Never skip the failing test.** A bug fix without a regression test will
  recur. The test is the value, the fix is the consequence.
- **Never weaken existing tests** to make the suite green. If an old test was
  asserting the buggy behavior, fix the test (and document it in the bitácora
  as a behavior correction, not as a "test fix").
- **No `--no-verify`.** If pre-commit hooks fail, fix the underlying issue.
- **Stay in scope.** If you discover another bug while fixing this one, list
  it in the bitácora's pending items, do not fix it inline.
- **Always `fix:` prefix** in the commit, per `commit-style.md`.
