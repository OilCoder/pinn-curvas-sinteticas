---
name: checkpoint
description: >
  Combined checkpoint workflow: update plan, document changes, log session,
  commit, and (optionally) push and open PR. Use when reaching a meaningful
  milestone, finishing a phase, or after a substantial change.
  Trigger phrases: "checkpoint", "snapshot", "wrap up the session",
  "log everything and commit".
argument-hint: "[optional one-line summary of the checkpoint]"
allowed-tools: Read Write Edit Bash(git:*) Bash(gh:*) Bash(date:*) Bash(ls:*) Grep Glob
---

# Checkpoint

Run the full end-of-milestone sequence in one shot:

1. Update `todo/PLAN.md` (mark completed tasks).
2. Document substantive changes (new public APIs, modified contracts).
3. Write a bitácora entry for today.
4. Commit with a descriptive message.
5. Ask the user about pushing and opening a PR.

This skill orchestrates `/plan-writing`, `/document`, `/bitacora`, and git/gh,
so you don't have to invoke them one by one.

## Pre-rendered context

- **Date**: !`date +%Y-%m-%d`
- **Time**: !`date +%H:%M`
- **Branch**: !`git branch --show-current 2>/dev/null || echo "(not a git repo)"`
- **Default branch**: !`git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main"`
- **Working tree status**:
```!
git status --short 2>/dev/null || echo "(not a git repo)"
```
- **Commits since last bitácora**:
```!
LATEST=$(ls -1t todo/bitacora-*.md 2>/dev/null | head -n1)
if [ -n "$LATEST" ]; then
  SINCE=$(date -r "$LATEST" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LATEST" 2>/dev/null)
  git log --since="$SINCE" --format="%h %s" 2>/dev/null
else
  git log --since="1 day ago" --format="%h %s" 2>/dev/null
fi
```
- **Diff against default branch (cumulative for PR)**:
```!
DEFAULT=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
git diff --stat "origin/$DEFAULT...HEAD" 2>/dev/null | tail -n40
```
- **Current PLAN.md (active phase)**:
```!
[ -f todo/PLAN.md ] && grep -B0 -A20 "^### Phase " todo/PLAN.md | head -n40 || echo "(no PLAN.md)"
```

## Procedure

### 1. Frame the checkpoint

If `$ARGUMENTS` is provided, use it as the headline of this checkpoint.
Otherwise, infer a one-line headline from the diff (e.g., "Implement OAuth callback handler").

State the headline to the user and confirm before proceeding (skip confirmation
if `$ARGUMENTS` is non-empty — assume the user already knows what they did).

### 2. Run verification

Per `verification.md`, do **not** create a checkpoint while verification is failing.

- Read verification commands from `project-guidelines.md` (under "Verification commands").
- Run the relevant subset for the changes (typically `lint` and `test`).
- If anything fails: stop, report, and let the user fix before continuing.
- If no verification commands are configured, say so and continue.

### 3. Update `todo/PLAN.md`

Apply the `/plan-writing` rules to update the plan:

- Mark tasks completed in this session as `- [x] (YYYY-MM-DD)` using the pre-rendered date.
- If a phase is fully done, append `(COMPLETED)` to its title.
- Promote any pending items from the latest bitácora that became real plan tasks.
- Do **not** add new tasks without user approval.

### 4. Update documentation (selective)

Apply the `/document` rules only where they apply:

- New public function/class with a non-trivial contract → ensure docstring is present.
- New module → ensure module docstring is present.
- Modified public API → update the corresponding `docs/NN_*.md` file if it exists.
- If a substantive new module deserves a fresh doc, create it under `docs/`.
- Skip docs entirely if the changes are tests, refactors with no API surface change, or trivial fixes.

State briefly which docs were updated (or "no docs needed for this checkpoint").

### 5. Write bitácora entry

Apply the `/bitacora` rules:

- Target file: `todo/bitacora-YYYY-MM-DD.md` (use pre-rendered date).
- Append a new entry with HH:MM (use pre-rendered time).
- Structure: technical changes / design decisions / pending / learnings.
- Use `$ARGUMENTS` (or the inferred headline) as the entry summary.

### 6. Commit

Per `commit-style.md`, choose the prefix from the 8-prefix set:

| Dominant nature of changes | Prefix |
|---|---|
| New functionality (came from `/phase-executor` or implementation) | `feat:` |
| Bug fix (came from `/bug-fix`, or fixes incorrect behavior) | `fix:` |
| Only `documentation/` changed (code docs) | `docs:` |
| Only `docs/` changed (GitHub Pages landing site) | `site:` |
| Only code restructure with no behavior change | `refactor:` |
| Only `tests/` changed | `test:` |
| Performance improvement with benchmark | `perf:` |
| Tooling / `.claude/` / configs / deps | `chore:` |

When changes span categories, pick the **dominant** one. When in doubt: `chore:`.

- Stage **specific files** (not `git add -A`). Choose only files relevant to this checkpoint.
- Compose a commit message:
  - Subject: `<type>(<scope>): <imperative>` ≤72 chars, no period.
  - Body: 2-4 lines on the **why**.
- Do **not** add Claude / "co-authored by" footers unless the project conventions require them.
- Do **not** use `--no-verify`. If pre-commit hooks fail, fix and re-stage.

Example structure:
```
git add <files>
git commit -m "$(cat <<'EOF'
<type>(<scope>): <subject>

<body line 1>
<body line 2>
EOF
)"
```

### 7. Push and PR (ask the user)

After the commit succeeds, ask the user:

> "Push to remote and open/update PR?"

If **yes**:

- Push: `git push` (or `git push -u origin <branch>` if no upstream).
- Check if a PR exists for this branch:
  ```bash
  gh pr view --json number 2>/dev/null
  ```
- If a PR exists: it auto-updates with the new commits. Optionally add a comment summarizing the checkpoint.
- If no PR exists: ask whether to create one. If yes, run `gh pr create` with title from headline and body summarizing the cumulative diff against the default branch.

If **no**: stop here. Report status to the user.

### 8. Final report

Summarize to the user:

- Headline of the checkpoint
- Files committed (count + key paths)
- Commit hash
- PLAN.md tasks marked complete
- Docs touched (or "none needed")
- Bitácora entry path
- Push/PR status

## Rules

- **Verification first**: never checkpoint over a failing build.
- **Specific staging**: never `git add -A` or `git add .` — pick files explicitly.
- **No `--no-verify`**: if pre-commit hooks fail, fix the underlying issue.
- **Confirm before push/PR**: these are external actions, always ask.
- **Skip what doesn't apply**: if there's nothing to document, say so. If PLAN.md is up to date, leave it alone.
- **One bitácora entry per checkpoint**: even if multiple commits get bundled, the bitácora gets one entry summarizing them.
- **Idempotent on re-run**: if the user runs `/checkpoint` twice in a row with no new changes, it should detect that and stop early.

## When to use vs not use

| Use `/checkpoint` when... | Use individual skills instead when... |
|---|---|
| You finished a phase or milestone | You only want to log the session (just `/bitacora`) |
| You made a substantive change worth committing | You're mid-work and not ready to commit |
| You want plan + docs + bitácora + commit in one go | You want to write the plan from scratch (`/plan-writing`) |
| You're about to take a break and want a clean stop | You're documenting an unrelated module (`/document`) |
