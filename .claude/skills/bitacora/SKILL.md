---
name: bitacora
description: >
  Log the current work session in the project's session log.
  Use after each commit or push, or when the user says
  "log the session", "record what we did", "bitacora", "session log".
argument-hint: "[optional summary message]"
allowed-tools: Read Write Edit Bash(git log:*) Bash(git diff:*) Bash(git status:*) Bash(git branch:*) Bash(date:*) Grep Glob
---

# Bitacora (Session Log)

Record the work done in the current session as a log entry.
Target file: `todo/bitacora-YYYY-MM-DD.md` inside the project.

## Pre-rendered session context

The following data is captured at skill invocation time. Use it instead of running
the commands again.

- **Date**: !`date +%Y-%m-%d`
- **Time**: !`date +%H:%M`
- **Branch**: !`git branch --show-current 2>/dev/null || echo "(not a git repo)"`
- **Today's commits**:
```!
git log --since="00:00" --format="%h %s" 2>/dev/null || echo "(no commits today)"
```
- **Files changed today**:
```!
git diff --stat HEAD~$(git log --oneline --since="00:00" 2>/dev/null | wc -l) HEAD 2>/dev/null || git diff --stat HEAD~1 HEAD 2>/dev/null || echo "(no diff available)"
```
- **Working tree status**: !`git status --short 2>/dev/null || echo "(clean)"`

## When to run

- After each `git commit` or `git push`
- When the user explicitly requests it
- At the end of a long work session
- As part of `/checkpoint` (combined workflow)

## Procedure

### 1. Check if today's bitácora file exists

- If `todo/bitacora-YYYY-MM-DD.md` exists, **append** a new section at the end (do not overwrite).
- If it does not exist, **create** the file with the full template.
- If `todo/` does not exist, create it.

### 2. Write the entry

Use the pre-rendered context above to fill in commits, files, and branch.
Each entry follows this structure:

```markdown
## HH:MM — [short summary of what was done]

### Technical changes
- modified_file.py — what changed and why
- new_file.py — what it does, why it was created

### Design decisions
- [Decision made] — [reason / alternatives discarded]

### Errors
- [What I tried / assumed that was wrong] — [how I noticed, what the actual right answer was]

### Pending
- [ ] Task left undone
- [ ] Something to review later

### Learnings
- [Something new discovered during the session]
```

### Errors vs Learnings — the boundary

These two sections are different and complementary:

- **Errors** are **specific incidents in this session**. First person, factual:
  *"Asumí que polars soportaba MultiIndex; perdí 1h adaptando el load. No lo soporta — usa structs. Volví a pandas."*

- **Learnings** are the **generalized insight**, transferable to future work:
  *"Para series temporales paralelas (pozos), MultiIndex sigue siendo más natural que structs."*

A single error often produces a learning. The error is the **what happened**;
the learning is the **what to remember next time**. Both go in the bitácora —
the error is the proof, the learning is the lesson.

Empty sections are OK. If no error occurred this session, omit the section. If
nothing generalizable came out, omit Learnings. Don't fabricate.

## Rules

- **Language**: Spanish for prose, English for code and file names (configurable per project).
- **Commits**: list short hashes and messages from the day (already pre-rendered above).
- **No fabrication**: only record what actually happened in the session.
- **Accumulate**: a single day can have multiple entries (one per session/commit).
- **Pending items**: mark with `- [ ]` so Cowork and `/plan-writing` can detect them.
- **Brevity**: each section should be concise, not an essay.
- If the user passes a message as argument (`$ARGUMENTS`), use it as the entry summary.

## New file template

```markdown
# Session Log — {readable date}

**Project**: {project name}
**Branch**: {current branch}

---

## HH:MM — {summary}

### Technical changes
- {file} — {description}

### Design decisions
- {decision} — {reason}

### Errors
- {what was wrong} — {how it was noticed and corrected}

### Pending
- [ ] {task}

### Learnings
- {learning}

---
*Day's commits*: {list of hashes}
```

## Cowork integration

This file is read by Cowork to generate the `log-YYYY-MM-DD.md` note in the Obsidian vault.
Cowork enriches the log with wikilinks, Obsidian frontmatter, and connections to projects and daily plans.
