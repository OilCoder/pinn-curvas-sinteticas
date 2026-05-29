---
name: investigate
description: >
  Create an isolated debug script to investigate a problem.
  Use when the user says "debug this", "investigate this error",
  "I don't understand why X fails", "trace", "exploratory script".
argument-hint: "[module or problem to investigate]"
allowed-tools: Read Write Bash(mkdir:*) Bash(ls:*) Grep Glob
---

# Investigate

Generate an isolated debug script to investigate a specific problem.
The script lives in `debug/` and never modifies production code.

## Procedure

### 1. Identify the problem

- If the user passed `$ARGUMENTS`, use it as the starting point.
- Read the affected module and understand the flow.
- Identify the probable point of failure.

### 2. Create the debug script

- Location: `debug/`
- Name: `dbg_<slug>[_<experiment>].<ext>` (see `file-naming.md` rule)
- If `debug/` does not exist, create it.
- Verify `debug/` is in `.gitignore`. If not, add it.

### 3. Script structure

```python
"""
Debug: <problem description>
Target: <module or function under investigation>
"""

# ----------------------------------------
# Step 1 — Reproduce the problem
# ----------------------------------------
# Load minimum data/state to reproduce

# ----------------------------------------
# Step 2 — Inspect intermediate values
# ----------------------------------------
# Print/log key variables at each step

# ----------------------------------------
# Step 3 — Hypothesis and verification
# ----------------------------------------
# Test the hypothesis about the root cause
```

## Rules

- Each debug script targets a specific module; keep that link explicit in the name.
- The script must be self-contained — executable without modifying source code.
- Use descriptive prints to trace the flow.
- Add inline comments to document findings and dead ends.
- If a bug is discovered, suggest creating a test with the `/test` skill.

## Clean code principle

- Temporary debug code (prints, flags, conditionals) may be added to production files **only during active debugging**.
- Before commit: all debug code must be removed from production files.
- Final production code must be clean, minimal, and production-ready.
- All deep debugging and verbose outputs must live exclusively in `debug/` scripts.

## Isolation and artifacts

- Debug scripts must never be imported by production code.
- Large files or outputs created during debugging go to `debug/.cache/` (also gitignored).
- All debug scripts live in `debug/` which must always be in `.gitignore`.

## Promotion path

If a debug script reveals a real bug worth preserving as a regression check:

1. Create a test with the `/test` skill that captures the bug.
2. Document the fix in the commit message.
3. Remove or archive the original debug script after resolving the bug.
