---
name: doc-enforce
description: >
  Review and generate docstrings for functions and modules.
  Use when the user says "review the docstrings", "add inline documentation",
  "enforce docs on this file", "missing docstrings".
argument-hint: "[file to review]"
allowed-tools: Read Edit Grep Glob
context: fork
agent: general-purpose
paths:
  - "src/**/*.{py,js,ts,jsx,tsx,go,rs,java,rb,php}"
  - "lib/**/*.{py,js,ts,jsx,tsx,go,rs,java,rb,php}"
  - "app/**/*.{py,js,ts,jsx,tsx,go,rs,java,rb,php}"
  - "pipeline/**/*.{py,js,ts,jsx,tsx,go,rs,java,rb,php}"
---

# Doc Enforce

Review a file and add or fix missing docstrings.
Read the `doc-enforcement.md` rule for the standards this skill enforces.

## Procedure

### 1. Scan the file

- If the user passed `$ARGUMENTS`, use it as the target.
- Read the complete file.
- Identify all functions, classes, and methods.
- List which ones have a docstring and which do not.

### 2. Report status

Show the user:

```
File: <path>
- func_a() ✅ has docstring
- func_b() ❌ missing docstring
- ClassC ✅ has docstring
  - method_d() ❌ missing docstring
```

### 3. Generate missing docstrings

For each function/class without a docstring, generate one using the project's configured format (Google Style by default):

```python
def function_name(param1: type, param2: type) -> return_type:
    """One-line summary of what this function does.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When something is invalid.
    """
```

### 4. Module header

If the file has no module docstring at the top, add one:

```python
"""
Short description of the module's purpose (1-3 lines).

Major functions/classes: list_them_here
"""
```

## Rules

- Docstrings must reflect actual behavior, not intentions.
- Keep under 100 words per docstring.
- Do not mix formats (respect the project's configured format).
- Avoid vague terms like "does something" or "helper function".
- Private functions (`_func`) only need a docstring if they contain nontrivial logic.
- Do not duplicate content across docstring sections.

## Enforcement scope

- The enforcement scope is defined in the `doc-enforcement.md` rule.
- If not defined, apply to all source code files.
- Functions without docstrings may be excluded from generated documentation.

## Consistency check

- Verify existing docstrings match current behavior.
- If a function changed but its docstring did not, flag it to the user.
- Do not modify existing docstrings without approval — only report inconsistencies.
