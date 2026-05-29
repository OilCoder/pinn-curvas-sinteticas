---
name: test
description: >
  Create a test script for a module or function.
  Use when the user says "test this", "create a test for X",
  "I need tests for this module".
argument-hint: "[module or function to test]"
allowed-tools: Read Write Bash(mkdir:*) Bash(ls:*) Grep Glob
paths:
  - "tests/**/*"
  - "**/test_*.{py,js,ts,jsx,tsx}"
  - "**/*_test.{py,go,js,ts}"
  - "**/*.test.{js,ts,jsx,tsx}"
  - "**/*.spec.{js,ts,jsx,tsx}"
---

# Test

Generate a focused test script for a specific module or function.

## Procedure

### 1. Identify the target

- If the user passed `$ARGUMENTS`, use it as the target.
- If not, ask which module or function to test.
- Read the source file to understand the interface (parameters, returns, side effects).

### 2. Create the test file

- Location: `tests/`
- Name: `test_<module>_<case>.<ext>` (see `file-naming.md` rule)
- If `tests/` does not exist, create it.

### 3. Test structure

```python
"""
Tests for <module_name>
"""

import pytest
# required imports


class TestModuleName:
    """Tests for <function or class>."""

    def test_<method>_<case>(self):
        """<What this test verifies>."""
        # Arrange
        # Act
        # Assert

    def test_<method>_<edge_case>(self):
        """<Edge case>."""
        # ...
```

## Rules

- Each test file targets a single module or class.
- Function names: `test_<method>_<case>()`.
- Use clear assert statements with descriptive messages.
- Tests must be independent of each other — no shared state.
- Execution order should not matter. Numeric prefix is for sorting only.
- Prefer fixtures over manual setup/teardown.
- Keep tests simple, readable, and focused on one behavior per test.
- Use `@pytest.mark.<tag>` to group tests (e.g., `integration`, `gpu`, `slow`).

## Cache and artifacts

- Do not commit `__pycache__/` or test artifacts.
- Test fixtures with temporary files should use temp directories (`tmpdir`, `tmp_path`).
- Large test data files go to `tests/fixtures/` (gitignored if heavy).
- `tests/` may be gitignored depending on the project.

## Git policy

- Defined per project: some commit tests, others gitignore them.
- If committed, tests must pass before each commit.
- If gitignored, keep tests updated locally for regressions.
