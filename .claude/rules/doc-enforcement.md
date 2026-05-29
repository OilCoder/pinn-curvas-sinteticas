---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
---

<!-- description: Enforce standardized docstrings and module-level documentation. Adjust paths in /setup per stack. -->

# Doc Enforcement

All public functions, methods, and classes must include a docstring.
This rule applies passively to all code Claude generates or modifies.

## Docstring required

- All public functions, methods, and classes must include a docstring.
- Private functions (starting with `_`) require a docstring only if they contain nontrivial logic.
- Docstring format: Google Style by default. Configurable per project (NumPy, JSDoc, etc.).

## Module header

- Every source file must begin with a top-level module docstring.
- Content: concise summary of the module's purpose (1–3 lines).
- May include an optional bullet list of major functions or classes.
- May include optional usage context (e.g., "Called by `main_pipeline.py`").
- Keep it under 100 words.

## Docstring structure

Must include (if applicable):

- One-line summary describing behavior (imperative mood).
- `Args:` section with parameter names, types, and descriptions.
- `Returns:` section with return type and explanation.
- `Raises:` section for explicitly raised exceptions.

## Style rules

- Must match the configured format for the project.
- Do not mix formats within a project.
- Avoid vague terms like "does something", "helper function", "processes data".
- If a function has no parameters or return value, still explain what it does.

## Consistency

- Docstrings must reflect actual behavior, not intentions or placeholders.
- Do not duplicate content across sections.
- Keep docstrings concise, specific, and informative.
- When modifying a function, update its docstring if the behavior changed.

## Enforcement scope

- Defined per project (e.g., "all files under `src/`", "`pipeline/` and `config.py`").
- Functions without docstrings may be excluded from generated documentation.

## Cross-references

- See `doc-enforce/SKILL.md` for the on-demand review and enforcement workflow.
- See `docs-style.md` for Markdown documentation standards.
