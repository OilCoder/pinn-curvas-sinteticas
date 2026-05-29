<!-- description: Enforce layout, naming, spacing, and step/substep structure in source files -->

# Code Style

All generated code must prioritize clarity, simplicity, and directness.
The goal is maintainable, focused code with no unnecessary complexity.

## Function structure

- Every function must have a single, well-defined responsibility.
- Avoid mixing unrelated logic inside the same function.
- Function bodies should be as short as reasonably possible.
- Use helper functions if a task has multiple logical steps.
- Concrete length limits should be enforced by a linter (e.g., `ruff` rule `C901`, `eslint` `max-lines-per-function`), not by this rule.

## Minimalism

- Only generate what is strictly necessary to fulfill the request.
- Avoid boilerplate, placeholder code, or speculative structures unless explicitly requested.
- Do not write future-proof abstractions unless the user explicitly asks for scalability.

## Naming

- Variable and function names must be self-explanatory and follow `snake_case`.
- Use short, meaningful names. Avoid generic or placeholder names (`temp`, `foo`, `data1`, `result`).
- Constants use `UPPER_SNAKE_CASE`.
- Adapt naming to the project language if configured (`snake_case` for Python/Octave, `camelCase` for JS/TS, etc.).

## Comments and visual structure

- **Comment language**: English (code comments, docstrings). Spanish for bitácora and PLAN.md.
- Use comments only where logic is not self-evident.
- For functions involving multiple stages, organize them with this visual structure:

```python
# ----------------------------------------
# Step N — <High-level action>
# ----------------------------------------

# Substep N.M — <Specific sub-action>
```

- For inline actions or clarifications inside a substep, use emoji markers:

```python
# ✅ Validate inputs
# 🔄 Loop through each item
# 💾 Save results to disk
# 📊 Plot output (if enabled)
```

- This structure improves readability and helps locate logic blocks during debugging.
- Avoid inline comments for trivial code lines; describe the logic block at a higher level.
- Do not generate excessive documentation blocks unless explicitly requested.

## Imports and dependencies

- Only import what is actually used in the generated code.
- Group imports logically: standard → external → internal.
- Avoid unnecessary third-party dependencies unless already in the project.

## Output format

- Code must be presented in clean, executable blocks.
- Do not include explanatory text in code output unless requested.
- When writing multiple functions, separate them with clear spacing.

## Scope discipline

- Never write more than what the request scope defines.
- Avoid solving problems that were not asked or predicted unless clarification is provided.

## Cross-references

- See `logging-policy.md` for all print/log guidance.
- See `file-naming.md` for naming files.
- See `doc-enforcement.md` for docstring standards.
