<!-- description: Control scope and safety of code modifications -->

# Code Change

## Edit scope

- Only modify the specific code the user requested.
- If the user does not pinpoint the exact location, inspect the codebase and find the minimal block to change.
- Never refactor, rename, or restructure code outside the explicit request scope.
- Preserve existing formatting, comments, and structure unless asked to change them.

## Before editing

- Read and understand the surrounding context before making changes.
- Identify dependencies — will this change break something else?
- If unsure about scope, ask before editing.

## Structural integrity

- Preserve existing order of imports and function declarations.
- Preserve existing blank lines and section separators.
- Do not reformat code that was not part of the request.
- If the file uses Step/Substep comment structure, maintain it.

## Multi-file changes

- When a change spans multiple files, list all affected files before editing.
- Apply changes in dependency order: utilities first, callers last.
- Each file edit must be self-contained — if the edit fails mid-way, remaining files should still work.
- Never assume a file's content — always read it first.

## Output format

- Return only the modified function, class, or block.
- Do not show or regenerate unchanged code.
- Show the full file only if explicitly asked.
- When multiple functions change in one file, show each modified function separately.

## Comments

- Only modify or add comments tied to the changed logic.
- Do not rephrase unrelated documentation.
- If the change removes logic, remove its associated comments too.

## Forbidden

- Do not leave debug code, test code, or print statements in production files.
- Do not introduce new dependencies without explicit approval.
- Do not change function signatures unless that is the explicit goal.
- Do not silently rename variables outside the edit scope.

## After editing

- Verify the modified code is syntactically correct.
- If a test or debug skill exists, suggest running it on the changed code.
- If the change affects a public API, flag that docstrings may need updating.

## Cross-references

- See `test/SKILL.md` for writing tests on changed code.
- See `investigate/SKILL.md` for isolating investigation scripts.
