<!-- description: Enforce consistent file naming across the project -->

# File Naming

All files must follow consistent and descriptive naming conventions
for discoverability, clarity, and traceability across the codebase.

## General conventions

- Use `snake_case` for all files: lowercase letters with underscores.
- Filenames must reflect the purpose or main component of the file.
- Avoid generic names like `script.py`, `data.json`, `utils.py` unless scoped in clearly named folders.
- No spaces, accented characters, or mixed camelCase/snake_case.
- File name language: defined per project (English by default).

## Execution order naming

- Pipeline or sequential scripts must carry a numeric prefix to indicate execution order.
- Default pattern: `NN_<descriptive_name>.<ext>` (e.g., `01_fetch_data.py`, `02_process.py`).
- Alternative pattern with sub-steps: `sNN[x]_<verb>_<noun>.<ext>` (e.g., `s01a_setup_grid.m`).
  Choose one pattern per project and apply it consistently.
- Orchestrators or main launchers use a high prefix (e.g., `99_run_pipeline.py`) to appear last in listings.
- The numeric prefix is for natural sorting only — it must not create dependencies.

## Output files

- Pattern for generated outputs: defined per project.
- Examples: `{id}_{qualifier}.png`, `result_<experiment>.csv`.
- Output files live in a dedicated folder (`outputs/`, `results/`, etc.) that may be gitignored.

## Data files

- Pattern: `<descriptive_name>.<ext>`
- Input data lives in a dedicated folder (`data/`, `input/`, etc.).
- Configuration files live in the project root or a `config/` folder.

## Test files

- Pattern: `test_<module>_<case>.<ext>`
- All test files live under `tests/` folder.
- `tests/` may be gitignored depending on the project.

## Debug files

- Pattern: `dbg_<slug>[_<experiment>].<ext>`
- All debug files live under `debug/` folder.
- `debug/` must always be in `.gitignore`.

## Documentation

- Pattern: `NN_<slug>.md` for ordered docs.
- Location defined per project (`docs/`, `obsidian-vault/`, etc.).
- See `docs-style.md` for content standards.
