<!-- description: Control use of print and logging across all code -->

# Logging Policy

## Print usage

- Temporary `print()` statements are allowed during development.
- Log/print message language: defined per project (English by default).
- All print calls must be removed before final commits unless:
  - They are part of CLI tools or user-facing interfaces.
  - They appear in notebooks used for demos or traceability.
  - They serve a critical runtime function (progress indicators, etc.).

## Logging usage

- Use the language's standard logging module for structured output in production.
- Avoid debug-level logging unless the logger is properly configured and output can be filtered.
- Use module-scoped loggers (e.g., `logger = logging.getLogger(__name__)` in Python).

## Progress output

- Long-running processes should show visible progress (`tqdm`, progress bars, etc.).
- Log errors per item without stopping the full batch when appropriate.
- A final summary line (total processed, skipped, failed) is recommended for batch operations.
- Specific progress tools and patterns: defined per project.

## Cleanup and isolation

- Extensive debug output must be isolated into scripts inside `debug/`.
- Debug code and verbose logs must be excluded from main modules.
- Final production code must not include leftover print or logging unless justified by scope.
- Treat print/log statements as disposable scaffolding unless approved by the project owner.
- Do not add `logging.debug()` noise to production modules unless the logger is properly configured.

## Exceptions

- In notebooks, print/logging is allowed for readability, validation, or user interaction.
- CLI scripts may retain structured logging or print if it improves UX or runtime feedback.
