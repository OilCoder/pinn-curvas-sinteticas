#!/usr/bin/env bash
# PostToolUse hook: warns when a file under src/ | lib/ | app/ | pipeline/
# imports from debug/. Reinforces logging-policy.md.
#
# Robustness:
#   - 3s overall timeout
#   - Logs failures to .claude/hooks.log
#   - Exits 0 on any error (never blocks the tool call result)

set -u

LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks.log"
HOOK_NAME="check-debug-isolation"

log_err() {
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
  printf '[%s] %s: %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" "$HOOK_NAME" "$1" >> "$LOG" 2>/dev/null || true
}

trap 'log_err "unexpected error at line $LINENO"; exit 0' ERR

INPUT=$(timeout 1 cat 2>/dev/null) || {
  log_err "stdin read failed"
  exit 0
}

FILE_PATH=$(printf '%s' "$INPUT" \
  | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' 2>/dev/null \
  | sed 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/' \
  | head -n1)

[[ -z "$FILE_PATH" ]] && exit 0

# Only check production source folders
if [[ ! "$FILE_PATH" =~ /(src|lib|app|pipeline)/ ]]; then
  exit 0
fi

[[ ! -f "$FILE_PATH" ]] && exit 0

VIOLATION=$(
  timeout 3 grep -lE '(from[[:space:]]+debug[[:space:]\.]|import[[:space:]]+debug[[:space:]\.;]|from[[:space:]]+["'"'"'][^"'"'"']*debug|require\([^)]*debug)' "$FILE_PATH" 2>/dev/null
) || {
  log_err "grep failed on $FILE_PATH"
  exit 0
}

[[ -z "$VIOLATION" ]] && exit 0

cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "WARNING: $FILE_PATH appears to import from debug/. Per logging-policy.md, debug scripts must never be imported by production code. Move the logic out of debug/ or remove the import."
  }
}
EOF

exit 0
