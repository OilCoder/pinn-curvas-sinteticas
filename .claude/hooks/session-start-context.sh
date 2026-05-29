#!/usr/bin/env bash
# SessionStart hook: injects project state into Claude's context at the start
# of every session. Reduces the need for Claude to discover state turn by turn.
#
# Outputs JSON with additionalContext containing:
#   - Active phase from PLAN.md (first non-completed phase)
#   - Pending items from latest bitácora
#   - Verification commands from project-guidelines.md
#
# Robustness:
#   - 10s overall timeout (kill if anything stalls)
#   - Logs failures to .claude/hooks.log instead of failing silently
#   - Exits 0 on any error (never blocks session start)

set -u

LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks.log"
HOOK_NAME="session-start-context"

log_err() {
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
  printf '[%s] %s: %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" "$HOOK_NAME" "$1" >> "$LOG" 2>/dev/null || true
}

# Catch any error and exit cleanly
trap 'log_err "unexpected error at line $LINENO"; exit 0' ERR

# Run main work in a subshell with a soft timeout (10s)
CTX=$(
  timeout 10 bash -c '
    CTX=""

    # Active phase from PLAN.md
    if [[ -f todo/PLAN.md ]]; then
      ACTIVE_PHASE=$(grep -E "^### Phase " todo/PLAN.md 2>/dev/null | grep -v "(COMPLETED)" | head -n1 || true)
      [[ -n "$ACTIVE_PHASE" ]] && CTX="${CTX}Active phase from todo/PLAN.md: ${ACTIVE_PHASE}"$'"'"'\n\n'"'"'
    fi

    # Pending items from latest bitácora
    LATEST=$(ls -1t todo/bitacora-*.md 2>/dev/null | head -n1 || true)
    if [[ -n "$LATEST" ]]; then
      PENDING=$(grep -E "^- \[ \]" "$LATEST" 2>/dev/null | head -n5 || true)
      if [[ -n "$PENDING" ]]; then
        CTX="${CTX}Pending from $(basename "$LATEST"):"$'"'"'\n'"'"'"${PENDING}"$'"'"'\n\n'"'"'
      fi
    fi

    # Verification commands from project-guidelines.md
    if [[ -f .claude/rules/project-guidelines.md ]]; then
      VERIF=$(sed -n "/## Verification commands/,/^## /p" .claude/rules/project-guidelines.md 2>/dev/null | grep -E "^(test|type-check|lint|format):" || true)
      if [[ -n "$VERIF" ]]; then
        CTX="${CTX}Verification commands:"$'"'"'\n'"'"'"${VERIF}"$'"'"'\n\n'"'"'
      fi
    fi

    printf "%s" "$CTX"
  ' 2>/dev/null
) || {
  log_err "main subshell failed or timed out"
  exit 0
}

# Nothing to inject — exit silently
[[ -z "$CTX" ]] && exit 0

# Emit additionalContext as JSON (escape backslashes, quotes, and newlines)
ESCAPED=$(printf '%s' "$CTX" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk 'BEGIN{ORS="\\n"} {print}')

printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}' "$ESCAPED" || {
  log_err "failed to emit JSON output"
  exit 0
}

exit 0
