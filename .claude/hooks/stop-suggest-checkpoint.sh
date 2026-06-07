#!/usr/bin/env bash
# Stop hook: when Claude finishes its turn, check whether a checkpoint is due.
# A checkpoint is due if there are uncommitted changes OR commits since
# the latest bitácora entry.
#
# Robustness:
#   - 5s overall timeout
#   - Logs failures to .claude/hooks.log
#   - Exits 0 on any error (never blocks Stop)

set -u

LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks.log"
HOOK_NAME="stop-suggest-checkpoint"

log_err() {
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
  printf '[%s] %s: %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" "$HOOK_NAME" "$1" >> "$LOG" 2>/dev/null || true
}

trap 'log_err "unexpected error at line $LINENO"; exit 0' ERR

# Skip if not a git repo
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

RESULT=$(
  timeout 5 bash -c '
    DIRTY=$(git status --porcelain 2>/dev/null || true)
    LATEST=$(ls -1t todo/bitacora-*.md 2>/dev/null | head -n1 || true)

    NEEDS=0
    REASONS=""

    if [[ -n "$DIRTY" ]]; then
      NEEDS=1
      REASONS="${REASONS}- uncommitted changes in working tree\n"
    fi

    if [[ -n "$LATEST" ]]; then
      MTIME=$(date -r "$LATEST" "+%s" 2>/dev/null || stat -f "%m" "$LATEST" 2>/dev/null || stat -c "%Y" "$LATEST" 2>/dev/null || echo 0)
      if [[ "$MTIME" -gt 0 ]]; then
        SINCE=$(date -d "@$MTIME" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -r "$MTIME" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "")
        if [[ -n "$SINCE" ]]; then
          NEW_COMMITS=$(git log --since="$SINCE" --oneline 2>/dev/null | wc -l | tr -d " ")
          if [[ "$NEW_COMMITS" -gt 0 ]]; then
            NEEDS=1
            REASONS="${REASONS}- ${NEW_COMMITS} commit(s) since last bitácora\n"
          fi
        fi
      fi
    fi

    if [[ "$NEEDS" -eq 1 ]]; then
      printf "%s" "$REASONS"
    fi
  ' 2>/dev/null
) || {
  log_err "main subshell failed or timed out"
  exit 0
}

[[ -z "$RESULT" ]] && exit 0

MSG="Checkpoint suggestion: this session has unrecorded work.\\n${RESULT}Run /checkpoint to update plan, document, log session, and commit."
ESCAPED=$(printf '%s' "$MSG" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' ')

printf '{"hookSpecificOutput":{"hookEventName":"Stop","additionalContext":"%s"}}' "$ESCAPED" || {
  log_err "failed to emit JSON output"
  exit 0
}

exit 0
