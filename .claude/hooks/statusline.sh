#!/usr/bin/env bash
# Statusline hook: renders the prompt status line on every Claude turn.
# Output format: <branch><dirty> | <active phase> | <bitácora flag>
#
# Robustness:
#   - 2s overall timeout (must be fast — runs on every turn)
#   - Logs failures to .claude/hooks.log
#   - On error or timeout, prints "(no git)" so the status line never breaks UX

set -u

LOG="${CLAUDE_PROJECT_DIR:-.}/.claude/hooks.log"
HOOK_NAME="statusline"

log_err() {
  mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
  printf '[%s] %s: %s\n' "$(date -u +%FT%TZ 2>/dev/null || echo unknown)" "$HOOK_NAME" "$1" >> "$LOG" 2>/dev/null || true
}

trap 'log_err "unexpected error at line $LINENO"; printf "(no git)\n"; exit 0' ERR

OUT=$(
  timeout 2 bash -c '
    BRANCH=$(git branch --show-current 2>/dev/null || echo "")

    DIRTY=""
    PORCELAIN=""
    if [[ -n "$BRANCH" ]]; then
      PORCELAIN=$(git status --porcelain 2>/dev/null | head -c1)
      [[ -n "$PORCELAIN" ]] && DIRTY="*"
    fi

    PHASE=""
    if [[ -f todo/PLAN.md ]]; then
      PHASE=$(grep -E "^### Phase " todo/PLAN.md 2>/dev/null | grep -v "(COMPLETED)" | head -n1 | sed "s/^### //; s/ —.*//" | tr -d "\n")
    fi

    BITACORA_FLAG=""
    TODAY=$(date +%Y-%m-%d 2>/dev/null)
    if [[ -n "$BRANCH" && -n "$TODAY" ]]; then
      if [[ ! -f "todo/bitacora-${TODAY}.md" && -n "$PORCELAIN" ]]; then
        BITACORA_FLAG=" | bitácora pendiente"
      fi
    fi

    PARTS=""
    [[ -n "$BRANCH" ]] && PARTS="${BRANCH}${DIRTY}"
    [[ -n "$PHASE" ]]  && PARTS="${PARTS} | ${PHASE}"
    PARTS="${PARTS}${BITACORA_FLAG}"

    [[ -z "$PARTS" ]] && PARTS="(no git)"

    printf "%s\n" "$PARTS"
  ' 2>/dev/null
) || {
  log_err "main subshell failed or timed out"
  printf '(no git)\n'
  exit 0
}

printf '%s\n' "${OUT:-(no git)}"
exit 0
