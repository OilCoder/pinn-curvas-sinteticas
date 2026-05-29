<!-- description: Decide whether work runs in main context, a subagent, or an agent team -->

# Delegation

When facing a task, decide **where** it runs before deciding **how** to do it.
The wrong layer wastes context (too local) or tokens (too distributed).

## Decision matrix

| Signal | Where it runs | Why |
|---|---|---|
| Self-contained, output is short, fits in current context | **Main session** | No delegation overhead |
| Side-research that reads many files | **Subagent** (`Explore` or `general-purpose`) | Keeps your main context clean |
| Specialized review (correctness, security, architecture) | **Subagent** — `code-reviewer`, `security-reviewer`, `architect` | Fresh-context bias matters; one specialist suffices |
| Implementation of a clearly specified function/module | **Subagent** — `implementer` | Rules preloaded, returns verified code |
| Bug fix with regression preserved | **Skill** — `/bug-fix` | TDD workflow: reproduce, fail, fix, confirm |
| Refactor without behavior change | **Skill** — `/simplify` (bundled) | Already provided by Claude Code, no need to recreate |
| **Multiple specialists must discuss findings before answer is final** | **Agent Team** (opt-in, experimental) | Inter-agent messaging adds value |
| **Adversarial investigation (hypotheses competing to disprove each other)** | **Agent Team** | Debate beats sequential exploration |
| **Parallel work on independent files with handoffs between owners** | **Agent Team** | Shared task list + self-claim |

## Heuristic

Ask: **"Do the workers need to talk to each other to reach a better answer?"**

- **No** → subagent (one or several, all reporting to main).
- **Yes** → consider Agent Team. But only if the answer's quality clearly justifies 3-5× token cost.

When in doubt, start with subagents. Escalate to a team only when a subagent attempt
showed that one perspective was insufficient.

## Cost awareness

- Main session: cheapest, but consumes your context window.
- Subagent: moderate cost, isolated context, summarized result returns.
- Agent Team: highest cost (each teammate is a full Claude session). Token cost
  scales linearly with team size.

A team of 5 reviewing a small PR is almost always wrong. A team of 3 debating
hypotheses on a hard production bug is often right.

## Agent Teams are experimental

Per the official Claude Code docs, agent teams are **disabled by default** and
require `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Treat them as opt-in for
specific moments, not as a default workflow.

The four agents in this base (`code-reviewer`, `security-reviewer`, `architect`,
`implementer`) work as both subagents (default) and as teammates inside a team.
No new agents are needed to use teams — just spawn them by name.

> Note: when an agent runs as a teammate, its `skills:` and `mcpServers:`
> frontmatter is **not** applied (per official docs). Plan accordingly if the
> agent depends on preloaded skills.

## Cross-references

- See `verification.md` for the gate that applies regardless of where work runs.
- See `agents/*.md` for the available specialist subagents.
- See `code-change.md` for scope discipline when work returns from delegation.
