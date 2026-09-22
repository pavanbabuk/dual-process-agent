# Parallel workstream plan — why 4 now, 2 later

Baseline commit: `1840ad3` (247 tests passing, screen suite repaired).

## Wave 1 — running in parallel (4 agents)

| Agent | Owns (write) | Why it can run concurrently |
|---|---|---|
| **WS1 — Real MCP client** | `src/dual_agent/mcp_manager.py`, `tests/test_mcp_manager.py` | Isolated file; nothing else imports its internals |
| **WS2 — Multi-bot identity** | NEW `src/dual_agent/bots.py`, `tests/test_multi_bot.py` | Net-new files; reads `team_manifest.py` but does not edit it |
| **WS3 — Sessions + scheduler** | `web/server.py`, `scheduler.py`, `dashboard.html`, `tests/test_web_api.py`, `tests/test_scheduler_tick.py` | UI/wiring layer only |
| **WS4 — Vision provider** | `system_two.py`, `screen.py`, `config.py`, `tests/test_vision_provider.py`, `tests/test_screen_control.py` | Provider interface layer only |

Every agent was given: the 1840ad3 baseline, the existing-file list it must NOT touch, the
test interpreter (`/tmp/da311/bin/python`), the no-fabricated-numbers rule, the
mocks-never-satisfy-acceptance rule, the DUAL_AGENT_HOME trap, and an explicit instruction
not to `git commit` or `git push` (the parent integrates).

## Wave 2 — queued, blocked on Wave 1 (do NOT start in parallel)

### WS5 — Composio integration — blocked on WS1

**Same file as WS1** (`mcp_manager.py`), and architecturally dependent: Composio should be
consumed as an **MCP server**, so a real MCP client must exist first. Built now, it would
be wired into a registry whose placeholder returns
`f"Dispatched to external MCP server '{n}': {args}"` — producing a Composio integration that
reports success without ever calling Gmail. Two broken integration paths instead of one.

Requirements when it runs:
- Composio over MCP, not a parallel subsystem.
- OAuth: browser flow + token storage with 0600 perms under the data dir.
- Credentials never logged, never committed.
- A real-app acceptance test only if a key is supplied; otherwise assert the honest
  "not configured" path and say so. Do not fake a Gmail call.

### WS6 — Screen agent loop — blocked on WS4

Vision must exist before the loop can observe. Requires:
- Reuse the existing dispatcher; do not write a second agent loop.
- Verify via `screen_diff`, never via the model's own claim.
- Stall guard must cover screen actions (the file-reading stall already proved this matters).
- Kill switch + per-action risk thresholds (a click that buys/overwrites needs a higher bar).
- Acceptance on a **verifiable** task (TextEdit → type → save → read the file back).
  Games are excluded: no machine-readable ground truth, so a pass cannot be distinguished
  from a lucky run.

## Integration checklist (parent, after each wave)

1. `git status` — confirm each agent touched only its owned files.
2. Full suite: `/tmp/da311/bin/python -m pytest -q --no-header -p no:cacheprovider`
   (must be ≥247 and green).
3. Secret scan the staged diff.
4. Confirm no test wrote to `~/.dual_agent/memory.db`
   (`stat -f "%m %z" ~/.dual_agent/memory.db` unchanged by a full run).
5. Verify each agent's headline claim against its OWN artifact, not its summary —
   a child claiming success is a self-report, not evidence.
6. Commit + push, then release Wave 2.
