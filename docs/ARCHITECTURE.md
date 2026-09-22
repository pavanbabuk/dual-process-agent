# Architecture

How the Dual-Process Agent runtime works **today**, read off the source in this repo. Every
claim below cites a file and a function so you can verify it. Nothing here is aspirational —
if something is only planned, it is not in this document.

Last verified against commit state at v2.0.0 (`pyproject.toml`).

---

## 1. The core idea

Two layers, split by how they decide:

| Layer | What it is | Where it lives | Cost per decision |
|---|---|---|---|
| **System 1** | A reflex router. Given a compact state string and a dict of tool names, returns *one choice* plus confidence, whether the task is finished, and whether generation is needed. | `src/dual_agent/typesafe_client.py` → `JevSystemOneClient.evaluate_state_and_route` | Jev bills input tokens only; output is free (see ROADMAP) |
| **System 2** | A deliberate generative reasoner. Only invoked when System 1 escalates or when the fast path refuses its own arguments. | `src/dual_agent/system_two.py` → `SystemTwoProvider.generate_step` | Provider-dependent; counted as `tokens_used` on the step |

System 1 never produces text. It produces a decision. That is the whole architectural
distinction: routing and termination checks cost zero generated tokens, so a step System 1
handles does not pay for an LLM round-trip.

---

## 2. Module map

Every module under `src/dual_agent/`.

| Module | Responsibility |
|---|---|
| `state.py` | `AgentState`, `StepRecord`, `StepType`. Holds the goal, step history, and serializes state into the two prompt shapes (`to_system_one_state` compact, `to_system_two_prompt` detailed). |
| `typesafe_client.py` | `JevSystemOneClient`. Wraps the TypeSafe SDK, sends three parallel questions per decision (`Choice` route, `Noul` is_finished, `Noul` needs_synthesis), and falls back to a local keyword stub when the live model is unavailable. |
| `dispatcher.py` | `DualProcessDispatcher`. The agent loop, the fast/slow path split, the confidence gate, `validate_tool_args`, stall detection, latency reconciliation, and session persistence. This is the heart of the runtime. |
| `system_two.py` | `SystemTwoProvider` ABC plus `HermesProvider`, `OpenAICompatibleProvider`, and `MockSystemTwoProvider`; `get_system_two_provider` is the factory that reads `SYSTEM_TWO_PROVIDER`. |
| `mcp_host.py` | `MCPHost`. Registers tool definitions (`MCPToolDefinition`), exposes names+descriptions to System 1, formatted schemas to System 2, and executes handlers with timing. Defines the 4 built-in local tools. |
| `mcp_manager.py` | `MCPManager`. Reads/writes `~/.dual_agent/mcp_servers.json` (Claude Desktop `mcpServers` format) and registers one placeholder bridge tool per configured server. |
| `permission_broker.py` | `PermissionBroker`. Renders the `Allow once / Allow session / Deny / Edit args` card, holds session-level allow/deny sets, denies when no terminal exists, and writes every decision to the `approval_audit` table. |
| `memory.py` | `MemoryEngine`. SQLite persistence: sessions, sessions_fts (FTS5), learned_skills, project_context, approval_audit, scheduled_jobs. Also owns USER.md read/append and `build_recall_context`. |
| `skills_manager.py` | `SkillsManager`. Synthesizes `.SKILL.md` files (YAML frontmatter + Markdown) after a successful run, parses them back, scores them by keyword overlap, and builds the skill context block. |
| `scheduler.py` | `CronScheduler` plus `parse_nl_to_cron`. Natural-language → cron parsing, SQLite-backed job CRUD, a 60-second asyncio tick loop, and a hand-rolled due-date matcher. |
| `team_manifest.py` | `TeamManifestManager`. Exports the config/providers/MCP servers/skills as a portable Markdown manifest with YAML frontmatter; imports one from a path or an HTTP(S) URL. |
| `config.py` | `AgentConfig` pydantic model, `.env` file loading with real-env precedence, `load_config`, `save_config` (0600), and the interactive setup wizard. |
| `evaluator.py` | `JevEvaluator`. Thin wrapper exposing `check_task_completion` and `score_output`. **Not wired into the dispatcher** — see limitations. |
| `updater.py` | `perform_update`. `git pull origin master` (falling back to `main`) if running from a clone, then reports the installed `typesafe_sdk` version. |
| `shell.py` | `InteractiveShell`. The Rich TUI REPL, slash-command dispatch (`/tools`, `/recall`, `/schedule`, `/export`, …), and the terminal entry into the dispatcher. |
| `cli.py` | Argument parsing and `run_agent_task`. Pre-dispatch routing for `config`, `update`, `--ui`, `--gateway`, and the measured-only telemetry table. |
| `web/server.py` | FastAPI app factory: dashboard HTML, REST endpoints, and the `/ws` WebSocket protocol. |
| `web/streaming_dispatcher.py` | `StreamingDispatcher` and `WebPermissionBroker`. Runs the dispatcher in an executor, emits per-step events, and routes approval cards to the browser instead of the terminal. |
| `gateway/base.py` | `GatewayAdapter` ABC, `IncomingMessage` normalization, `send_chunked`. Platform-agnostic. |
| `gateway/telegram_adapter.py` | `TelegramAdapter`. python-telegram-bot polling, the user-id allow-list gate, and outbound Markdown messages. |
| `gateway/session_router.py` | `SessionRouter`. One dispatcher (and therefore one memory DB) per `chat_id`. |
| `gateway/runner.py` | `GatewayRunner`. Wires adapter → router → dispatcher, adds the non-interactive permission broker, starts the scheduler as a concurrent task, and does the fail-fast preflight. |

`bridges/` sits at the repo root, not under `src/`:

| Module | Responsibility |
|---|---|
| `bridges/hermes_middleware.py` | `HermesJevRoutingMiddleware.intercept_tool_decision` — a drop-in router for an external agent loop that returns a Jev choice or calls a caller-supplied fallback. |
| `bridges/mcp_evaluator_server.py` | A standalone stdio JSON-RPC MCP server exposing two tools (`jev_evaluate_noul`, `jev_score_rubric`) so any MCP client can call Jev directly. |

---

## 3. The dispatcher loop

`DualProcessDispatcher.run` in `src/dual_agent/dispatcher.py`.

Setup before the loop:

1. Reset `_verification_clock_ms` / `_pending_verification_latency_ms`, build `AgentState(goal, max_steps)`.
2. `self.mcp.get_tool_descriptions()` — the tool name→description map handed to System 1.
3. `self.memory.build_recall_context(goal, limit=3)` — FTS5 top-3 past sessions.
4. `self.skills.build_skill_context(goal)` — keyword-overlap top-3 `.SKILL.md` procedures.
5. `self.memory.find_matching_skill(goal)` — cached routine lookup. **Logged only**; it does not alter routing today.
6. Warn loudly if `self.s1.force_simulation` is set.

Then `for step_idx in range(1, max_steps + 1)`, breaking early if `state.is_completed`.

### 3.1 Termination check (every step, first thing)

`self.s1.evaluate_state_and_route(state_summary, tool_descriptions, allow_escalation=True)`, where
`state_summary` is `AgentState.to_system_one_state()` — a compact block of GOAL, CURRENT STEP
n/max, the last 5 actions, and variable keys.

If `decision.is_terminal` or `decision.selected_tool == "finish_task"`, the loop sets
`state.is_completed`, records a `StepType.TERMINATION` step, and breaks. Termination and routing
are the same model call.

### 3.2 The confidence gate

`_compute_fast_path_confidence(decision, state)` returns the **minimum** of:

1. `decision.confidence` (the router's self-report), and
2. the confidence of the **strongest rival tool, re-checked in isolation**.

Gate 2 exists because the aggregate score is self-reported *and* because tool descriptions differ
in specificity: a narrowly-worded declaration ("read a specific file") loses to a broadly-worded
one ("list a directory") on vague goals and gets outvoted inside one request. Re-asking about the
runner-up alone cancels that bias.

The rival is picked from `decision.probabilities`, excluding the selected tool,
`escalate_to_system_two`, and `finish_task`; non-numeric values are skipped (YAML `yes`/`no` parse
to `bool`, and `float(True) == 1.0` would become a perfect-confidence signal). If there is no
rival, the gate degenerates to the self-report alone.

The re-check goes through `_check_fast_path_confidence`, which calls
`self.s1.evaluate_state_and_route` again with a one-tool dict. This is a **real billed model call**.
`_check_fast_path_confidence` accumulates its wall-clock time into `_verification_clock_ms`, and
`_compute_fast_path_confidence` moves that delta into `_pending_verification_latency_ms`, which the
main loop adds to `s1_latency_total` so verification is attributed rather than hidden.

**Fast path is taken only if all three hold:**

```
fast_path_confidence >= self.confidence_threshold   # default 0.85, env SYSTEM_ONE_CONFIDENCE_THRESHOLD
and not decision.needs_generation
and decision.selected_tool in tool_descriptions
```

### 3.3 Argument inference and the refusal

On the fast path, `tool_name = decision.selected_tool` and
`default_args = self._infer_default_args(tool_name, state)`. That function is **regex extraction
from goal text, not understanding**:

- `list_directory` → `{"path": _extract_path(goal, default=".")}`
- `read_file` → `{"path": _extract_path(goal, default="")}`
- `run_shell_command` → `{"command": _extract_shell_command(goal)}` — returns a command only when
  the goal quotes one; never invents one. Anything else gets `{}`.

Then `validate_tool_args(self.mcp.get_tool(tool_name), default_args)` checks the declared schema:
every `required` name present and non-blank, and every supplied value matching its declared type
(`string`/`integer`/`boolean`, with `bool` explicitly rejected for `integer`).

**On failure the fast path is refused.** The dispatcher logs it and either
(a) if `DUAL_AGENT_ALLOW_UNVERIFIED_FAST_PATH=true`, executes anyway with a warning, or
(b) sets `decision.needs_generation = True` and `can_use_fast_path = False`, falling through to
System 2 so a model can read the goal and supply real arguments.

This exists because the router only chose a *name*. Sending plausible-but-wrong args (reading
`pyproject.toml` for an unrelated goal) looks like success and is worse than failing.

### 3.4 Permission gate and execution

If `tool_def.requires_approval`, `self.broker.request_approval(tool_name, default_args,
risk_level)` runs. A `DENY` appends a step record with output `[DENIED by user]` and `continue`s —
note it consumes a step index. An `EDITED`/`ALLOW` returns possibly-rewritten args that are then
used. `write_file` (medium risk) and `run_shell_command` (high risk) both carry
`requires_approval=True`.

Then `self.mcp.execute_tool(tool_name, default_args)` and one `StepRecord` is appended with
`step_type=SYSTEM_ONE_FAST_TOOL`, `tokens_used=0`, and `metadata={"simulated": decision.simulated}`.

### 3.5 Stall detection

After a fast-path step, `signature = (tool_name, repr(default_args), str(output_val)[:200])`. If
the signature equals `last_signature`, `repeat_count += 1`, else it resets to 1. At
`repeat_count >= 3` the run ends early with a `final_output` explaining that the goal is not
progressing.

This matters economically: in live mode every one of those steps would be a paid Jev call, so the
runtime stops rather than paying to produce nothing.

### 3.6 The slow path (System 2)

Entered when the gate fails, when `needs_generation` is set, when the chosen tool is not registered,
or when argument validation refused the fast path.

1. `context_prefix` = recall context + skill context, each followed by a newline.
2. `s2_prompt = context_prefix + state.to_system_two_prompt(self.mcp.get_formatted_tool_list_for_system_two())`.
3. `self.s2.generate_step(s2_prompt)`.
4. If `action == "finish_task"`, complete and record `SYSTEM_TWO_GENERATION` carrying
   `s2_response.tokens_used`.
5. Otherwise, **the same permission broker gates the System 2 call** (identical
   `requires_approval` / deny / record path), then `self.mcp.execute_tool(s2_action, s2_args)`.

Note: System 2's `args` are **not** run through `validate_tool_args`. Only the fast path is
schema-validated.

### 3.7 Accounting and reconciliation

Two corrections run after the loop so the reported numbers cannot drift optimistic:

- **Verification remainder.** Router time spent on the step that then `break`s out via the slow
  path would never reach `total_latency_ms`. If `_pending_verification_latency_ms` is non-zero it is
  folded into the last record (or, with no history, recorded as a synthetic
  `SYSTEM_ONE_EVALUATION` / `route_verification` step). Without this, System 1 latency could exceed
  total run latency, which is impossible.
- **Wall-clock reconciliation.** `wall_clock_ms = time.perf_counter() - run_started_at` is compared
  to `state.total_latency_ms`. If the wall clock is larger, the difference is added to the last
  record and tagged in `metadata["latency_reconciled_ms"]`. Hand-maintained sums drift; the wall
  clock cannot be mislaid, so it wins.

Then `memory.save_session(...)` writes the run, and if the run completed with at least one tool
step, `memory.save_learned_skill(...)` and `skills.synthesize_skill(...)` both fire (the latter
inside its own try/except so a skill write failure cannot fail the run).

---

## 4. One agent step, end to end

A fast-path step, in order:

| # | Phase | Code | What happens |
|---|---|---|---|
| 1 | **Route** | `JevSystemOneClient.evaluate_state_and_route` | Sends state + all tool names + `escalate_to_system_two` + `finish_task` as `Choice`, plus `is_finished` and `needs_synthesis` as `Noul`. Returns `JevDecision`. |
| 2 | **Check terminal** | `dispatcher.run` | `is_terminal` or `finish_task` → complete, record `TERMINATION`, break. |
| 3 | **Verify** | `_compute_fast_path_confidence` → `_check_fast_path_confidence` | Re-asks about the strongest rival in isolation. Takes the **min** of self-report and rival score. Bills the call to `_pending_verification_latency_ms`. |
| 4 | **Gate** | `dispatcher.run` | `confidence >= threshold && !needs_generation && tool in descriptions`. Any failure → slow path. |
| 5 | **Infer args** | `_infer_default_args` | Regex-extracted from goal text. A guess. |
| 6 | **Validate args** | `validate_tool_args` | Schema check. Failure → refuse, set `needs_generation`, go to System 2. |
| 7 | **Permission** | `PermissionBroker.request_approval` | Only if `requires_approval`. Deny → record `[DENIED by user]`, `continue`. |
| 8 | **Execute** | `MCPHost.execute_tool` | Calls the handler, times it, returns `ToolExecutionResult` (success/error captured, never raised). |
| 9 | **Record** | `state.history.append(StepRecord(...))` | `SYSTEM_ONE_FAST_TOOL`, real `latency_ms`, `tokens_used=0`. Also updates the stall signature. |

### Where telemetry is recorded

| Signal | Recorded where | Notes |
|---|---|---|
| Per-step latency | `StepRecord.latency_ms` | Router time + tool execution time |
| Per-step tokens | `StepRecord.tokens_used` | `0` on fast path; `s2_response.tokens_used` on slow path |
| Run totals | `DispatchResult` (`total_latency_ms`, `system_one_latency_ms`, `system_two_latency_ms`, `tokens_used`, per-layer step counts) | Measured only |
| Simulation disclosure | `DispatchResult.used_simulated_system_one`, `.system_one_fallback_reason`, `.simulated_latency_ms` | Also `StepRecord.metadata["simulated"]` |
| Reconciliation | `StepRecord.metadata["latency_reconciled_ms"]` | Only when the wall clock exceeded the step sum |
| Durable history | SQLite `sessions` row: `total_steps`, `system_one_steps`, `system_two_steps`, `total_latency_ms`, `tokens_used`, `steps_json` | `token_savings_pct` is written as `NULL` |
| Approval decisions | SQLite `approval_audit` (`tool_name`, `args_json`, `decision`, `decided_at`) | Via `PermissionBroker._log_decision` |

`DispatchResult` deliberately has **no** savings or speedup field. See
`tests/test_end_to_end.py::test_dual_process_run_reports_measured_values_only`, which asserts those
attributes do not exist.

---

## 5. System 1: live vs simulated

`JevSystemOneClient.__init__` sets `simulation_reason` and `force_simulation` when any of:
`force_simulation=True` was passed, `TYPESAFE_API_KEY` is unset, or `typesafe_sdk` is not
importable. A `TypeSafeClient` construction failure also flips it, with the exception text as the
reason.

`_call_live_jev` sends one `system_one(state, questions)` request with three parallel questions and
reports `simulated=False`; any exception flips the client permanently to simulation and returns a
simulated decision.

`_call_simulated_jev` is a **deterministic keyword-matching stub**, not the Jev model:

- `time.sleep(0.012)` — a **simulation artifact** emulating a network round-trip. `_call_simulated_jev`
  uses it in place of the sleep; `evaluate_output_score` uses `0.008` offline.
- Terminal if `"goal achieved"` or `"all steps completed"` appears in the state text.
- Needs generation if `synthesize`, `write a novel`, `creative`, or `complex refactor` appears.
- Otherwise picks the first tool whose last name segment appears in the state text, else the first
  regular tool, else `escalate_to_system_two`.
- Confidence `0.94` (or `0.45` when generation is needed) — **a constant, not a measurement**.
- Every return carries `simulated=True` and `fallback_reason`.

`evaluate_output_score` has the same split: live uses `Jev Score` over a 5-point criteria list
(clamped 1–5); offline returns `5` if the text is longer than 10 characters else `2` — a length
check, not a quality judgement.

---

## 6. Tool layer

### 6.1 MCPHost and the built-in tools

`MCPHost` holds `Dict[str, MCPToolDefinition]` and registers four tools in
`_register_default_tools`:

| Tool | Schema | Approval | Risk | Behaviour |
|---|---|---|---|---|
| `list_directory` | `{path: string}` required | no | low | `os.listdir`, first 50 entries + total, as JSON |
| `read_file` | `{path: string}` required | no | low | Reads up to 10,000 chars |
| `write_file` | `{path, content: string}` required | **yes** | medium | `makedirs` + write, reports char count |
| `run_shell_command` | `{command: string}` required | **yes** | high | `shlex.split` then `subprocess.run(shell=False, timeout=15)` |

`run_shell_command` never uses `shell=True`. With `shell=True`, a command assembled from model
output plus an interpolated argument turns any quoting mistake into injection. The argv form also
means the approval card shows the real executable, so approving `ls -la` cannot run something else.
Consequence, stated in the tool's own description: pipes, redirects, globbing and `$VAR` expansion
are unavailable. Output is capped at 2,000 chars; timeouts return an error string after 15s.

`execute_tool` catches every exception and returns `success=False` with the error text — a tool
failure never propagates out of the dispatcher loop.

### 6.2 The MCP story, accurately

`MCPManager.attach_to_host` iterates `mcp_servers.json`, skips `disabled` entries, and registers
**one placeholder tool per server** named `mcp_<name>_dispatch` whose handler is:

```python
lambda args, n=name: f"Dispatched to external MCP server '{n}': {args}"
```

There is no process spawn, no JSON-RPC handshake, and no tool enumeration from the remote server.
The `mcp>=2.0.0` dependency in `pyproject.toml` is **never imported anywhere in `src/`**. So
"external MCP server" today means: a placeholder the router may select, which returns a string
saying it was dispatched. `bridges/mcp_evaluator_server.py` is a real MCP server implementation,
but it is the thing being served *to* other clients, not something this runtime connects to.

---

## 7. Memory

`MemoryEngine` (`src/dual_agent/memory.py`) — one SQLite file at
`~/.dual_agent/memory.db`, directory mode `0700`, `PRAGMA journal_mode=WAL`.

| Table | Purpose |
|---|---|
| `sessions` | One row per run: goal, outcome, completion, per-layer step counts, latency, `tokens_used`, `token_savings_pct`, `steps_json` |
| `sessions_fts` | FTS5 external-content virtual table over `sessions.goal` + `sessions.outcome`, kept in sync by the `sessions_ai` AFTER INSERT trigger |
| `learned_skills` | `name` UNIQUE, space-joined `intent_keywords`, `tool_sequence_json`, `success_count`, timestamps. Upsert increments the count. |
| `project_context` | Per-`workspace_path` tech stack + preferences JSON |
| `approval_audit` | Every broker decision |
| `scheduled_jobs` | Cron jobs (also created defensively by `CronScheduler._init_table`) |

**Recall.** `full_text_search(query, limit)` runs `WHERE sessions_fts MATCH ? ORDER BY rank, id DESC`
joined back to `sessions`. Any exception falls back to `_fallback_search`, which is an `OR`-joined
`LIKE` over the first three query terms. `build_recall_context(query, limit=3)` formats matches into
a `[RELEVANT PAST SESSIONS]` … `[END PAST SESSIONS]` block and returns `""` when there are none —
this is what gets prepended to the System 2 prompt.

**Skill lookup.** `find_matching_skill(goal)` scans `learned_skills` by `success_count` and returns
the first row whose keyword set is a **subset** of the goal's word set.

**USER.md.** `get_user_profile` / `update_user_profile` read and append to `<data_dir>/USER.md`,
creating a dated header on first write. Exposed via `/whoami`. **Nothing in the dispatcher calls
`update_user_profile`** — see limitations.

**Honest aggregates.** `get_aggregate_stats` returns `avg_token_savings_pct=None` (not `0.0`) when
no session carries a measured figure, so "not measured" is distinguishable from "measured zero". The
shell prints `n/a (no baseline run)` in that case.

---

## 8. Skills synthesis

`SkillsManager` (`src/dual_agent/skills_manager.py`), directory `~/.dual_agent/skills/` (mode
`0700`, overridable via `DUAL_AGENT_SKILLS_DIR`).

After a completed run with at least one action other than `finish_task`, the dispatcher calls
`synthesize_skill(goal, tool_sequence, outcome)`:

- `_slugify(goal)` → filename `<slug>.SKILL.md` (lowercased, non-alphanumerics stripped, spaces →
  hyphens, 40 chars max).
- If the file exists, it re-parses and rewrites it with `success_count + 1`. Otherwise it writes a
  new file: YAML frontmatter (`name`, `description`, `intent_keywords`, `success_count`,
  `last_used`) plus a body with a numbered Tool Sequence, a "When to Use" keyword line, and a note
  that the file is auto-synthesized and meant to be hand-edited.
- `_extract_keywords` filters a small stopword set and keeps up to 6 words longer than 3 chars.

Retrieval is keyword overlap: `find_relevant_skills` scores `len(keyword_set & goal_words) *
success_count` and takes the top 3. `build_skill_context` renders them as a
`[RELEVANT SKILLS FROM MEMORY]` block for prompt injection — a separate mechanism from the SQLite
`learned_skills` table, which is written in parallel.

Parsing is regex over frontmatter (`_parse_file`), not a YAML library, which is why
`_check_fast_path_confidence` types its numeric guard so carefully.

---

## 9. Scheduler

`parse_nl_to_cron` (`src/dual_agent/scheduler.py`) handles: `every minute`, `every N minutes`,
`every hour`/`hourly`, `every night`/`nightly` → `0 2 * * *`, `every morning` → `0 9 * * *`,
`every week`/`weekly` → `0 9 * * 1`, `every day at H[:MM] [am|pm]`, and `every <weekday> at H[:MM]
[am|pm]`. Anything else returns `None` and `add_job` raises `ValueError` with usage examples.

`CronScheduler` stores jobs in the `scheduled_jobs` table (shared with the main memory DB, which is
why the shell's `/schedule` writes are visible to the gateway daemon).

`run_forever(interval_seconds=60)` ticks in a loop, swallowing per-tick exceptions. `_tick` iterates
enabled jobs and calls `_is_due`. `_is_due` is a **hand-rolled, minute-resolution** matcher:

- Requires exactly 5 space-separated fields.
- `matches` supports `*`, `*/N` (modulo), and exact string equality — **no** lists (`1,2`), ranges
  (`1-5`), or names.
- Day-of-week is computed as `now.weekday() + 1 if weekday < 6 else 0` — Monday=1 … Saturday=6,
  **Sunday=0**.
- Guards against double-fire with a 55-second window since `last_run_at`.

`_run_job` builds a fresh dispatcher from `dispatcher_factory()` (or skips if none is configured),
runs the goal, and updates `last_run_at` / `run_count` in a `finally`.

**The scheduler only runs inside a long-lived process.** `GatewayRunner.run` starts
`asyncio.create_task(self.scheduler.run_forever())` alongside polling. `dual-agent --ui` constructs
a `CronScheduler` for the `/api/schedules` endpoint but never ticks it.

---

## 10. Permission broker

`PermissionBroker.request_approval(tool_name, args, risk_level)` decides in this order:

1. Tool in `_session_allowed` → `ALLOW_ONCE` (no card).
2. Tool in `_session_denied` → `DENY`.
3. `auto_allow` (from `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS`) → `ALLOW_ONCE`.
4. `non_interactive` → log a warning, audit `DENY`, return `DENY`. A `Prompt.ask` here would block
   on stdin or raise `EOFError` and hang the daemon; an unanswerable approval is denied instead.
5. Otherwise render the Rich card and loop on `Prompt.ask` with choices `a/s/d/e`:
   - `a` → `ALLOW_ONCE`
   - `s` → add to `_session_allowed`, `ALLOW_SESSION`
   - `d` → `DENY`
   - `e` → read a JSON line from the user; valid JSON is returned as `EDITED` args, invalid JSON
     falls back to `ALLOW_ONCE` with the original args.

Every path except the session shortcut calls `_log_decision`, which inserts into `approval_audit`
inside a try/except so audit failure never breaks the agent.

`WebPermissionBroker` (`web/streaming_dispatcher.py`) subclasses this for the dashboard: it emits a
`permission` event with a short `request_id`, parks an `asyncio.Future` in the module-level
`_pending_approvals` dict, and blocks the worker thread on a `concurrent.futures.Future` with a
**120-second timeout** that resolves to `"deny"` on expiry. `resolve_permission(request_id, ...)` is
what the WebSocket handler calls to release it.

---

## 11. Gateway

`GatewayRunner` (`src/dual_agent/gateway/runner.py`) owns the long-lived process.

Construction: one shared `MCPHost` (with `MCPManager` bridges attached), **one shared
`JevSystemOneClient`** (so a live-API failure is global and honest rather than per-chat and
inconsistent), one shared System 2 provider, one `MemoryEngine` for jobs, a `CronScheduler` whose
factory targets `session_id="scheduler"`, and a `SessionRouter`.

`build_dispatcher(session_id)` creates `~/.dual_agent/sessions/<session_id>/memory.db` — **one
memory DB per chat** — plus a `PermissionBroker(non_interactive=not auto_allow)`.

`handle_message` resolves the per-chat dispatcher, runs `dispatcher.run` in a thread executor
(SQLite and subprocesses are blocking; keep the event loop free), and replies via
`format_reply`, which prints the outcome, the measured `steps (S1: n, S2: n) in Nms` line, and a
`⚠️ Router was SIMULATED (...)` line when applicable.

`_preflight` fails fast and loudly: a missing `TELEGRAM_BOT_TOKEN` raises a `ValueError` with the
BotFather + `@userinfobot` + `.env` steps instead of a traceback, and it warns about simulation
mode, an empty allow-list, and the denial of risky tools.

**Authorization.** `TelegramAdapter` reads `TELEGRAM_ALLOWED_USER_IDS` into a set of strings. In
`_handle`, `sender_id` is compared against that set and a mismatch logs a warning, replies
`⛔ Not authorized...`, and returns. **An empty list admits nobody** — the default is closed. This
matters because the bot can write files and run shell commands on the host, so an open bot is
remote code execution for whoever finds it.

**Non-interactive deny policy.** `build_dispatcher` passes `non_interactive=not auto_allow`, so
with `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS` unset, `write_file` and `run_shell_command` are refused
rather than silently permitted. There is no terminal behind a chat to answer a card.

Only text messages are handled — the handler filter is `filters.TEXT & ~filters.COMMAND`.

---

## 12. Dashboard and WebSocket protocol

`create_app` (`src/dual_agent/web/server.py`) builds one shared `MemoryEngine`, `MCPHost`,
`SkillsManager`, `CronScheduler`, and `DualProcessDispatcher`. Note the dispatcher is constructed
with `force_simulation=True` on System 1 — **the dashboard never calls live Jev**, which keeps the
UI free to poke at.

Dashboard HTML is read from `web/templates/dashboard.html` and cached in a dict, with literal
`{{ provider }}` / `{{ tool_count }}` / `{{ version }}` string substitution — no Jinja2 render
(avoiding a Python 3.14 cache bug).

REST endpoints:

| Method | Path | Returns |
|---|---|---|
| GET | `/` | Dashboard HTML |
| GET | `/api/skills` | Parsed `.SKILL.md` list |
| GET | `/api/memory` | `get_aggregate_stats()` + top-10 learned skills |
| GET | `/api/tools` | Tool name, description, `requires_approval`, `risk_level` |
| GET | `/api/schedules` | All cron jobs |
| POST | `/api/schedule` | `{description, goal}` → `{ok, job_id}` or `{ok: false, error}` |
| GET | `/api/profile` | USER.md content |
| WS | `/ws` | Below |

**WebSocket protocol.** Server → client events:

| `type` | Payload |
|---|---|
| `init` | `provider`, `tool_count`, `skills_count` — sent on connect |
| `started` | `goal` |
| `step` | `step`, `path` (`S1_FAST`), `action`, `args`, `output` (400 chars), `latency_ms`, `tokens_used` |
| `permission` | `request_id`, `tool`, `args`, `risk` |
| `done` | `is_completed`, `final_output`, `total_steps`, `system_one_steps`, `system_two_steps`, `total_latency_ms`, `tokens_used`, `used_simulated_system_one` |
| `error` | `message` |
| `pong` | reply to `ping` |

Client → server: `{"type": "run", "goal": str, "max_steps": int}` (empty goal → `error` event),
`{"type": "permission_response", "request_id", "decision", "edited_args"}`,
`{"type": "ping"}`.

Implementation notes worth knowing: `run_streaming` swaps in the `WebPermissionBroker`, copies the
existing session allow/deny sets across, **monkey-patches `dispatcher.mcp.execute_tool`** to emit a
`step` event after each real tool call, and restores both in `finally`. The `step` events are
therefore emitted from the `execute_tool` wrapper, not from the dispatcher, and they always report
`path: "S1_FAST"` — the path label does not distinguish fast from slow steps.

`run_server` refuses to bind to anything but loopback unless
`DUAL_AGENT_UI_ALLOW_PUBLIC_BIND=true`, because the dashboard has **no authentication** and its
WebSocket can execute tools with your privileges.

---

## 13. Configuration

`load_config` (`src/dual_agent/config.py`) calls `load_env_files()` first, then reads
`~/.dual_agent/config.json` (written `0600`), then fills gaps from environment variables.
`_candidate_env_files` checks `./.env` and `~/.dual_agent/.env`, and **existing environment
variables always win** — `key not in os.environ` guards every assignment, so an exported key is
never clobbered by a stale file.

Relevant environment variables:

| Variable | Default | Effect |
|---|---|---|
| `TYPESAFE_API_KEY` | — | Absent ⇒ System 1 runs simulated, with the reason recorded |
| `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` | Jev endpoint |
| `SYSTEM_TWO_PROVIDER` | `mock` | `hermes`/`ollama`, `grok`/`xai`, `openai`/`gpt`, else mock |
| `SYSTEM_ONE_CONFIDENCE_THRESHOLD` | `0.85` | Fast-path gate |
| `DUAL_AGENT_ALLOW_UNVERIFIED_FAST_PATH` | `false` | Executes with unvalidated args |
| `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS` | `false` | Skips approval cards; also disables the gateway's non-interactive deny |
| `TELEGRAM_BOT_TOKEN` | — | Required for `--gateway` |
| `TELEGRAM_ALLOWED_USER_IDS` | *empty* | **Empty ⇒ every message rejected** |
| `DUAL_AGENT_UI_ALLOW_PUBLIC_BIND` | `false` | Permits a non-loopback dashboard bind |
| `DUAL_AGENT_HOME` | `~/.dual_agent` | Data directory |
| `DUAL_AGENT_SKILLS_DIR` | `<home>/skills` | Skill files |
| `HERMES_BASE_URL` / `HERMES_MODEL` | `http://localhost:11434/v1` / `nous-hermes-3-llama-3.1-8b` | Local Hermes provider |
| `GROK_API_KEY` / `GROK_MODEL` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | Remote System 2 providers |

---

## 14. Entry points

`cli.py:main` dispatches on `sys.argv[1]` before argparse:

| Invocation | Path |
|---|---|
| *(no args)* | `InteractiveShell().start()` — the TUI REPL |
| `config` | `run_configuration_wizard()` |
| `update` | `updater.perform_update()` |
| `ui` / `--ui` | `web.server.run_server(host="localhost", ...)`, `--port=N`, `--no-browser` |
| `--gateway` / `gateway` / `--daemon` / `daemon` | `gateway.runner.run_gateway()`, `--max-steps=N` |
| anything else | argparse with `--goal`, `--provider`, `--threshold`, `--max-steps` |

`--provider` offers `mock`, `hermes`, `grok`, `anthropic`, `openai`. **`anthropic` is accepted by
argparse and by the wizard but is not handled by `get_system_two_provider`** — it falls through to
`MockSystemTwoProvider`.

---

## 15. Known limitations

Plain statements of what does not work or is weak today, read off the code.

**Generation**

- **System 2 defaults to `mock`.** `get_system_two_provider` returns `MockSystemTwoProvider` unless
  `SYSTEM_TWO_PROVIDER` says otherwise. `MockSystemTwoProvider.generate_step` sleeps 300ms and
  returns one of two canned branches: a hardcoded summary string, or a `write_file` call writing
  `generated_script.py` containing `print('System 2 output')`. It reports `tokens_used=450` — a
  constant — for either branch. **Out of the box the agent cannot actually generate anything.**
- `HermesProvider` and `OpenAICompatibleProvider` both **silently fall back to the mock** on any
  exception, including a missing API key. A provider outage is indistinguishable from a mock run in
  the step record, which carries the mock's `tokens_used=450` as if it were real.
- `anthropic` is a selectable provider that is not implemented.

**Routing**

- **The internal router cannot express state transitions.** `JevDecision` carries one tool name
  plus flags; `AgentState.variables` exists but is written by nothing in the dispatcher, so a
  multi-step workflow with branches or carried values has no representation. Each step is an
  independent choice over the same goal text.
- The simulated router picks by substring-matching a tool's last name segment against the state
  text, and its confidence is a hardcoded `0.94` / `0.45`. Fast-path quality on the stub is
  keyword luck, not judgement.
- `evaluate_output_score` offline returns `5` for any text longer than 10 characters and `2`
  otherwise. It is a length check, not a quality judgement — do not read it as a score.
- `_resolve_fast_path_tool` and `_select_argv0_tool` handle the "router named only `git`, not
  `git add`" case by asking System 2 to disambiguate. With the mock provider, that prompt returns
  the canned `write_file` branch and the disambiguation fails closed with "No tool was executed."
- The fast path's argument inference is regex over the goal string. It is a guess, which is exactly
  why the default path refuses to run on it.

**Tools**

- **No MCP client exists.** `MCPManager.attach_to_host` registers a placeholder whose handler
  returns `f"Dispatched to external MCP server '{n}': {args}"`. No subprocess, no handshake, no tool
  discovery. The `mcp` dependency is never imported. Only the 4 built-in tools do real work.
- `run_shell_command` cannot use pipes, redirects, globs, or `$VAR`, because it is argv-only by
  design. There is no allow-list or blocklist on the binary either — anything on `PATH` is
  reachable if the user approves the card.
- `validate_tool_args` covers only `string`, `integer`, and `boolean`, and does not reject unknown
  keys. System 2's tool arguments are not validated at all — only the fast path is.
- `read_file` truncates at 10,000 characters with no indication that truncation happened.

**Loop and termination**

- `max_steps` (default 15 in `run`, 10 from the CLI and gateway) is the only hard stop. There is no
  goal-decomposition step, so a goal needing more than `max_steps` sequential steps simply ends
  incomplete.
- Stall detection compares `(tool, args, first 200 chars of output)`. A loop that alternates between
  two tools, or whose output varies slightly each step, is not caught.

**Memory and learning**

- **`token_savings_pct` is written as `NULL`**, and `get_aggregate_stats` returns `None` for the
  average, because no baseline run is performed. The shell renders `n/a (no baseline run)`. This is
  deliberate — the predecessor code multiplied step count by hardcoded constants (1500 tokens /
  1200 ms) to produce "savings" and a CLI column labelled `Traditional LLM Baseline`. There is no
  timing, token-saving, or speedup claim anywhere in this runtime, and none should be inferred.
- **`USER.md` is never written by the agent.** `update_user_profile` exists and is tested, but
  nothing in `dispatcher.py`, `shell.py`, or `gateway/` calls it. `/whoami` will report "No USER.md
  profile yet" on a fresh install no matter how much you use it.
- `find_matching_skill`'s subset test breaks on any extra goal word: keywords `["inspect",
  "pyproject"]` matches "inspect pyproject" but not "please inspect the pyproject files". The result
  is **logged only** — it does not currently change routing or skip steps, so the "learned skill
  replay" path is inert.
- Skill retrieval scores `overlap * success_count`, where `success_count` increments on every
  re-synthesis. A frequently-repeated skill can outrank a better keyword match.
- `build_recall_context` is computed once before the loop, so context never updates mid-run.
- `skills.synthesize_skill` fires on *any* completed run with ≥1 tool step, including runs that
  completed because stall detection gave up. Failed or blocked runs can therefore produce a
  `.SKILL.md` describing a procedure that did not work.
- `save_project_context` / `get_project_context` exist (and are created in the schema) but nothing
  calls them. `AgentConfig.auto_learn_skills` and `auto_scan_workspace` are never read.

**Scheduler**

- `_is_due` supports only `*`, `*/N`, and exact matches — no lists, ranges, or month/weekday names.
- Jobs only fire while `dual-agent --gateway` is running. `--ui` builds a `CronScheduler` but never
  ticks it, so `/api/schedules` can list jobs that never run.
- `_run_job` renders its output with a `rich` `Console` to stdout, which in a daemon is usually
  discarded. Scheduled results are not sent over the gateway to any chat.
- `_tick` catches and logs per-job exceptions inside `_run_job`, but `_tick` itself iterating a job
  whose row is malformed raises into `run_forever`'s handler for the whole tick.

**Gateway and dashboard**

- Telegram only. `GatewayAdapter` is an ABC with one implementation; `filters.TEXT & ~COMMAND` means
  commands, photos, documents, and voice are ignored.
- Gateway runs default to `max_steps=10`.
- `handle_message` runs a blocking `dispatcher.run` per message in the default executor. Two
  concurrent messages in the same chat share one dispatcher, whose `_current_goal` and state are
  single-valued — there is no per-chat run lock.
- `SessionRouter._sessions` grows without bound; `destroy` is never called by the runner.
- The dashboard has **no authentication** and can execute tools as the user. Loopback-only by
  default, gated by an explicit opt-out for non-loopback binds.
- The dashboard hardcodes `force_simulation=True` for System 1, so UI runs never exercise live Jev.
- WebSocket `step` events are emitted by a monkey-patched `execute_tool` and always tag
  `path: "S1_FAST"`; the slow path is indistinguishable in the event stream.
- Streaming is step-level, not token-level. There is no partial-output streaming anywhere: System 2
  responses are requests with `stream` unset, and `httpx.Client.post` reads the whole body.
- WebSocket responses are not correlated to the request that started them; two overlapping `run`
  messages interleave events on one socket.

**Permissions**

- With `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true`, `non_interactive` is set to `False` in the gateway
  — meaning approval *cards* are enabled rather than the deny policy. Risky tools are permitted
  unattended, which the code documents but cannot enforce.
- `WebPermissionBroker.request_approval` instantiates `ApprovalDecision(decision_str)`, which raises
  `ValueError` on any unrecognized string.

**Packaging and operations**

- There is no Dockerfile, no CI config, and no published-package workflow in the repo; `install.sh`
  is the distribution mechanism.
- The test suite is the only verification. There is no benchmark harness, no baseline runner, and no
  evaluation dataset.
- `updater.perform_update` shells out to `git pull origin master` / `main` in whatever directory the
  process happens to be in. It does not reinstall, verify signatures, or check what changed.
- `bridges/hermes_middleware.py`'s docstring claims offloading saves "up to 80% of outer-loop token
  costs". That figure is not measured by anything in this repo and contradicts the measured-only
  policy everywhere else. Treat it as documentation debt.
