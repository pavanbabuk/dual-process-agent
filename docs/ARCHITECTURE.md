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
| `screen.py` | Perception, Set-of-Mark grid annotation, non-model pixel outcome verification (`screen_diff`), and desktop actuation primitives (`mouse_click`, `mouse_move`, `key_press`) with bounds validation, kill switch (`DUAL_AGENT_SCREEN_CONTROL`), and dynamic Retina scale calibration. |
| `scheduler.py` | `CronScheduler` plus `parse_nl_to_cron`. Natural-language → cron parsing, SQLite-backed job CRUD, a 60-second asyncio tick loop, and a hand-rolled due-date matcher. |
| `team_manifest.py` | `TeamManifestManager`. Exports the config/providers/MCP servers/skills as a portable Markdown manifest with YAML frontmatter; imports one from a path or an HTTP(S) URL. |
| `config.py` | `AgentConfig` pydantic model, `.env` file loading with real-env precedence, `load_config`, `save_config` (0600), and the interactive setup wizard. |
| `evaluator.py` | `JevEvaluator`. Fast-path outcome verification, step guardrails, and quality scoring. Actively wired into `dispatcher.py` (`verify_step_outcome` for post-execution checks, `check_task_completion` for completion validation, and `score_output`). |
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

**Fast path is taken only if all conditions hold (Two-Condition Distribution Gate & Per-Action Risk Threshold):**

```python
# 1. Router distribution concentration and option probability gates (rejects flat / uncertain distributions)
distribution_confidence >= THRESHOLD_DISTRIBUTION_CONCENTRATION  # default 0.65
and option_probability >= THRESHOLD_OPTION_PROBABILITY            # default 0.50

# 2. Risk-calibrated threshold per action
and fast_path_confidence >= self._get_tool_confidence_threshold(tool_name)
# (low risk read-only: ~0.75; medium risk file modification: ~0.88; high risk destructive: ~0.92)

# 3. Decision flags and tool validity
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

### 3.4 Classification separate from authorization

In accordance with vendor guidance, Jev is used strictly for intent classification and reflex routing,
never for authorizing destructive actions.

Authorization is governed strictly by the `PermissionBroker`:
If `tool_def.requires_approval`, `self.broker.request_approval(tool_name, default_args, risk_level)`
runs. A `DENY` appends a step record with output `[DENIED by user]` and `continue`s — note it consumes
a step index. An `EDITED`/`ALLOW` returns possibly-rewritten args that are then used. `write_file`
(medium risk) and `run_shell_command` (high risk) both carry `requires_approval=True`.

### 3.5 The Plan → Act → Verify → Correct Loop

Before execution begins, `_create_plan(goal, ...)` generates an ordered sequence of concrete
`PlanStep` subgoals attached to `state.plan`.

1. **Plan**: System 2 or heuristic goal decomposition generates 2 to 4 structured subgoals.
2. **Act**: The active step is executed either via System 1 reflex or System 2 deliberate generation.
3. **Verify**: Following execution, `JevEvaluator.verify_step_outcome(subgoal, action, args, output)` validates:
   - Output validity (null/empty outputs rejected, except valid empty stdout on successful shell commands).
   - Tool error and denial detection.
   - Filesystem effects: target file presence for `write_file`/`patch_file`/`read_file`.
   - AST validation: `validate_python_syntax` runs on all modified `.py` files to catch syntax regressions immediately.
4. **Correct**: If verification fails:
   - Feedback notes and verification error guidance are injected into the prompt for the next step.
   - Bounded retries: up to 3 attempts per subgoal. If a subgoal fails 3 times, the agent stops immediately and reports an honest failure (`is_completed=False`) rather than hallucinating success.

### 3.6 Screen Perception, Desktop Actuation, and Visual Verification

When the goal involves visual desktop interaction (or when `enable_screen_loop=True` / `DUAL_AGENT_SCREEN_LOOP=1`), the agent interacts directly with the live graphical environment:

1. **Perception**: At the start of each step, `screenshot` captures the active display and `grid_overlay` renders a calibrated coordinate grid overlay (`step_grid_path`), which is passed to multimodal System 2 models via `generate_step(prompt, images=[step_grid_path])`.
2. **Dynamic Calibration**: Coordinate scaling is computed dynamically at runtime (`pixel_resolution / logical_bounds`) via macOS Quartz APIs, supporting varied Retina scale factors (~1.336 or 2.0) without hardcoded offsets.
3. **Desktop Actuation Primitives**:
   - `mouse_click(x, y, button, click_type)`: Actuates clicks at logical coordinates via Quartz.
   - `mouse_move(x, y)`: Moves mouse cursor on active screen.
   - `key_press(key, modifiers)`: Sends keystrokes against a strict allow-list of named keys (`return`, `tab`, `escape`, arrow keys) and Unicode characters.
4. **Safety & Permission Gating**:
   - **Single checkpoint**: Actuation tools carry `requires_approval=True` and `risk_level="high"`. They pass through identical schema validation and the `PermissionBroker` on both fast and slow paths.
   - **Hardware kill switch**: `DUAL_AGENT_SCREEN_CONTROL=0` immediately halts and rejects any actuation attempt.
   - **Display bounds enforcement**: Clicks or moves outside logical screen bounds are rejected before posting events.
   - **OS Accessibility permissions**: Detects if macOS accessibility (`AXIsProcessTrusted`) is absent and raises actionable instructions.
5. **Physical Outcome Verification (`screen_diff`)**:
   - Following every actuation action (`mouse_click`, `key_press`), `screen_diff` computes the pixel difference against the pre-action screenshot without calling a model.
   - **No model self-reporting**: The model's claim that a button was clicked is treated as a hypothesis. Only `screen_diff` decides whether the screen changed.
   - If `screen_diff` reports no change (`changed=False`), the step fails verification (`is_verified=False`), injecting auto-correction guidance for coordinate or timing retry.
   - Three consecutive identical actuation attempts with no visual screen change trigger the stall guard and terminate the run.

### 3.7 Stall detection

After a fast-path step, `signature = (tool_name, repr(default_args), str(output_val)[:200])`. If
the signature equals `last_signature`, `repeat_count += 1`, else it resets to 1. At
`repeat_count >= 3` the run ends early with a `final_output` explaining that the goal is not
progressing.

This matters economically: in live mode every one of those steps would be a paid Jev call, so the
runtime stops rather than paying to produce nothing.

### 3.7 The slow path (System 2)

Entered when the gate fails, when `needs_generation` is set, when the chosen tool is not registered,
or when argument validation refused the fast path.

1. `context_prefix` = recall context + skill context + verification failure correction guidance.
2. `s2_prompt = context_prefix + state.to_system_two_prompt(self.mcp.get_formatted_tool_list_for_system_two())`.
3. `self.s2.generate_step(s2_prompt)`.
4. If `action == "finish_task"`, complete and record `SYSTEM_TWO_GENERATION` carrying `s2_response.tokens_used`.
5. **Schema Validation**: Both System 1 and System 2 arguments are strictly validated against `validate_tool_args`.
6. **Permission Gate**: The same permission broker gates the System 2 call (identical `requires_approval` / deny / record path), then `self.mcp.execute_tool(s2_action, s2_args)`.
7. **Outcome Verification**: `verify_step_outcome` runs on System 2 outcomes and updates plan progress.

### 3.8 Accounting and reconciliation

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

**Resolved — this section previously described a placeholder, and is now a real client.**

`MCPManager.attach_to_host` iterates `mcp_servers.json`, skips `disabled` entries, and for each
server spawns a **real subprocess**, performs the MCP `initialize` handshake, sends
`notifications/initialized`, calls `tools/list`, and registers **each discovered tool under its
real name, description and `inputSchema`**.

The protocol is implemented directly (newline-delimited JSON-RPC 2.0) rather than through the
`mcp` SDK. The reason is specific: the installed `mcp` is 2.2.0, where `FastMCP` was renamed
`MCPServer` and client signatures changed; the wire specification is the stable part, so an SDK
upgrade cannot silently break tool calls.

Formally:

| Behaviour | Before | Now |
|---|---|---|
| Process spawn | none | real subprocess, lazily on first use |
| Handshake / discovery | none | `initialize` + `tools/list` |
| Tool registration | one schema-less `mcp_<name>_dispatch` per server | one tool per discovered tool, with its real schema |
| Tool call | returned an f-string | forwarded over JSON-RPC; returns the server's actual result |
| Failed server | reported **success** | `Error:`-prefixed message naming the server and reason |
| Arg validation | none | reuses `validate_tool_args` from `dispatcher.py` |
| Lifecycle | none | `connect()` / `shutdown()` / `close(name)` / `atexit`, plus `health()` |

`mcp_<name>_dispatch` is retained only as a working forwarder taking `tool`/`arguments`.
Discovered tools are `requires_approval=True` and never overwrite a built-in — a server that
tries to redefine `run_shell_command` is renamed rather than allowed to shadow it.

`HAS_MCP` gates **HTTP transport only**; stdio needs no SDK, so stdio is not gated by it. An HTTP
server without the `mcp` package reports that dependency by name.

`bridges/mcp_evaluator_server.py` remains a real MCP server implementation — the thing being
served *to* other clients, not something this runtime connects to.

**Verified:** `tests/test_mcp_manager.py` drives a real stdio server
(`tests/fixtures/echo_mcp_server.py`) that computes its own results; a failed server produces a
named error, and the old success-looking placeholder string is asserted unreachable.

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
runs the goal, and updates `last_run_at` / `run_count` / `last_status` in a `finally`. A raise from
one job is contained per job inside `_tick`, so a broken entry cannot starve the rest of the pass.

**The scheduler runs inside every long-lived process, and nowhere else.** `GatewayRunner.run` starts
`asyncio.create_task(self.scheduler.run_forever())` (60s) alongside polling. `dual-agent --ui` starts
its own tick loop in the FastAPI lifespan (20s, in `web/server.py`), executed via
`run_in_executor` so the blocking SQLite/agent work cannot stall HTTP or WebSocket traffic, and
cancelled on shutdown. `/api/schedules` reports `scheduler_running` from that live loop rather than
asserting `true`, alongside `tick_seconds` and `last_tick_error`.

**Delivery is stdout only.** `_run_job` prints results to the host process's console; there is no
chat transport in the scheduler, so a job result is never pushed to a chat. `/api/schedules` returns
`delivery: "stdout"` and says so in its `note`. Additionally, the dashboard has no terminal attached,
so an interactive tool that requests approval fails with `EOF when reading a line` — unattended runs
need `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true` under `dual-agent --gateway`.

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
| GET | `/api/schedules` | `{jobs, scheduler_running, tick_seconds, last_tick_error, delivery, note}` |
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
  `SYSTEM_TWO_PROVIDER` or `config.json` specifies a real provider (`deepseek`, `grok`, `openai`, `custom`).
- Real S2 providers fail loudly if misconfigured or unreachable rather than quietly faking success.
- `anthropic` is not implemented; requests for it raise a clear `ValueError`.

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

- **External MCP client connection:** `MCPManager` fails loudly if an external MCP server is called,
  stating that external JSON-RPC connections are not yet supported, rather than returning fake success strings.
- `run_shell_command` cannot use pipes, redirects, globs, or `$VAR`, because it is argv-only by
  design. There is no allow-list or blocklist on the binary either — anything on `PATH` is
  reachable if the user approves the card.
- `validate_tool_args` rejects unknown keys and validates schema types on both fast and slow paths.
- `read_file` appends an explicit truncation marker `...[truncated N characters of M]` if the file
  exceeds 10,000 characters so downstream steps know the content is incomplete.

**Loop and termination**

- `max_steps` (default 15 in `run`, 10 from the CLI and gateway) is the only hard stop. There is no
  goal-decomposition step, so a goal needing more than `max_steps` sequential steps simply ends
  incomplete.
- Stall detection halts runs after 3 identical steps without progress. Stalled runs are excluded
  from skill synthesis and user profile recording.

**Memory and learning**

- **`token_savings_pct` is written as `NULL`**, and `get_aggregate_stats` returns `None` for the
  average, because no baseline run is performed. The shell renders `n/a (no baseline run)`. This is
  deliberate — there is no benchmark harness in this repo.
- `USER.md` is updated on completed runs with durable facts about the goal and tools used,
  accumulating across sessions and viewable via `/whoami`.
- `build_recall_context` is recomputed per step to incorporate actions and findings from earlier steps.
- `AgentConfig.auto_learn_skills` controls skill synthesis; `auto_scan_workspace` inspects project files
  and records context in memory.

**Scheduler**

- `_is_due` supports wildcards, steps (`*/N`), exact values, comma-separated lists, ranges, range steps,
  month names (`jan`-`dec`), and weekday names (`mon`-`sun`).
- `dual-agent ui` runs a background tick loop every 20s, in the lifespan, via `run_in_executor`.
  Scheduled jobs execute in that process; results are printed to its console and the run outcome is
  stored in `scheduled_jobs.last_status`. Nothing is delivered to a chat.

**Sessions**

- `SessionRouter` keeps at most `max_sessions` resident chats (default `DEFAULT_MAX_SESSIONS = 256`,
  override with `DUAL_AGENT_MAX_SESSIONS`), evicting least-recently-used first. `get_or_create` on an
  existing chat refreshes its recency, so a busy chat is not evicted for being oldest.
- Eviction drops only the in-memory dispatcher. The chat's on-disk session directory is left intact,
  so a returning chat reopens its own history. Isolation is preserved: each chat keeps its own
  dispatcher and its own `memory.db`.

**Gateway and dashboard**

- Telegram only. `GatewayAdapter` is an ABC with one implementation; `filters.TEXT & ~COMMAND` means
  commands, photos, documents, and voice are ignored.
- Gateway runs default to `max_steps=10`.
- The dashboard has **no authentication** and can execute tools as the user. Loopback-only by
  default, gated by an explicit opt-out for non-loopback binds.
- WebSocket `step` events are emitted via `step_callback` with the real path taken (`S1_FAST`, `S2_SLOW`, `TERMINAL`),
  including actual latency and token counts.
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
- `bridges/hermes_middleware.py` previously claimed, in its docstring, that offloading routing to Jev
  reduces outer-loop token costs by a fixed large percentage. Nothing in this repo ever measured that,
  and it contradicted the measured-only policy applied everywhere else. **Resolved:** the figure has
  been removed and the docstring now states plainly that no savings number is claimed because no
  baseline harness exists. `tests/test_recall_and_web_safety.py` fails the build if an unmeasured
  percentage claim reappears in any `.py` or `.md` file.
