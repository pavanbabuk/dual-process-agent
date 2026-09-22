# Roadmap

Where this project is, and what has to be built to reach **Hermes-class** capability.

Read `docs/ARCHITECTURE.md` first — this document assumes it. Everything marked **Exists** below
was verified against the source; the file and function are cited so you can check.

**One rule governs this whole document:** this repo has no baseline harness, so no measured
performance improvement, token saving, or speedup is claimed anywhere. Where a benefit is expected
from a design choice, it is phrased as a design expectation and marked *unverified*. The predecessor
README claimed "70–90% lower token costs, 4–80× faster responses" computed from hardcoded constants
(1500 tokens / 1200 ms per step); that code was removed. Do not reintroduce a number that was not
measured.

---

## 1. What "Hermes-class" means, in checkable terms

Nine capabilities. Each is defined by something you can point at, not by an adjective.

| # | Capability | Definition (checkable) | Status today |
|---|---|---|---|
| 1 | **Persistent skills** | Procedures are written to disk, survive process restart, are parsed back, and measurably affect what the agent does next. | **Partial.** Files are written and re-parsed (`skills_manager.py` → `synthesize_skill`, `load_all_skills`) and injected into the System 2 prompt (`build_skill_context`). But the SQLite `find_matching_skill` replay path is logged only and never changes routing, so a skill file's presence does not yet change behaviour. |
| 2 | **Cross-session memory and recall** | Past runs are persisted, searchable by natural language, and injected into a later prompt. | **Exists.** `memory.py` — SQLite `sessions` + FTS5 `sessions_fts` with an AFTER INSERT trigger, `full_text_search` (with `LIKE` fallback), and `build_recall_context` prepended to the System 2 prompt. Surface: `/recall`, `GET /api/memory`. |
| 3 | **Scheduled / cron jobs** | A job can be registered in natural language, persists across restarts, and fires on schedule without a human present. | **Partial.** `parse_nl_to_cron` + `CronScheduler` (SQLite CRUD, 60s asyncio tick, 55s double-fire guard) work, but only inside `dual-agent --gateway`. `--ui` never ticks. Results are printed to a daemon's stdout, not delivered. |
| 4 | **Messaging gateway** | A user can drive the agent from a chat app, with authorization, and per-conversation isolation. | **Exists (one platform).** `gateway/runner.py` wires `TelegramAdapter` → `SessionRouter` → per-chat dispatcher with its own memory DB. Allow-list gate; empty list admits nobody. Telegram only; text messages only. |
| 5 | **MCP tool ecosystem** | Third-party MCP servers can be configured and their tools become callable, with real I/O. | **Does not exist.** `MCPManager.attach_to_host` registers a placeholder per server whose handler returns `f"Dispatched to external MCP server '{n}': {args}"`. No subprocess, no JSON-RPC, no tool discovery. `mcp>=2.0.0` is declared in `pyproject.toml` and never imported in `src/`. Only the 4 built-in tools do work. |
| 6 | **Multi-provider model routing** | Several model backends are selectable by config, and a provider failure is visible rather than silently substituted. | **Partial.** `get_system_two_provider` resolves hermes/ollama, grok/xai, openai/gpt. But `anthropic` is accepted by `--provider` and the wizard and is unimplemented (falls through to mock), and both real providers silently fall back to `MockSystemTwoProvider` on any exception, so an outage looks like a mock run. |
| 7 | **Self-update** | The tool can update itself and report what changed, verifiably. | **Weak.** `updater.perform_update` runs `git pull origin master` (or `main`) in the current working directory and prints the installed `typesafe_sdk` version. No reinstall, no integrity check, no changelog, no rollback. |
| 8 | **Permission gating** | Risky operations require explicit approval, decisions are audited, and unattended contexts default to deny. | **Exists.** `permission_broker.py` — Rich cards with Allow once / Allow session / Deny / Edit args, session-scoped allow/deny sets, `non_interactive` deny-by-default, and every decision written to `approval_audit`. The dashboard has its own WebSocket-backed broker (120s timeout → deny). |
| 9 | **Observability** | Per-step and per-run telemetry is recorded, is measured rather than modelled, and surfaces what was simulated. | **Partial.** `StepRecord` / `DispatchResult` carry measured latency, tokens, per-layer step counts, and explicit simulation flags; SQLite `sessions.steps_json` persists them; `GET /api/memory` exposes aggregates. Missing: no tracing, no metrics endpoint, no export, no dashboard for the approval audit, and WebSocket step events mislabel every step `path: "S1_FAST"`. |

### Scorecard

| Capability | Exists | Partial | Missing |
|---|---|---|---|
| Persistent skills | | ● | |
| Cross-session memory & recall | ● | | |
| Scheduled / cron jobs | | ● | |
| Messaging gateway | ● | | |
| MCP tool ecosystem | | | ● |
| Multi-provider model routing | | ● | |
| Self-update | | ● | |
| Permission gating | ● | | |
| Observability | | ● | |

Three of nine are done. The two structural gaps are **#5 (MCP)** and the fact that **#1's replay
path is inert** — both mean the system's advertised tool and learning surface is narrower than it
looks.

---

## 2. The cost model, with a source

Jev bills **input tokens only**. Output tokens are free. As of Jev 1.13 (`jev-1.13.0`), the price
is **$42 per billion input tokens** ($0.042 per million). Rate limits are 250,000 tokens/second and
1,200 requests/minute; context is 64k tokens per request (32k for `state` plus the longest
question). Jev is not fine-tuned on customer data.

Source: <https://docs.typesafe.ai/models>

TypeSafe's published benchmark for the model averages roughly **$0.0004 per routing decision**
(source: <https://docs.typesafe.ai/models>). Treat that as an input to planning, not as a property
of this runtime — it is the vendor's number for the model, and this repo has not measured its own.

### What this means for the architecture

The dispatcher makes **one routing decision per step**, and the confidence gate can make a
**second** call per step — `_compute_fast_path_confidence` calls
`_check_fast_path_confidence(rival_tool, ...)`, a real billed model call re-checking the runner-up.
So the honest picture is:

- **Cost scales with steps, not with answers.** A goal that needs 12 steps of routing costs roughly
  12 routing decisions — more when the confidence gate fires a verification call.
- **A stuck loop is a paid loop.** The stall detector exists precisely because of this. Its
  docstring: *"In live mode each of those steps is a paid Jev call, so a stuck loop costs real
  money to produce nothing."* Three consecutive identical `(tool, args, output[:200])` signatures
  end the run.
- **Reliable termination is therefore an economic requirement, not a nicety.** Every mechanism that
  ends a goal in fewer steps reduces Jev input tokens linearly. This is the single strongest
  argument for prioritising Milestone 2.
- **`max_steps` is an implicit spend cap.** `run` defaults to 15; the CLI and gateway default to 10.
  Nothing enforces a token or dollar budget.

At the published $0.0004/decision figure, a 10-step goal is on the order of $0.004 in routing cost.
That is the vendor's average, it is **unverified against this runtime**, and it excludes System 2
generation entirely — which, with a real provider, will dominate total cost. Any claim that this
architecture is cheaper than an LLM-per-step loop is a **design expectation requiring a baseline
harness to test** (Milestone 6). Do not assert it before that harness exists.

---

## 3. Milestones

Ordered by what unblocks the most. Each has acceptance criteria another engineer can run.

### M0 — Stop the bleeding (no new capability)

Small correctness fixes that make later milestones testable. Can ship together.

| Item | Where | Acceptance criterion |
|---|---|---|
| Dashboard cannot run live Jev | `web/server.py` hardcodes `force_simulation=True` | `create_app` honours config; a test asserts the client is live when a key is present |
| `anthropic` is a selectable non-provider | `cli.py`, `config.py` wizard, `system_two.py` | Either implemented, or removed from `choices` in both places |
| Silent mock fallback hides outages | `HermesProvider.generate_step`, `OpenAICompatibleProvider.generate_step` | Fallback records a `system_two_fallback_reason` on the step record; a test asserts a connection failure surfaces instead of reporting mock `tokens_used=450` |
| WebSocket step events always say `S1_FAST` | `web/streaming_dispatcher.py` `patched_execute` | A slow-path run emits at least one `path: "S2_SLOW"` event |
| `USER.md` is never written | `dispatcher.py` | Completing a run appends at least one fact via `memory.update_user_profile`; a test asserts the file grows |
| `find_matching_skill` is inert | `dispatcher.py` | Either the replay is wired into a step, or the log line and `LearnedSkill` claims are removed |
| `bridges/hermes_middleware.py` claimed an unmeasured savings percentage | docstring | ✅ **Done** — the figure was removed and replaced with an explicit statement that no savings number is claimed; `tests/test_recall_and_web_safety.py` now fails the build if such a claim reappears |

**Why first:** three of these are honesty defects in the same class as the constants that were
already removed. Leaving them makes every later milestone harder to verify.

---

### M1 — A real System 2 provider path

**Priority: highest of any capability work. The agent cannot generate anything today.**
`MockSystemTwoProvider.generate_step` returns one of two canned branches and reports
`tokens_used=450` for both. Until this is fixed, nothing downstream is exercisable end to end.

Scope:

- Make one real provider work first end to end. `HermesProvider` (local Ollama/vLLM, no API key) is
  the cheapest target for local iteration; `OpenAICompatibleProvider` covers Grok/OpenAI.
- Implement `anthropic` properly (`/v1/messages`, not the OpenAI chat shape) or drop it.
- Introduce a single `SystemTwoResult` that carries a `fallback_reason` and a `provider` field so a
  mock response can never be recorded as a real generation.
- Parse the model's JSON defensively. Today `json.loads(raw_content)` on the whole content means any
  prose around the JSON raises `JSONDecodeError` and silently degrades to mock.

**Acceptance criteria**

1. With `SYSTEM_TWO_PROVIDER=hermes` and a local Ollama serving the configured model, a goal that
   forces escalation of *"write a haiku about SQLite"* returns text not present in
   `system_two.py`.
2. The final `StepRecord` has `tokens_used` equal to the provider's reported usage, **not** the
   constant `450`.
3. With Ollama stopped, the run reports a visible provider failure; a test asserts
   `system_two_fallback_reason` is populated and non-null.
4. A provider returning JSON with a markdown code fence still parses (or fails loudly, not silently).

---

### M2 — Reliable loop termination for multi-step goals

**Priority: second highest. Cost scales with steps, so this is where the money is.**

Today, termination is `decision.is_terminal` from the same router call that does routing, plus a
3-identical-step stall check and `max_steps`. There is no goal decomposition, no progress model, and
no state carried between steps (`AgentState.variables` is written by nothing in the dispatcher).

Scope:

- **Represent state transitions.** Give `AgentState.variables` a writer and feed a compact
  `variables` block into `to_system_one_state` (which already prints `VARIABLES: []`). This is the
  prerequisite for any multi-step workflow — today a goal needing "fetch, then transform, then
  write" has no way to express "I am on part 2 of 3".
- **Decompose the goal once.** A single System 2 call on step 1 that produces an ordered checklist
  of sub-goals, stored in `variables`, gives both layers something to check off. Independent
  *design expectation*: this should reduce the number of routing decisions per completed goal
  because steps stop being re-derived from the raw goal text each time. **Unverified** — it needs
  the M6 harness to test.
- **A progress predicate distinct from the router.** Termination should require positive evidence
  of completion (the sub-goal checklist in `variables`), not only one model's `is_terminal` flag on
  the step it happens to also be asked to route.
- **Widen stall detection.** Current signature is `(tool_name, repr(args), output[:200])`, so an
  A-B-A-B oscillation is never caught. Add a cycle detector over recent signatures.
- **Make `max_steps` an explicit budget**, reported in `DispatchResult` (`steps_budget`,
  `steps_used`) so an incomplete run is visibly truncated rather than merely `is_completed=False`.

**Acceptance criteria**

1. A 3-part goal (*"list the Python files here, count their total lines, then write the count to
   linecount.txt"*) completes with 3 distinct tool actions and no repeated identical signature.
2. On such a run, `state.variables` is non-empty at completion and `to_system_one_state` includes its
   keys.
3. An oscillating mock (`read_file` → `list_directory` → `read_file` → …) terminates early via the
   cycle detector in < `max_steps` steps.
4. Same goal, same seed, no live provider: run twice and get the same step count.

---

### M3 — Real MCP tool ecosystem

**Priority: third. This is the difference between "4 hardcoded tools" and an agent platform.**

`MCPManager.attach_to_host` currently registers a placeholder. The `mcp` package is declared and
never imported.

Scope:

- Use the official `mcp` Python SDK to spawn each configured server from
  `mcp_servers.json` (`command`, `args`, `env` — already parsed into `MCPServerEntry`).
- On connect, call `tools/list` and register **one `MCPToolDefinition` per remote tool**, with the
  server's real JSON Schema as `parameters_schema`. This makes the existing `validate_tool_args`
  work on remote tools for free.
- Implement `execute_tool` dispatch over a stdio `ClientSession`, with per-server lifecycle
  (connect on first use, restart on crash, close on shutdown).
- Namespace tool names as `<server>.<tool>` to avoid collisions with the 4 built-ins, and make the
  description shown to System 1 include the server name.
- Default the *resolved* MCP tool set to `requires_approval=True` / `risk_level="high"` unless the
  server is explicitly marked trusted in config — a remote server can do anything its process can.
- Delete or clearly rewrite the `mcp_<name>_dispatch` placeholder path.

**Acceptance criteria**

1. Configure the `filesystem` reference server in `~/.dual_agent/mcp_servers.json`; `/tools` lists
   its real tools with real descriptions, not one `mcp_<name>_dispatch` entry.
2. `GET /api/tools` includes them with their true `parameters_schema`.
3. A goal that routes to a remote tool returns the server's actual output.
4. A server whose command does not exist produces a clear connection error and the other servers
   still load.
5. `validate_tool_args` refuses a call with a missing required argument of a remote tool.
6. A test asserts `mcp` is imported and that no handler returns the string "Dispatched to external
   MCP server".

---

### M4 — Streaming responses

**Priority: fourth. Mostly a UX gap, but it also removes the 120s permission ceiling constraint.**

Nothing streams today. Both providers post with `stream` unset, and the dashboard's "streaming" is
step-level events emitted from a monkey-patched `execute_tool`.

Scope:

- Add `stream=True` support to the System 2 providers and an incremental callback, so partial text
  reaches the surface as it arrives.
- Add token/partial events to the WebSocket protocol (`stream_delta` with a `run_id`) and render
  them in `dashboard.html`. Existing event types stay for compatibility.
- Correlate events to a `run_id` so overlapping `run` messages stop interleaving on one socket.
- Update `step` events so `path` reflects the real path, and emit them from the dispatcher rather
  than from a monkey-patched `execute_tool`.

**Acceptance criteria**

1. A long generation arrives as multiple `stream_delta` events, and their concatenation equals the
   final `done.final_output`.
2. `dashboard.html` renders text progressively (verifiable by a slow mock provider emitting chunks
   with delays).
3. Two concurrent `run` messages produce events that can be separated by `run_id`.
4. A slow-path step emits `path: "S2_SLOW"`.

---

### M5 — Packaging and distribution

**Priority: fifth, but blocking any adoption.**

Install is `bash install.sh`. There is no Dockerfile, no CI, no release workflow, and no published
package. `pyproject.toml` pins `typesafe-sdk>=0.7.0`, `mcp>=2.0.0`, `pydantic>=2.10.0`,
`httpx>=0.27.0`, `rich>=13.0.0`, with `gateway`/`ui`/`all` extras, and maps `bridges` to the repo
root via `[tool.setuptools.package-dir]`.

Scope:

- A GitHub Actions workflow running the suite on Python 3.11 and 3.14 (the range `pyproject.toml`
  claims, and the range the existing regression tests target).
- Publish `dual-agent` to PyPI so `pipx install dual-agent` works, with the `[all]` extra.
- A `Dockerfile` for the gateway and the dashboard.
- Pin the Jev model ID rather than floating on an alias. TypeSafe's own docs warn that an alias
  moves when a new release ships, and that confidence thresholds tuned against one version should
  pin that version's ID.

**Acceptance criteria**

1. `pipx install dual-agent` then `dual-agent --help` exits 0 on a clean machine.
2. CI is green on 3.11 and 3.14.
3. `docker run` of the gateway image starts, logs the preflight, and exits non-zero with the
   documented message when `TELEGRAM_BOT_TOKEN` is absent.
4. The Jev model is configurable and defaults to a versioned ID, not `jev-latest`.

---

### M6 — The baseline harness (prerequisite for any performance claim)

**Not optional if a performance number is ever to be published.**

Scope:

- An independent runner that executes the *identical goal* through a plain single-model agent loop —
  one LLM call per step, no router — on the same machine and the same tool handlers.
- Both runs persist to SQLite: steps, latency, tokens per layer, and a completion flag.
- `memory.sessions.token_savings_pct` starts being populated **from a real comparison** rather than
  always `NULL`. `get_aggregate_stats` already returns `None` rather than `0.0` when nothing is
  measured, so the plumbing is ready; `DispatchResult` deliberately has no savings field, and
  `tests/test_end_to_end.py::test_dual_process_run_reports_measured_values_only` asserts it stays
  that way. Any new field must be derived from the harness, not from constants.
- Report distributions, not best cases: median and spread over N ≥ 30 goals, both completion rates,
  and failures reported as failures.

**Acceptance criteria**

1. Running one goal through both paths writes one baseline row and one dual-process row sharing a
   `goal_key`.
2. `token_savings_pct` is non-NULL and equals `(baseline_tokens - actual_tokens) / baseline_tokens`
   for that pair.
3. The harness is a separate entry point and cannot run as part of `dual-agent` normal operation.
4. Documented results include the harness invocation, the goal set, and the raw rows.

**Until M6 exists, this project publishes no savings, speedup, or token-reduction figure.**

---

### M7 — Breadth, after the core is honest

Deliberately last. Each is real work but none unblocks the others.

| Item | Detail |
|---|---|
| Second gateway | A second `GatewayAdapter` (Discord or Slack) to prove the ABC is genuinely platform-agnostic. Handle non-text message types. |
| Scheduler delivery | Send job results to a chat via the gateway; drop the stdout `Console` output. Tick the scheduler under `--ui` too. |
| Fuller cron | Lists, ranges and names in `_is_due`; or delegate to `croniter`. |
| `USER.md` that is actually maintained | Derive preference facts automatically and use them in the System 2 prompt. |
| Skill replay that fires | Make `find_matching_skill` a real fast path: a matched routine proposes its `tool_sequence` instead of one model-chosen tool. |
| Approval-audit surface | Expose `approval_audit` in the dashboard and as an export. |
| Per-chat run lock | Serialise concurrent runs in one `SessionRouter` chat; `_current_goal` is single-valued today. |
| Session eviction | `SessionRouter._sessions` never shrinks and `destroy` is never called. |

---

## 4. Non-goals

Things this project will not do, so effort and review attention stay where they belong.

1. **No performance claims without a baseline.** No savings percentage, speedup multiple, or
   token-reduction figure may appear in code, docs, CLI output, or a chat reply unless M6 produced
   it from a real paired run. `DispatchResult` has no savings field by design and should keep none.
2. **Not a general-purpose LLM.** Jev is a decision model — `Choice`, `Noul`, `Score`. It does not
   generate text, and nothing here will ask it to. Generation is System 2's entire job.
3. **Not a model-training or fine-tuning project.** Jev is not fine-tuned on customer data. Domain
   behaviour is shaped through `state`, `instructions`, and `criteria` in the request.
4. **Not a hosted multi-tenant service.** No accounts, no billing, no per-user quotas. The gateway
   is single-operator with an explicit allow-list, and the dashboard is loopback-only with no auth.
5. **Not a sandbox.** `run_shell_command` enforces argv-only execution and the permission card, and
   that is the security boundary. There is no container, seccomp profile, or filesystem jail for
   approved commands, and adding one is out of scope — approve accordingly.
6. **No fabricated or simulated benchmark output.** `used_simulated_system_one`,
   `system_one_fallback_reason` and `simulated_latency_ms` exist so simulated runs are labelled. A
   simulated run's numbers must never be presented as Jev measurements. The `time.sleep(0.012)` in
   `_call_simulated_jev` is a simulation artifact and stays documented as one.
7. **Not a RAG or vector-search system.** Recall is SQLite FTS5 keyword search with a `LIKE`
   fallback. Embeddings, vector stores, and rerankers are out of scope; do not describe the current
   recall as semantic.
8. **Not a web-scraping or browser-automation framework.** No browser tools are planned as built-ins.
   If that capability is wanted, it belongs in an MCP server (M3).
9. **No multi-agent orchestration.** One dispatcher, one loop. Sub-agents, handoffs, and parallel
   planners are not on this roadmap.
10. **No language other than Python.** No Node/Go/Rust rewrite, and no second SDK surface. The
    `mcp` SDK and `typesafe-sdk` are the only protocol clients.

---

## 5. Execution order

```
M0  honesty fixes              ──┐
M1  real System 2 provider     ──┼─► unblocks everything that needs generation
M2  reliable termination       ──┘   (and is where routing cost is actually controlled)
M3  real MCP tools
M4  streaming
M5  packaging / distribution
M6  baseline harness           ──┐
M7  breadth                    ──┴─► only after M6 gates any performance claim
```

Rationale, stated once:

- **M1 before M2.** Termination cannot be tested properly against a provider that returns one of two
  fixed strings.
- **M2 before M3.** Cost is proportional to steps, so the mechanism that reduces steps is worth more
  than the mechanism that adds tools. It is also the cheapest way to reduce Jev input tokens —
  fewer decisions, not cheaper decisions.
- **M6 before any published number.** Anything else is the mistake this repo already made once.
- **M7 last.** Breadth on top of an unverified core compounds the verification debt.
