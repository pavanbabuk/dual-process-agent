# Dual-Process Agent (`dual-agent`)

A high-performance **Turnkey AI Assistant** combining **TypeSafe AI's Jev** (System 1) with **Model Context Protocol (MCP)** tools, **System 2 LLMs** (DeepSeek V3/R1, xAI Grok, OpenAI, Local Ollama/vLLM), **Persistent SQLite Memory**, **Self-Improving Skills**, **FTS5 Cross-Session Recall**, **Interactive Permission Cards**, **Telegram Gateway**, and **Cron Scheduler**.

> **Architecture:** Sub-20ms reflex routing (System 1) handles deterministic tool execution. Complex reasoning escalates to System 2 LLMs only when needed.

---

## Quick Install (One-Liner)

```bash
curl -fsSL https://raw.githubusercontent.com/pavanbabuk/dual-process-agent/master/install.sh | bash
```
*(Or install locally with `pip install -e .[all]`)*

---

## Key Features

1. **⚡ System 1 Reflex Engine (TypeSafe AI Jev):** Evaluates goals sub-20ms. Runs fast-path tool execution without waiting for LLM generation.
2. **🧠 System 2 Reasoner (DeepSeek V3/R1, Grok, OpenAI, Custom LLM):** Deliberate reasoner invoked only when creative synthesis or complex reasoning is needed. Includes chain-of-thought extraction for DeepSeek R1.
3. **⚙ Live Web UI & Settings Dashboard:** Launch with `dual-agent --ui` at `http://localhost:7860` to configure API keys, switch models live, and view real-time execution step traces.
4. **🔐 Interactive Permission Broker:** Renders inline approval cards in terminal and Web UI (`[Allow Once] [Allow Session] [Deny] [Edit]`).
5. **📚 Autonomous Skill Synthesis:** Synthesizes portable `.SKILL.md` files after every successful task execution.
6. **🔍 FTS5 Cross-Session Memory:** Full-text search over past sessions and automatically updated `USER.md` profile.
7. **⏰ Natural Language Scheduler:** Natural language cron scheduling (e.g. `/schedule "every day at 9am"`).
8. **🌐 Omni-Channel Gateway:** Background daemon for Telegram integration (`dual-agent --gateway`).

**On the performance claims.** Earlier revisions of this README advertised *"70–90% lower token costs, 4–80× faster responses"* and the CLI printed a matching `Traditional LLM Baseline` column. Those numbers were **not measured**: they came from multiplying the step count by hardcoded constants (1500 tokens / 1200 ms per step) and reporting the difference. That code is gone, because a benchmark you compute from a constant you chose is not a benchmark.

What the runtime reports today is measured only — steps, real latency, real token counts — plus an explicit flag when System 1 was a local stub rather than Jev. A genuine savings figure requires running the same goal through a plain single-model agent loop and diffing the results; that baseline runner is not implemented, so **no savings percentage is printed**. If you want to publish a number, build that runner first.

The architectural claim still stands on its own: routing and termination checks cost 0 generated tokens, so a step that System 1 handles does not pay for an LLM round-trip.

---

## What's New in v2.0 (Hermes + OpenMausBot Features)

| Feature | Inspired By | How to Use |
|---|---|---|
| **Self-Improving `.SKILL.md` Files** | Hermes Agent | Auto-synthesized after each task. Run `/skills` |
| **FTS5 Cross-Session Recall** | Hermes Agent | `/recall git` searches all past sessions |
| **USER.md User Profile** | Hermes Agent | `/whoami` — growing model of your preferences |
| **Interactive Permission Cards** | OpenMausBot | `write_file` and `run_shell_command` ask before executing |
| **Natural Language Cron Scheduler** | Hermes Agent | `/schedule "every day at 9am"` |
| **Portable Team Manifests** | OpenMausBot | `/export` → share your full setup as a `.md` file |
| **Telegram Gateway** | Hermes Agent | `dual-agent --gateway` for 24/7 bot |

---

## All Shell Commands

```
╭──────────────────────────────────────────────────────────────────╮
│ Dual-Process Agent Shell (v2.0)                                  │
│ System 1: TypeSafe AI Jev | System 2: GROK | Tools: 5           │
│ Type your instruction, or /help for available commands.          │
╰──────────────────────────────────────────────────────────────────╯
```

| Command | Description |
|---|---|
| `/tools` | List all registered MCP tools |
| `/memory` | Performance stats and SQLite skill cache |
| `/skills` | List self-synthesized `.SKILL.md` procedural files |
| `/recall <query>` | Full-text search across past sessions (FTS5) |
| `/whoami` | Display your auto-maintained USER.md profile |
| `/schedule <desc>` | Schedule a recurring task (natural language) |
| `/schedules` | List all scheduled cron jobs |
| `/export` | Export agent config as a portable Markdown manifest |
| `/import <path\|url>` | Import a team manifest from disk or GitHub URL |
| `/config` | Re-run the interactive configuration wizard |
| `/update` | Pull latest updates |
| `/clear` | Clear screen |
| `/exit` | Quit |

---

## Architecture

```
~/.dual_agent/
├── config.json           # Provider keys and preferences
├── memory.db             # Sessions + FTS5 + skills + scheduled_jobs + audit_log
├── mcp_servers.json      # External MCP server registry (Claude Desktop compatible)
├── skills/               # .SKILL.md procedural memory files (agentskills.io standard)
│   ├── inspect-pyproject.SKILL.md
│   └── write-pytest-test.SKILL.md
└── USER.md               # Auto-maintained user profile (Hermes-style)

src/dual_agent/
├── memory.py             # SQLite persistence + FTS5 + USER.md
├── dispatcher.py         # Dual-process orchestrator with skills + recall + broker
├── shell.py              # Interactive TUI REPL
├── mcp_host.py           # MCP tool registry (with requires_approval / risk_level)
├── skills_manager.py     # .SKILL.md synthesis and lookup
├── permission_broker.py  # Interactive [Allow/Deny/Edit] approval cards
├── scheduler.py          # Hermes-style NL cron scheduler (asyncio)
├── team_manifest.py      # Export/Import portable team manifests
└── gateway/              # Telegram gateway (Hermes-style omni-channel)
    ├── base.py
    ├── telegram_adapter.py
    └── session_router.py
```

---

## Permission Broker (OpenMausBot-style)

When the agent attempts a risky operation (`run_shell_command`, `write_file`), an inline approval card appears:

```
╭─ 🔐 Permission Required ──────────────────────────────╮
│ Tool:    run_shell_command                             │
│ Risk:    HIGH                                         │
│ Args:                                                 │
│   command: rm -rf build/                              │
╰───────────────────────────────────────────────────────╯
  [A] Allow once  [S] Allow for session  [D] Deny  [E] Edit args
```

Set `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true` in `.env` to skip prompts in CI.

---

## Telegram Gateway

```bash
# Install gateway dependencies
pip install "dual-agent[gateway]"

# Set your token in .env, plus the user ids allowed to talk to the bot
echo "TELEGRAM_BOT_TOKEN=<your-token>" >> ~/.dual_agent/.env
echo "TELEGRAM_ALLOWED_USER_IDS=<your-telegram-user-id>" >> ~/.dual_agent/.env

# Start the gateway daemon
dual-agent --gateway
```

The gateway routes each Telegram chat to an isolated agent session with separate memory.

Running the gateway also starts the **scheduler daemon** (ticking every 60s). The dashboard starts its own tick loop (every 20s) for as long as the UI is open, so `/schedule` jobs fire in either process; the interactive shell alone runs neither.

**Scheduled job output goes to the console of whichever process runs it, and is never sent to a chat.** A job result is printed to stdout and its outcome recorded in `scheduled_jobs.last_status`; nothing is pushed to Telegram. `/api/schedules` returns `delivery: "stdout"` so the dashboard does not imply a notification that never arrives. Also, the dashboard has no terminal behind it, so a scheduled job that needs approval fails with `EOF when reading a line` — for unattended runs use `dual-agent gateway` with `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true`.

**Authorization is required.** The bot can write files and run shell commands on the machine hosting it, so it will only accept messages from ids listed in `TELEGRAM_ALLOWED_USER_IDS`. With that list empty, **every** message is rejected — the default is closed, not open. Set it before starting the daemon, and never leave a shell-capable bot reachable by whoever happens to find it.

**Risky tools are denied in gateway mode** unless you set `DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=true`. There is no terminal behind a chat to answer an approval card, so `write_file` and `run_shell_command` are refused rather than permitted silently.

Configuration is read from `.env` (checked at `~/.dual_agent/.env` and `./.env`); real environment variables take precedence over the file.

---

## Screen Control & Desktop Interaction

Dual-Process Agent can perceive the graphical desktop and execute verified mouse and keyboard actions:

```bash
# Install screen control extra (Pillow + PyObjC Quartz)
pip install "dual-agent[screen]"
```

### Perception & Actuation Tools
- `screenshot`: Captures the display, dynamically calibrating the Retina scale factor (`capture_pixels / logical_points`, e.g. 1.336 or 2.0).
- `grid_overlay`: Annotates screenshots with a labeled coordinate grid overlay (Set-of-Mark) to guide visual targeting.
- `screen_diff`: Pixel-level visual difference comparison between screenshots. Fast, local, and requires no model calls.
- `mouse_click`, `mouse_move`, `key_press`: Actuates clicks, moves, and keystrokes at logical coordinates. All actuation tools are classified as **HIGH RISK** and require approval via the `PermissionBroker`.

### Safety & Permission Safeguards
- **Emergency Kill Switch**: Set `DUAL_AGENT_SCREEN_CONTROL=0` to immediately disable all actuation tools.
- **Single Checkpoint**: All actuation paths (fast-path reflexes and deliberate System 2 planning) pass through schema validation and permission broker gating.
- **Physical Outcome Verification**: The agent never assumes an action succeeded from model assertions alone. `screen_diff` checks whether the screen physically changed; if no change is detected after a click, verification fails and auto-correction guidance is triggered.
- **macOS Permissions**: Requires macOS Screen Recording permission (for display capture) and Accessibility permission (in System Settings -> Privacy & Security -> Accessibility) for sending input events.

### Vision Models & Cost Prerequisite

**The screen agent is not free.** Screen control has two halves — actuation (which is local and costs nothing) and *sight* (which requires a multimodal model and does cost money or local hardware). Without the second half the agent is blind, and a blind run is reported as a failure rather than presented as an observation.

**What was measured on this machine (2026-09-22), not assumed:**

- A real capture on this machine returned **3420x2224** pixels. `get_display_geometry()` reported the logical size as **1710x1112**, giving a measured scale factor of **2.0**; the 2560x1664 logical figure and ≈1.336 scale recorded earlier in this project's notes did not reproduce in this run. The scale factor is read from the live display at capture time rather than assumed, which is why the two differ.
- The provider configured here is `deepseek` / `deepseek-chat`, served as **`deepseek-flash`**. It is **not** text-only. It correctly named a solid red frame `Red`, a solid blue frame `Blue`, and a solid green frame `Green`; with no textual question at all it described the image in prose. The hard-coded "text-only" rejection that used to live in `DeepSeekProvider.generate_step` was therefore **stale for this endpoint** and has been removed: it discarded images the endpoint accepts and substituted canned text.
- The same model answered **`White`** for a **pure black** frame. So it has a vision tower and it is **not reliable**: treat any single frame reading as evidence, never as ground truth. Verify with `screen_diff` after acting.
- **No local VLM is installed.** Ports 11434 (Ollama), 1234 (LM Studio) and 8080 (vLLM) are all closed. A local VLM would remove the per-call cost but is not present here.
- **Rate limits are real but were not where this repo said.** Direct calls to `api.deepseek.com` returned HTTP 200 on 12/12 consecutive vision requests — no 429. The 429s come from the **local OmniRoute proxy on `localhost:20128`**: requests first return `504 RATE_LIMIT_EXECUTION_TIMEOUT` (OmniRoute's own 15 s queue deadline, not an upstream timeout), then `429 model_cooldown` ("All credentials for model gemini-3.7-flash are cooling down", `reset_seconds: 56`). A 429 is a quota failure and is reported as one — it is never a screen reading.

**Consequences, stated plainly:**

1. A screen run needs either a **paid vision model** (OpenAI `gpt-4o`, xAI `grok-2-vision-1212`) or a **locally installed VLM** (Ollama `llava`/`qwen2-vl`, vLLM). Both are configuration work you have to do; neither is present by default.
2. **Grid overlays do not eliminate misses.** `grid_overlay` is a Set-of-Mark coordinate aid: it measurably helps a vision model land near a target, and it does not guarantee a hit. Coordinate errors remain. Nothing here claims an accuracy figure, because none was measured.
3. A run with vision disabled **aborts with a named precondition** instead of falling back to the mock. `get_vision_provider()` returns a `BlockedVisionProvider` carrying the specific unmet requirement, and the dispatcher stops the run on it. Canned text is never returned as a screen reading.
4. Screenshots are **downscaled to a 1400 px long edge** before transmission (a 3420x2224 capture → 1400x910). The exact sent dimensions and byte size are recorded in `SystemTwoResponse.metadata["images_sent"]`, so cost is auditable per step instead of estimated.

Configure a vision model:

```bash
# Hosted (paid)
VISION_PROVIDER=openai  VISION_MODEL=gpt-4o                 VISION_API_KEY=sk-...   dual-agent --goal "..."
VISION_PROVIDER=grok    VISION_MODEL=grok-2-vision-1212    VISION_API_KEY=xai-...  dual-agent --goal "..."

# Local VLM (no per-call cost, requires the model to be pulled first)
VISION_PROVIDER=custom  VISION_MODEL=qwen2-vl  VISION_BASE_URL=http://localhost:11434/v1 dual-agent --goal "..."
```

`DUAL_AGENT_VISION_ENABLED=1` (or `vision_enabled: true` in `~/.dual_agent/config.json`) is required for any of these to take effect.

---

## Team Manifests (OpenMausBot-style)

Export your full configuration:
```bash
dual-agent /export     # Saves to ~/.dual_agent/team_manifest_<timestamp>.md
```

Import on a new machine:
```bash
dual-agent /import ~/.dual_agent/team_manifest_20260921.md
# Or from GitHub:
dual-agent /import https://raw.githubusercontent.com/user/repo/main/team.md
```

---

## Scheduling (Hermes-style)

```bash
dual-agent> /schedule "Run git status and send a report every day at 9am"
✓ Scheduled Job #1 added!
Description: Run git status and send a report every day at 9am
Cron: 0 9 * * *

dual-agent> /schedules
ID  Description                           Cron        Runs  Last Run  Enabled
1   Run git status every day at 9am       0 9 * * *   0     Never     ✅
```

---

## MCP Integration

```json
// ~/.dual_agent/mcp_servers.json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token"}
    }
  }
}
```

---

## Test Suite

```bash
pytest -v --cov=dual_agent --cov=bridges
```

**353 passing** unit and integration tests on Python 3.11 — including regression tests for argument validation, shell-injection resistance, gateway authorization, telemetry honesty, `.env` loading, stall detection, and vision-provider honesty (multimodal request shape, downscale dimensions, 429 reporting) asserted against a real local HTTP server rather than a mock.
