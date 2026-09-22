# Dual-Process Agent v2.0 (Turnkey AI Assistant)

A high-performance **Turnkey AI Assistant** combining **TypeSafe AI's Jev** (System 1) with **Model Context Protocol (MCP)** tools, **System 2 LLMs** (Nous Research Hermes, xAI Grok, Anthropic Claude, OpenAI), **Persistent SQLite Memory**, **Self-Improving Skills**, **FTS5 Cross-Session Recall**, **Interactive Permission Cards**, **Telegram Gateway**, and **Cron Scheduler**.

---

## Quick Install (Hermes-style one-liner)

### Linux, macOS, WSL2

```bash
bash install.sh
```

Or as a one-liner after hosting:
```bash
curl -fsSL https://your-host/install.sh | bash
```

Then:
```bash
source ~/.zshrc      # or ~/.bashrc
dual-agent config    # enter your API keys
dual-agent           # start chatting!
```

### Manual (existing workflow)

```bash
git clone https://github.com/pavanbabuk/dual-process-agent.git
cd dual-process-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
dual-agent config
dual-agent
```

---

## The Dual-Process Advantage

Traditional agents use a heavy generative LLM for every decision. **Dual-Process Agent** splits execution:

1. **⚡ System 1 (TypeSafe AI Jev):** Reflex engine that routes tools and checks safety — **0 generated tokens per decision.**
2. **🧠 System 2 (Hermes, Grok, Claude):** Deliberate reasoner invoked only when creative synthesis or complex reasoning is needed.

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

**Authorization is required.** The bot can write files and run shell commands on the machine hosting it, so it will only accept messages from ids listed in `TELEGRAM_ALLOWED_USER_IDS`. With that list empty, **every** message is rejected — the default is closed, not open. Set it before starting the daemon, and never leave a shell-capable bot reachable by whoever happens to find it.

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

**107 passing** unit and integration tests covering all modules — including regression tests for argument validation, shell-injection resistance, gateway authorization, and the telemetry-honesty guarantees described above.
