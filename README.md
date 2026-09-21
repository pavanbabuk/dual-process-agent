# Dual-Process Agent (Turnkey AI Assistant)

A high-performance **Turnkey AI Assistant** combining **TypeSafe AI's Jev** ("System 1") with **Model Context Protocol (MCP)** tools, **System 2 LLMs** (Nous Research Hermes, xAI Grok, Anthropic Claude, OpenAI), and **Persistent SQLite Memory**.

---

## The Dual-Process Advantage

Traditional agents (Hermes, Grokbot, AutoGPT) use heavy generative LLMs for every micro-decision, burning tokens and taking 20–40s per task.

**Dual-Process Agent** splits execution into two cognitive systems:
1. **⚡ System 1 (TypeSafe AI Jev):** Sub-20ms reflex engine that routes tools, verifies safety policies, and checks loop exit conditions using typed primitives (`Choice`, `Noul`, `Score`). **0 generated LLM tokens.**
2. **🧠 System 2 (Hermes, Grok, Claude):** Deliberate reasoner invoked *only* when open-ended code generation, creative writing, or complex synthesis is required.

**Result:** **70–90% lower token bills** and **4x to 80x faster response times** than standard ReAct loops.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/charming-hopper.git
cd charming-hopper
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configuration Wizard

Run the interactive setup wizard to enter your keys:

```bash
dual-agent config
```

You will be prompted for:
* **TypeSafe AI API Key:** (from [console.typesafe.ai](https://console.typesafe.ai))
* **System 2 Reasoner Provider:** `[mock, hermes, grok, anthropic, openai]`
* Keys/URLs for your chosen System 2 model.

Configuration is saved in `~/.dual_agent/config.json`.

---

## Usage

### A. Interactive Conversational Shell (TUI)
Simply launch `dual-agent` with no arguments to start the interactive assistant:

```bash
dual-agent
```

```text
╭──────────────────────────────────────────────────────────────╮
│ Dual-Process Agent Shell (v1.0)                              │
│ System 1: TypeSafe AI Jev | System 2: GROK | Tools: 5        │
│ Type your instruction, or /help for available commands.      │
╰──────────────────────────────────────────────────────────────╯

dual-agent> /tools
dual-agent> /memory
dual-agent> Check pyproject.toml and find all missing test dependencies.
```

#### Slash Commands in the Shell:
| Command | Description |
| :--- | :--- |
| `/tools` | List all registered MCP tools and parameters |
| `/memory` | Inspect learned skills, session count, and cumulative token savings |
| `/config` | Run the interactive configuration wizard |
| `/clear` | Clear screen and reset conversation view |
| `/exit` | Exit the shell |

---

### B. Single-Shot Command Execution
Execute instructions directly from the terminal with live benchmarking telemetry:

```bash
# Fast-path repository inspection:
dual-agent "Inspect current directory and summarize python packages"

# Escalated task requiring code generation (Grok / Hermes):
dual-agent --goal "Write a Python script that scrapes Hacker News" --provider grok
```

---

## Architecture & Components

```
~/.dual_agent/
├── config.json          # User settings, API keys, and model preferences
├── memory.db            # Persistent SQLite database (sessions & learned skills)
└── mcp_servers.json     # External MCP server definitions (Claude Desktop compatible)
```

### 1. Persistent SQLite Memory (`src/dual_agent/memory.py`)
* **Session History:** Tracks all tasks, step traces, latency breakdowns, and token savings.
* **Learned Skills:** Automatically stores verified tool sequences from successful tasks. If a similar goal is requested, the routine replays via Jev in under 100 ms.
* **Project Context:** Remembers project tech stacks and user preferences across terminal sessions.

### 2. Model Context Protocol (MCP) Integration (`src/dual_agent/mcp_manager.py`)
Add any external MCP server (e.g. GitHub, PostgreSQL, Docker) in `~/.dual_agent/mcp_servers.json`:

```json
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
All tools are automatically registered into the agent and routed via Jev.

### 3. Integration Bridges (`bridges/`)
* **Hermes Agent Middleware (`bridges/hermes_middleware.py`):** Drop-in router interceptor for Nous Research Hermes Agent loops.
* **Jev MCP Evaluator Server (`bridges/mcp_evaluator_server.py`):** Standalone MCP server exposing Jev's `Score` and `Noul` to Claude Desktop, Cursor, or Antigravity.

---

## Test Suite

```bash
pytest -v --cov=dual_agent --cov=bridges
```

**24 passing unit and integration tests** covering SQLite persistence, MCP management, slash commands, edge cases, and end-to-end dispatching.
