"""Pytest configuration — keep the suite away from real state and real spend.

This file exists because its absence caused real damage, twice.

1. Data. Most tests construct a MemoryEngine without an explicit db_path, which
   resolves to ~/.dual_agent/memory.db — the developer's actual database. Running
   the suite repeatedly filled it with hundreds of junk sessions and grew an
   un-checkpointed WAL to 4MB. With two suites running at the same time (which
   happens whenever several agents work in parallel) the shared file was
   corrupted outright:

       sqlite3.DatabaseError: database disk image is malformed

   That blocked every subsequent write until the database was rebuilt. Pointing
   DUAL_AGENT_HOME at a per-run temp directory means the suite cannot touch real
   state, and two concurrent runs cannot collide because each gets its own.

2. Money. The library loads ./.env relative to the working directory, so running
   pytest from the repository root pulled the developer's LIVE TYPESAFE_API_KEY
   into the test process. Any test that built a dispatcher without
   force_simulation=True would then make genuine, billed Jev API calls. Removing
   the provider credentials here means a test cannot spend money even by mistake.
"""

import atexit
import os
import shutil
import tempfile

# --- isolate all persistent state -------------------------------------------
_session_home = tempfile.mkdtemp(prefix="dual-agent-tests-")
os.environ["DUAL_AGENT_HOME"] = _session_home

# --- make live API calls impossible -----------------------------------------
# Cleared rather than merely unused: getenv() is read at call time throughout
# the codebase, so an inherited key would reach the network on any code path
# that forgets force_simulation=True.
for _secret in (
    "TYPESAFE_API_KEY",
    "DEEPSEEK_API_KEY",
    "CUSTOM_LLM_API_KEY",
    "HERMES_API_KEY",
    "OPENAI_API_KEY",
    "GROK_API_KEY",
    "ANTHROPIC_API_KEY",
    "OMNIROUTE_API_KEY",
    "TELEGRAM_BOT_TOKEN",
):
    os.environ.pop(_secret, None)

# Deterministic, offline System 2 for every test that does not inject its own.
os.environ["SYSTEM_TWO_PROVIDER"] = "mock"
# A local endpoint that cannot resolve, so even an accidental call fails fast
# instead of hanging on a real network timeout.
os.environ["HERMES_BASE_URL"] = "http://127.0.0.1:1/v1"
os.environ["SYSTEM_TWO_TIMEOUT"] = "2"


@atexit.register
def _cleanup_session_home() -> None:
    shutil.rmtree(_session_home, ignore_errors=True)
