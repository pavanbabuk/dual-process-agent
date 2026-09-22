#!/usr/bin/env bash
# Dual-Process Agent — Zero-dependency one-liner installer (Hermes-style)
# Usage: curl -fsSL https://your-host/install.sh | bash
#   or:  bash install.sh

set -euo pipefail

REPO="https://github.com/pavanbabuk/dual-process-agent"
PACKAGE="dual-agent"
MIN_PYTHON="3.11"
DATA_DIR="${DUAL_AGENT_HOME:-$HOME/.dual_agent}"
BIN_DIR="${HOME}/.local/bin"

# ── Colors ─────────────────────────────────────────────────────────────────
BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[0;33m'; RED='\033[0;31m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[dual-agent]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*" >&2; exit 1; }

# ── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}  ╔══════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}  ║   Dual-Process Agent Installer v2.0  ║${RESET}"
echo -e "${BOLD}${CYAN}  ╚══════════════════════════════════════╝${RESET}"
echo ""

# ── OS detection ────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "${OS}" in
  Linux*)   PLATFORM="linux" ;;
  Darwin*)  PLATFORM="macos" ;;
  *)        error "Unsupported OS: ${OS}. Try WSL2 on Windows." ;;
esac
info "Platform: ${PLATFORM}"

# ── Python check ────────────────────────────────────────────────────────────
find_python() {
  for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
      version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
      major=${version%%.*}; minor=${version##*.}
      if [[ "$major" -ge 3 && "$minor" -ge 11 ]]; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON=$(find_python || true)
if [[ -z "$PYTHON" ]]; then
  error "Python ${MIN_PYTHON}+ not found. Install from https://python.org and re-run."
fi
success "Python: $($PYTHON --version)"

# ── Install uv (Astral's fast package manager) ──────────────────────────────
if ! command -v uv &>/dev/null; then
  info "Installing uv (Rust-based package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
success "uv: $(uv --version)"

# ── Clone or update repo ────────────────────────────────────────────────────
INSTALL_DIR="${HOME}/.dual_agent_src"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  info "Updating existing installation..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  info "Cloning ${REPO}..."
  git clone --depth=1 "${REPO}" "${INSTALL_DIR}"
fi

# ── Install the package via uv ──────────────────────────────────────────────
# NOTE: no --system. Installing into the user's system environment can shadow or
# overwrite OS-managed packages; the console script lands in ~/.local/bin either
# way, which is already on PATH below.
info "Installing Dual-Process Agent (this may take 30-60s)..."
uv tool install --force "${INSTALL_DIR}" 2>/dev/null \
  || uv pip install --user -e "${INSTALL_DIR}[all]" 2>/dev/null \
  || uv pip install --user -e "${INSTALL_DIR}" 2>/dev/null \
  || $PYTHON -m pip install --user -e "${INSTALL_DIR}" --quiet

success "Package installed"

# ── Create data directory and .env template ─────────────────────────────────
mkdir -p "${DATA_DIR}" && chmod 700 "${DATA_DIR}"

ENV_FILE="${DATA_DIR}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cat > "$ENV_FILE" <<'EOF'
# Dual-Process Agent Configuration
# Edit this file and then run: dual-agent config

TYPESAFE_API_KEY=your_typesafe_api_key_here
TYPESAFE_BASE_URL=https://api.typesafe.ai

SYSTEM_TWO_PROVIDER=mock
# Options: mock | hermes_ollama | grok | anthropic | openai

HERMES_BASE_URL=http://localhost:11434/v1
HERMES_MODEL=nous-hermes-3-llama-3.1-8b

GROK_API_KEY=your_xai_api_key_here
GROK_MODEL=grok-2-latest

ANTHROPIC_API_KEY=your_anthropic_api_key_here

OPENAI_API_KEY=your_openai_api_key_here

SYSTEM_ONE_CONFIDENCE_THRESHOLD=0.85

# Gateway (Telegram)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
# REQUIRED for the gateway to accept anything: comma-separated Telegram user ids.
# The bot runs shell commands and writes files, so with this empty it rejects
# every message rather than trusting whoever finds the bot.
TELEGRAM_ALLOWED_USER_IDS=

# Set to true to skip permission approval cards (useful for CI)
DUAL_AGENT_AUTO_ALLOW_PERMISSIONS=false
EOF
  success "Created config template at ${ENV_FILE}"
else
  info "Config already exists at ${ENV_FILE} — skipping."
fi

# ── Shell rc integration ─────────────────────────────────────────────────────
add_to_rc() {
  local rc_file="$1"
  local line="export PATH=\"\$HOME/.local/bin:\$PATH\""
  if [[ -f "$rc_file" ]] && ! grep -q "dual.agent" "$rc_file" 2>/dev/null; then
    echo "" >> "$rc_file"
    echo "# Dual-Process Agent" >> "$rc_file"
    echo "$line" >> "$rc_file"
  fi
}

[[ "${PLATFORM}" == "macos" ]] && add_to_rc "${HOME}/.zshrc"
add_to_rc "${HOME}/.bashrc"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ✅  Installation complete!${RESET}"
echo ""
echo -e "  Next steps:"
echo -e "    ${CYAN}source ~/.zshrc${RESET}   (or open a new terminal)"
echo -e "    ${CYAN}dual-agent config${RESET}  (enter your API keys)"
echo -e "    ${CYAN}dual-agent${RESET}         (start chatting!)"
echo ""
echo -e "  Gateway:  ${CYAN}dual-agent --gateway${RESET}   (Telegram bot)"
echo -e "  Help:     type ${CYAN}/help${RESET} inside the shell"
echo ""
