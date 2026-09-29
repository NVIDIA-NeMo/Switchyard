#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs the Switchyard background server as a systemd --user service, sets
# up a `sy` Codex profile, and aliases `codex` to use it.
#
# Every step is idempotent and never overwrites a file you have edited.
# Run with --dry-run to print what would happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# shellcheck source=scripts/linux/common.sh
source "$SCRIPT_DIR/common.sh"

# Runs a command, or prints it when dry running.
run() {
  if (( DRY_RUN )); then
    say "  would run: $*"
  else
    "$@"
  fi
}

# Writes stdin to a file, leaving an existing file untouched.
write_once() {
  local path="$1"
  if [[ -f "$path" ]]; then
    say "  keeping existing $path"
    cat >/dev/null
    return
  fi
  if (( DRY_RUN )); then
    say "  would create $path"
    cat >/dev/null
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    say "  created $path"
  fi
}

# Writes stdin to a file, keeping any existing version as a timestamped backup.
write_with_backup() {
  local path="$1"
  if (( DRY_RUN )); then
    [[ -f "$path" ]] && say "  would back up $path"
    say "  would write $path"
    cat >/dev/null
    return
  fi
  mkdir -p "$(dirname "$path")"
  local incoming
  incoming="$(mktemp)"
  cat > "$incoming"
  if [[ -f "$path" ]] && cmp -s "$incoming" "$path"; then
    say "  $path is already up to date"
    rm -f "$incoming"
    return
  fi
  if [[ -f "$path" ]]; then
    local backup
    backup="$path.switchyard-backup.$(date +%Y%m%d%H%M%S)"
    cp "$path" "$backup"
    say "  backed up $path to $backup"
  fi
  cat "$incoming" > "$path"
  rm -f "$incoming"
  say "  wrote $path"
}

# Writes stdin to a file, replacing it. Used only for files this script owns.
write_always() {
  local path="$1"
  if (( DRY_RUN )); then
    say "  would write $path"
    cat >/dev/null
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    say "  wrote $path"
  fi
}

if [[ "$(uname -s)" != "Linux" ]]; then
  say "This installer is for Linux only." >&2
  exit 1
fi

step "Building release binary"
run cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p switchyard-server

step "Installing binary into $SY_HOME/bin"
run mkdir -p "$SY_HOME/bin" "$SY_HOME/logs"
run install -m 755 "$REPO_ROOT/target/release/switchyard-server" "$SY_HOME/bin/switchyard-server"

step "Writing server config"
# The composite router from examples/run_codex.sh: Terra classifies each user
# turn and sets the tier, Stage drives the tool loop underneath it.
write_once "$SY_HOME/composite.toml" <<'EOF'
schema_version = 1

[llm_clients.chatgpt_backend]
format = "openai_responses"
base_url = "https://chatgpt.com/backend-api/codex"
forward_auth = true

[targets.capable]
id = "gpt-5.6-sol"
llm_client = "chatgpt_backend"

[targets.efficient]
id = "gpt-5.6-luna"
llm_client = "chatgpt_backend"

# chatgpt.com/backend-api/codex is Codex CLI's own private endpoint, not the
# public OpenAI Responses API. It 400s unless store=false and stream=true are
# set explicitly, and it rejects max_output_tokens outright, so the
# classifier's own token cap has to be dropped before the request goes out.
[targets.terra]
id = "gpt-5.6-terra"
llm_client = "chatgpt_backend"
extra_body = { store = false, stream = true }
omit_body_fields = ["max_output_tokens"]

[routes.switchyard]
id = "switchyard"
type = "composite"

[routes.switchyard.classifier]
target = "terra"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.5
EOF

step "Validating the server config"
if (( DRY_RUN )); then
  say "  would run: $SY_HOME/bin/switchyard-server --config $SY_HOME/composite.toml --dry-run"
else
  "$SY_HOME/bin/switchyard-server" --config "$SY_HOME/composite.toml" --dry-run
fi

step "Writing the systemd user service"
write_always "$SYSTEMD_USER_DIR/$SERVICE_NAME" <<EOF
[Unit]
Description=Switchyard LLM router server

[Service]
Type=simple
ExecStart=$SY_HOME/bin/switchyard-server --config $SY_HOME/composite.toml --host 127.0.0.1 --port $SY_PORT --routing-log-file $SY_HOME/routing.jsonl
Restart=on-failure
Environment=RUST_LOG=info
StandardOutput=append:$SY_HOME/logs/server.log
StandardError=append:$SY_HOME/logs/server.err.log

[Install]
WantedBy=default.target
EOF

step "Loading the systemd user service"
if (( DRY_RUN )); then
  say "  would run: systemctl --user daemon-reload"
  say "  would run: systemctl --user enable --now $SERVICE_NAME"
else
  systemctl --user daemon-reload
  systemctl --user enable --now "$SERVICE_NAME"
  say "  enabled and started $SERVICE_NAME"
fi

step "Adding the sy Codex profile"
# `codex --profile sy` reads ~/.codex/sy.config.toml. A [profiles.sy] table in
# config.toml is legacy config that Codex refuses to start with, so the
# profile lives in its own file instead.
#
# This profile only changes which router answers. Approval and sandbox
# settings are deliberately left out, so the profile cannot loosen how Codex
# asks before it acts. Set those yourself if you want them.
write_with_backup "$CODEX_PROFILE_CONFIG" <<EOF
model = "switchyard"
model_provider = "sy"

[model_providers.sy]
name = "Switchyard"
base_url = "http://127.0.0.1:$SY_PORT/v1"
wire_api = "responses"
requires_openai_auth = true
EOF

step "Aliasing codex"
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if [[ -f "$rc" ]] && grep -qF "$ALIAS_START" "$rc"; then
    say "  alias already in $rc"
  elif (( DRY_RUN )); then
    say "  would add the codex alias to $rc"
  else
    printf '\n%s\nalias codex="codex --profile sy"\n%s\n' "$ALIAS_START" "$ALIAS_END" >> "$rc"
    say "  added the codex alias to $rc"
  fi
done

step "Done"
say "Server: http://127.0.0.1:$SY_PORT  (logs in $SY_HOME/logs)"
say "Manage it with: systemctl --user {status,restart,stop} $SERVICE_NAME"
say "Open a new shell, or run: alias codex=\"codex --profile sy\""
say ""
say "If you want the server running even when you are logged out, run:"
say "  loginctl enable-linger \$USER"
