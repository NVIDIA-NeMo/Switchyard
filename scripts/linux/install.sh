#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs the Switchyard background server as a systemd --user service and sets
# up a `sy` Codex profile.
#
# Keeps existing composite.toml, replaces the service unit, and backs up
# sy.config.toml before replacing it. Use systemctl --user edit switchyard
# for service changes that survive reinstalling.
# Run with --dry-run to print what would happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRY_RUN=0
case "$#:${1:-}" in
  0:) ;;
  1:--dry-run) DRY_RUN=1 ;;
  *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

# shellcheck source=scripts/linux/common.sh
source "$SCRIPT_DIR/common.sh"

if [[ "$SY_HOME" =~ [[:space:][:cntrl:]] || "$SY_HOME" == *\\ ]]; then
  say "SY_HOME must not contain whitespace, control characters, or a trailing backslash." >&2
  exit 1
fi
if [[ ! "$SY_PORT" =~ ^[0-9]+$ ]]; then
  say "SY_PORT must contain only digits." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  say "This installer is for Linux only." >&2
  exit 1
fi

step "Building release binary"
run cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p switchyard-server

step "Installing binary into $SY_HOME/bin"
run mkdir -p "$SY_HOME/bin"
run install -m 755 "$REPO_ROOT/target/release/switchyard-server" "$SY_HOME/bin/switchyard-server"

step "Writing server config"
write_once "$SY_HOME/composite.toml" < "$REPO_ROOT/scripts/config/composite.toml"

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

[Install]
WantedBy=default.target
EOF

step "Loading the systemd user service"
run systemctl --user daemon-reload
run systemctl --user enable "$SERVICE_NAME"
run systemctl --user restart "$SERVICE_NAME"
run sleep 2
if (( ! DRY_RUN )) && ! systemctl --user is-active --quiet "$SERVICE_NAME"; then
  say "Service failed to start. See: journalctl --user -u switchyard" >&2
  exit 1
fi

step "Adding the sy Codex profile"
# This profile only changes which router answers. Approval and sandbox
# settings are deliberately left out, so the profile cannot loosen how Codex
# asks before it acts. Set those yourself if you want them.
sed "s/@SY_PORT@/$SY_PORT/g" "$REPO_ROOT/scripts/config/codex.sy.toml" | write_with_backup "$CODEX_PROFILE_CONFIG"

step "Done"
say "Server: http://127.0.0.1:$SY_PORT"
say "Logs: journalctl --user -u switchyard"
say "Manage it with: systemctl --user {status,restart,stop} $SERVICE_NAME"
say "Use it with: codex -p sy (requires Codex CLI 0.134.0 or newer)"
say ""
say "If you want the server running even when you are logged out, run:"
say "  loginctl enable-linger \$USER"
