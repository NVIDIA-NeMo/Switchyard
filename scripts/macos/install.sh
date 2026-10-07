#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs the Switchyard background server as a per-user LaunchAgent and sets
# up a `sy` Codex profile.
#
# Keeps existing composite.toml and backs up sy.config.toml before replacing it.
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

# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

xml_escape_text() {
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g'
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  say "This installer is for macOS only." >&2
  exit 1
fi

step "Building release binaries"
run cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
  -p switchyard-server

step "Installing binaries into $SY_HOME/bin"
run mkdir -p "$SY_HOME/bin" "$SY_HOME/logs"
run install -m 755 "$REPO_ROOT/target/release/switchyard-server" \
  "$SY_HOME/bin/switchyard-server"

step "Writing server config"
# Keep macOS and Linux on the same routing defaults.
write_once "$SY_HOME/composite.toml" < "$REPO_ROOT/scripts/config/composite.toml"

step "Validating the server config"
if (( DRY_RUN )); then
  say "  would run: $SY_HOME/bin/switchyard-server --config $SY_HOME/composite.toml --dry-run"
else
  "$SY_HOME/bin/switchyard-server" --config "$SY_HOME/composite.toml" --dry-run
fi

step "Writing LaunchAgents"
XML_SY_HOME="$(xml_escape_text "$SY_HOME")"
write_always "$LAUNCH_AGENTS/$SERVER_LABEL.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$SERVER_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$XML_SY_HOME/bin/switchyard-server</string>
    <string>--config</string>
    <string>$XML_SY_HOME/composite.toml</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>$SY_PORT</string>
    <string>--routing-log-file</string>
    <string>$XML_SY_HOME/routing.jsonl</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>$XML_SY_HOME/logs/server.log</string>
  <key>StandardErrorPath</key>
  <string>$XML_SY_HOME/logs/server.err.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>RUST_LOG</key>
    <string>info</string>
  </dict>
</dict>
</plist>
EOF

step "Loading LaunchAgents"
if (( DRY_RUN )); then
  say "  would reload gui/$UID/$SERVER_LABEL"
else
  launchctl bootout "gui/$UID/$SERVER_LABEL" 2>/dev/null || true
  # bootout returns before the job leaves the domain, and bootstrapping a
  # service that is still shutting down fails with an I/O error.
  for _ in $(seq 50); do
    launchctl print "gui/$UID/$SERVER_LABEL" >/dev/null 2>&1 || break
    sleep 0.1
  done
  launchctl bootstrap "gui/$UID" "$LAUNCH_AGENTS/$SERVER_LABEL.plist"
  say "  loaded $SERVER_LABEL"
fi

step "Adding the sy Codex profile"
# This profile only changes which router answers. Approval and sandbox
# settings are deliberately left out, so the profile cannot loosen how Codex
# asks before it acts. Set those yourself if you want them.
sed "s/@SY_PORT@/$SY_PORT/g" "$REPO_ROOT/scripts/config/codex.sy.toml" |
  write_with_backup "$CODEX_PROFILE_CONFIG"

step "Done"
say "Server:   http://127.0.0.1:$SY_PORT  (logs in $SY_HOME/logs)"
say "Use it with: codex -p sy (requires Codex CLI 0.134.0 or newer)"
say "Codex profile: $CODEX_PROFILE_CONFIG"
