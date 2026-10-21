#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Installs the Switchyard background server and its menu bar companion as
# per-user LaunchAgents and sets up a `sy` Codex profile.
#
# Keeps existing server and menu bar settings and backs up sy.config.toml
# before replacing it.
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

toml_escape_basic_string() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  say "This installer is for macOS only." >&2
  exit 1
fi

step "Building release binaries"
run cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
  -p switchyard-server -p switchyard-menubar

step "Installing binaries into $SY_HOME/bin"
run mkdir -p "$SY_HOME/bin" "$SY_HOME/logs"
for binary in switchyard-server switchyard-menubar; do
  run install -m 755 "$REPO_ROOT/target/release/$binary" "$SY_HOME/bin/$binary"
done

step "Writing server config"
# Keep macOS and Linux on the same routing defaults.
write_once "$SY_HOME/composite.toml" < "$REPO_ROOT/scripts/config/composite.toml"

step "Writing menu bar settings"
# Existing price files override the bundled table for both applications.
TOML_SY_HOME="$(toml_escape_basic_string "$SY_HOME")"
XML_SY_HOME="$(xml_escape_text "$SY_HOME")"
PRICING_SETTING=""
PRICING_XML=""
SERVER_ARGS=(
  --config "$SY_HOME/composite.toml"
  --routing-log-file "$SY_HOME/routing.jsonl"
)
if [[ -e "$SY_HOME/prices.json" ]]; then
  SERVER_ARGS+=(--pricing-file "$SY_HOME/prices.json")
  PRICING_SETTING="pricing_file = \"$TOML_SY_HOME/prices.json\""
  PRICING_XML="<string>--pricing-file</string>
    <string>$XML_SY_HOME/prices.json</string>"
fi
write_once "$SY_HOME/menubar.toml" <<EOF
server_url = "http://127.0.0.1:$SY_PORT"
routing_log = "$TOML_SY_HOME/routing.jsonl"
config_file = "$TOML_SY_HOME/composite.toml"
launchd_label = "$SERVER_LABEL"
refresh_seconds = 30

# The model the traffic is assumed to have used without Switchyard. Savings
# are the difference between that bill and what actually ran, with
# Switchyard's own classifier calls counted against it.
baseline_model = "gpt-5.6-sol"

$PRICING_SETTING
EOF

step "Validating the server config"
run "$SY_HOME/bin/switchyard-server" "${SERVER_ARGS[@]}" --dry-run

step "Writing LaunchAgents"
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
    $PRICING_XML
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

# SuccessfulExit false means the Quit menu item stays quit, while a crash is
# still restarted. Aqua-only, since there is no menu bar without a login session.
write_always "$LAUNCH_AGENTS/$MENUBAR_LABEL.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$MENUBAR_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$XML_SY_HOME/bin/switchyard-menubar</string>
    <string>$XML_SY_HOME/menubar.toml</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>LimitLoadToSessionType</key>
  <string>Aqua</string>
  <key>StandardOutPath</key>
  <string>$XML_SY_HOME/logs/menubar.log</string>
  <key>StandardErrorPath</key>
  <string>$XML_SY_HOME/logs/menubar.err.log</string>
</dict>
</plist>
EOF

step "Loading LaunchAgents"
for label in "$SERVER_LABEL" "$MENUBAR_LABEL"; do
  if (( DRY_RUN )); then
    say "  would reload gui/$UID/$label"
    continue
  fi
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  # bootout returns before the job leaves the domain, and bootstrapping a
  # service that is still shutting down fails with an I/O error.
  for _ in $(seq 50); do
    launchctl print "gui/$UID/$label" >/dev/null 2>&1 || break
    sleep 0.1
  done
  launchctl bootstrap "gui/$UID" "$LAUNCH_AGENTS/$label.plist"
  # macOS can defer RunAtLoad jobs until an explicit start is requested.
  launchctl kickstart "gui/$UID/$label"
  say "  started $label"
done

step "Adding the sy Codex profile"
# This profile only changes which router answers. Approval and sandbox
# settings are deliberately left out, so the profile cannot loosen how Codex
# asks before it acts. Set those yourself if you want them.
sed "s/@SY_PORT@/$SY_PORT/g" "$REPO_ROOT/scripts/config/codex.sy.toml" |
  write_with_backup "$CODEX_PROFILE_CONFIG"

step "Done"
say "Server:   http://127.0.0.1:$SY_PORT  (logs in $SY_HOME/logs)"
say "Use it with: codex -p sy (requires Codex CLI 0.134.0 or newer)"
say "Settings: $SY_HOME/menubar.toml"
say "Look for the Switchyard glyph in the menu bar."
