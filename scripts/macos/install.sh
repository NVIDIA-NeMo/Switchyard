#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This script installs Switchyard.app, loads the server and app as per-user
# LaunchAgents, and sets up the selected Codex profile.
#
# It keeps existing server and app settings and backs up the selected profile
# before replacing it. The --dry-run option prints the installation commands.

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
(
  cd "$REPO_ROOT"
  run cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" \
    --target-dir "$REPO_ROOT/target" -p switchyard-server -p switchyard-menubar
)

step "Installing binaries into $SY_HOME/bin"
run mkdir -p "$SY_HOME/bin" "$SY_HOME/logs"
for binary in switchyard-server switchyard-menubar; do
  run install -m 755 "$REPO_ROOT/target/release/$binary" "$SY_HOME/bin/$binary"
done

step "Installing $APP_PATH"
run mkdir -p "$APP_PATH/Contents/MacOS" "$APP_PATH/Contents/Resources"
for binary in switchyard-menubar switchyard-server; do
  run install -m 755 "$REPO_ROOT/target/release/$binary" "$APP_PATH/Contents/MacOS/$binary"
done
write_always "$APP_PATH/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>com.nvidia.switchyard</string>
<key>CFBundleName</key><string>Switchyard</string>
<key>CFBundleDisplayName</key><string>Switchyard</string>
<key>CFBundleExecutable</key><string>Switchyard</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$REPO_ROOT/Cargo.toml" | head -1)</string>
<key>LSMinimumSystemVersion</key><string>12.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
# Bash's printf %q keeps source and settings paths literal in generated scripts.
shell_quote() { printf '%q' "$1"; }
write_always "$APP_PATH/Contents/MacOS/Switchyard" <<EOF
#!/bin/bash
exec "\$(dirname "\$0")/switchyard-menubar" $(shell_quote "$SY_HOME/menubar.toml")
EOF
write_always "$APP_PATH/Contents/Resources/Update.command" <<EOF
#!/bin/bash
set -euo pipefail
export PATH="\$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
export SY_HOME=$(shell_quote "$SY_HOME")
export SY_PORT=$(shell_quote "$SY_PORT")
export SY_PROFILE=$(shell_quote "$SY_PROFILE")
export SY_MODEL=$(shell_quote "$SY_MODEL")
export CODEX_HOME=$(shell_quote "$CODEX_DIR")
bash $(shell_quote "$SCRIPT_DIR/install.sh")
EOF
run chmod 755 "$APP_PATH/Contents/MacOS/Switchyard" "$APP_PATH/Contents/Resources/Update.command"
run codesign --force --deep --sign - "$APP_PATH"

step "Writing server config"
# Both installers use the same routing defaults.
write_once "$SY_HOME/composite.toml" < "$REPO_ROOT/scripts/config/composite.toml"

step "Writing menu bar settings"
# Rates are list prices per million tokens and are only used to estimate
# savings. Edit them to match what you actually pay.
TOML_SY_HOME="$(toml_escape_basic_string "$SY_HOME")"
write_once "$SY_HOME/menubar.toml" <<EOF
server_url = "http://127.0.0.1:$SY_PORT"
routing_log = "$TOML_SY_HOME/routing.jsonl"
config_file = "$TOML_SY_HOME/composite.toml"
launchd_label = "$SERVER_LABEL"
refresh_seconds = 30

# The baseline model prices caller-facing calls as if routing had not changed models.
# Savings subtract actual calls, including classifier calls, from that estimate.
baseline_model = "gpt-5.6-sol"

[prices."gpt-5.6-sol"]
input_per_mtok = 1.25
cached_input_per_mtok = 0.125
output_per_mtok = 10.0

[prices."gpt-5.6-luna"]
input_per_mtok = 0.25
cached_input_per_mtok = 0.025
output_per_mtok = 2.0

[prices."gpt-5.6-terra"]
input_per_mtok = 0.05
cached_input_per_mtok = 0.005
output_per_mtok = 0.4

# With a ChatGPT login, Codex sends "Approve for me" reviews to
# codex-auto-review even without Switchyard, so this price copies the
# baseline_model rates. Reviews then add the same amount to the actual cost and
# the baseline. The dollars saved stay the same, but the percentage drops a
# little. Update this price if you change baseline_model.
[prices."codex-auto-review"]
input_per_mtok = 1.25
cached_input_per_mtok = 0.125
output_per_mtok = 10.0
EOF

step "Validating the server config"
if (( DRY_RUN )); then
  say "  would run: $SY_HOME/bin/switchyard-server --config $SY_HOME/composite.toml --dry-run"
else
  "$SY_HOME/bin/switchyard-server" --config "$SY_HOME/composite.toml" --dry-run
fi

step "Writing LaunchAgents"
XML_SY_HOME="$(xml_escape_text "$SY_HOME")"
XML_APP_PATH="$(xml_escape_text "$APP_PATH")"
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

# SuccessfulExit false makes launchd restart crashes but leave a normal Quit alone.
# The app runs only in an Aqua login session, where the tray is available.
write_always "$LAUNCH_AGENTS/$MENUBAR_LABEL.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$MENUBAR_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$XML_APP_PATH/Contents/MacOS/Switchyard</string>
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
  say "  loaded $label"
done

step "Adding the $SY_PROFILE Codex profile"
# This profile only changes which router answers. Approval and sandbox
# settings are deliberately left out, so the profile cannot loosen how Codex
# asks before it acts. Set those yourself if you want them.
{
  printf 'model = "%s"\n' "$(toml_escape_basic_string "$SY_MODEL")"
  tail -n +2 "$REPO_ROOT/scripts/config/codex.sy.toml" | sed "s/@SY_PORT@/$SY_PORT/g"
} | write_with_backup "$CODEX_PROFILE_CONFIG"

step "Done"
say "Server:   http://127.0.0.1:$SY_PORT  (logs in $SY_HOME/logs)"
say "Use it with: codex -p $SY_PROFILE (requires Codex CLI 0.134.0 or newer)"
say "Settings: $SY_HOME/menubar.toml"
say "App: $APP_PATH (open it from Finder or Spotlight)"
say "Use Update from source… in the app to rebuild from this checkout."
