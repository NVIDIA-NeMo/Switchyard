#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Removes what install.sh added: the two LaunchAgents, the `sy` Codex profile,
# and the codex alias. Your config, routing log, and binaries stay put; the
# paths are printed so you can delete them yourself.
#
# Run with --dry-run to print what would happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

# Deletes a file this script owns.
remove_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    say "  nothing to remove at $path"
  elif (( DRY_RUN )); then
    say "  would delete $path"
  else
    rm -f "$path"
    say "  deleted $path"
  fi
}

step "Unloading LaunchAgents"
for label in "$SERVER_LABEL" "$MENUBAR_LABEL"; do
  if (( DRY_RUN )); then
    say "  would unload gui/$UID/$label and delete its plist"
  else
    launchctl bootout "gui/$UID/$label" 2>/dev/null || true
    rm -f "$LAUNCH_AGENTS/$label.plist"
    say "  unloaded $label"
  fi
done

step "Removing the sy Codex profile"
remove_file "$CODEX_PROFILE_CONFIG"
remove_file "$CODEX_SWITCHYARD_CONFIG"
# Older installs of this script put the profile in config.toml instead.
strip_block "$CODEX_CONFIG" "$PROFILE_START" "$PROFILE_END" "the legacy sy profile" ||
  say "  no legacy profile in $CODEX_CONFIG"

step "Unrouting Codex.app"
# Leaving a routed config.toml behind would point Codex.app at a server that
# is no longer running, so put the snapshot back before the agents go away.
if [[ -f "$CODEX_CONFIG" ]] && grep -qE '^[[:space:]]*model_provider[[:space:]]*=[[:space:]]*"sy"' "$CODEX_CONFIG"; then
  if [[ ! -f "$CODEX_DIRECT_CONFIG" ]]; then
    say "  config.toml is routed but $CODEX_DIRECT_CONFIG is missing; edit it by hand"
  elif (( DRY_RUN )); then
    say "  would restore $CODEX_DIRECT_CONFIG over config.toml"
  else
    cp "$CODEX_DIRECT_CONFIG" "$CODEX_CONFIG"
    rm -f "$CODEX_DIRECT_CONFIG"
    say "  restored the unrouted config.toml"
  fi
else
  remove_file "$CODEX_DIRECT_CONFIG"
fi

step "Removing the codex alias"
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  strip_block "$rc" "$ALIAS_START" "$ALIAS_END" "the codex alias" ||
    say "  no alias in $rc"
done

step "Done"
say "Left in place, delete them if you want:"
say "  $SY_HOME (binaries, config, routing log, logs)"
say "  $CODEX_CONFIG.switchyard-backup.* (backups taken at install time)"
