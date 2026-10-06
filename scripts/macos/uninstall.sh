#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Removes what install.sh added: the LaunchAgent, the `sy` Codex profile,
# and the codex alias. Your config, routing log, and binaries stay put; the
# paths are printed so you can delete them yourself.
#
# Run with --dry-run to print what would happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
case "$#:${1:-}" in
  0:) ;;
  1:--dry-run) DRY_RUN=1 ;;
  *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

# shellcheck source=scripts/macos/common.sh
source "$SCRIPT_DIR/common.sh"

step "Unloading LaunchAgents"
if (( DRY_RUN )); then
  say "  would unload gui/$UID/$SERVER_LABEL and delete its plist"
else
  launchctl bootout "gui/$UID/$SERVER_LABEL" 2>/dev/null || true
  rm -f "$LAUNCH_AGENTS/$SERVER_LABEL.plist"
  say "  unloaded $SERVER_LABEL"
fi

step "Removing the sy Codex profile"
remove_file "$CODEX_PROFILE_CONFIG"

step "Removing the codex alias"
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  strip_block "$rc" "$ALIAS_START" "$ALIAS_END" "the codex alias" ||
    say "  no alias in $rc"
done

step "Done"
say "Left in place, delete them if you want:"
say "  $SY_HOME (binaries, config, routing log, logs)"
say "  $CODEX_PROFILE_CONFIG.switchyard-backup.* (profile backups)"
