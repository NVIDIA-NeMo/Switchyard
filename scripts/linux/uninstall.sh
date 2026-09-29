#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Removes what install.sh added: the systemd user service, the `sy` Codex
# profile, and the codex alias. Your config, routing log, and binaries stay
# put; the paths are printed so you can delete them yourself.
#
# Run with --dry-run to print what would happen.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# shellcheck source=scripts/linux/common.sh
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

step "Stopping the systemd user service"
if (( DRY_RUN )); then
  say "  would run: systemctl --user disable --now $SERVICE_NAME"
  say "  would delete $SYSTEMD_USER_DIR/$SERVICE_NAME"
else
  systemctl --user disable --now "$SERVICE_NAME" 2>/dev/null || true
  rm -f "$SYSTEMD_USER_DIR/$SERVICE_NAME"
  systemctl --user daemon-reload
  say "  stopped and removed $SERVICE_NAME"
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
say "  $SY_HOME (binary, config, routing log, logs)"
say "  $CODEX_PROFILE_CONFIG.switchyard-backup.* (backups taken at install time)"
