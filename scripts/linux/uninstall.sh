#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Removes what install.sh added: the systemd user service, the `sy` Codex
# profile, and any codex alias from an older install. Config and binaries stay
# put; the paths are printed so you can delete them yourself.
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

# shellcheck source=scripts/linux/common.sh
source "$SCRIPT_DIR/common.sh"

step "Removing the sy Codex profile"
remove_file "$CODEX_PROFILE_CONFIG"

step "Removing the codex alias"
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if [[ -f "$rc" ]] && grep -qF "$ALIAS_START" "$rc"; then
    strip_block "$rc" "$ALIAS_START" "$ALIAS_END" "the codex alias"
  else
    say "  no alias in $rc"
  fi
done

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

step "Done"
say "Left in place, delete them if you want:"
say "  $SY_HOME (binary, config, routing log)"
say "  $CODEX_PROFILE_CONFIG.switchyard-backup.* (backups taken at install time)"
