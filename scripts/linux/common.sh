# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paths, markers, and helpers shared by install.sh and uninstall.sh.
# The markers must match between the two, which is why they live here.

SY_HOME="${SY_HOME:-$HOME/.switchyard}"
SY_PORT="${SY_PORT:-4123}"
SERVICE_NAME="switchyard.service"
SYSTEMD_USER_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
# Since Codex 0.134.0, `--profile sy` reads sy.config.toml and fails if
# config.toml still has [profiles.sy].
CODEX_PROFILE_CONFIG="$CODEX_DIR/sy.config.toml"
ALIAS_START="# >>> switchyard codex alias >>>"
ALIAS_END="# <<< switchyard codex alias <<<"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

# Deletes the marked block, inclusive, leaving the rest of the file alone.
strip_block() {
  local path="$1" start="$2" end="$3" label="${4:-the switchyard block}"
  if [[ ! -f "$path" ]] || ! grep -qF "$start" "$path"; then
    return 1
  fi
  if ! grep -qF "$end" "$path"; then
    say "  $path has no end marker for $label; not editing it" >&2
    return 1
  fi
  if (( DRY_RUN )); then
    say "  would remove $label from $path"
    return 0
  fi
  local temp
  temp="$(mktemp)"
  awk -v start="$start" -v end="$end" '
    index($0, start) { skipping = 1 }
    !skipping { print }
    index($0, end) { skipping = 0 }
    END { if (skipping) exit 1 }
  ' "$path" > "$temp"
  cat "$temp" > "$path"
  rm -f "$temp"
  say "  removed $label from $path"
  return 0
}
