# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paths, markers, and helpers shared by install.sh and uninstall.sh.
# The markers must match between the two, which is why they live here.

SY_HOME="${SY_HOME:-$HOME/.switchyard}"
SY_PORT="${SY_PORT:-4123}"
SERVER_LABEL="com.nvidia.switchyard.server"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
CODEX_CONFIG="$CODEX_DIR/config.toml"
# Codex reads `--profile sy` from its own file next to config.toml. A
# [profiles.sy] table in config.toml is rejected outright as legacy config.
CODEX_PROFILE_CONFIG="$CODEX_DIR/sy.config.toml"
# Codex.app ignores profiles entirely, so it gets a whole config.toml with the
# routing inlined, plus a snapshot of the unrouted one to swap back to.
CODEX_SWITCHYARD_CONFIG="$CODEX_DIR/config.toml.sy"
CODEX_DIRECT_CONFIG="$CODEX_DIR/config.toml.direct"
ALIAS_START="# >>> switchyard codex alias >>>"
ALIAS_END="# <<< switchyard codex alias <<<"
PROFILE_START="# >>> switchyard sy profile >>>"
PROFILE_END="# <<< switchyard sy profile <<<"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

# Deletes the marked block, inclusive, leaving the rest of the file alone.
strip_block() {
  local path="$1" start="$2" end="$3" label="${4:-the switchyard block}"
  if [[ ! -f "$path" ]] || ! grep -qF "$start" "$path"; then
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
  ' "$path" > "$temp"
  cat "$temp" > "$path"
  rm -f "$temp"
  say "  removed $label from $path"
  return 0
}

codex_config_uses_switchyard() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  awk '
    function is_string(value, wanted, quote, prefix, suffix) {
      sub(/^[[:space:]]*/, "", value)
      quote = substr(value, 1, 1)
      if (quote != "\"" && quote != sprintf("%c", 39)) return 0
      prefix = quote wanted quote
      if (substr(value, 1, length(prefix)) != prefix) return 0
      suffix = substr(value, length(prefix) + 1)
      return suffix ~ /^[[:space:]]*(#.*)?$/
    }
    BEGIN { top_level = 1 }
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*\[/ { top_level = 0; next }
    top_level && /^[[:space:]]*model_provider[[:space:]]*=/ {
      value = $0
      sub(/^[^=]*=[[:space:]]*/, "", value)
      if (is_string(value, "sy")) found = 1
    }
    END { exit !found }
  ' "$path"
}
