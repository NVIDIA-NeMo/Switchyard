# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file defines paths, markers, and helpers shared by install.sh and uninstall.sh.
# The markers must match between the two, which is why they live here.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SERVER_LABEL="com.nvidia.switchyard.server"
MENUBAR_LABEL="com.nvidia.switchyard.menubar"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
APP_PATH="$HOME/Applications/Switchyard.app"
CODEX_CONFIG="$CODEX_DIR/config.toml"
# Codex reads named profiles from separate files beside config.toml.
# These paths identify configs written by older installers for cleanup and restore.
CODEX_SWITCHYARD_CONFIG="$CODEX_DIR/config.sy.toml"
CODEX_DIRECT_CONFIG="$CODEX_DIR/config.toml.direct"
PROFILE_START="# >>> switchyard sy profile >>>"
PROFILE_END="# <<< switchyard sy profile <<<"

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

SY_PROFILE="${SY_PROFILE-sy}"
SY_MODEL="${SY_MODEL-composite-gpt-6-sol-gpt-6-luna}"
if [[ ! "$SY_PROFILE" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || (( ${#SY_PROFILE} > 128 )); then
  say "SY_PROFILE must start with a letter or digit and contain only letters, digits, underscores, or hyphens, up to 128 characters." >&2
  exit 2
fi
if [[ -z "$SY_MODEL" || "$SY_MODEL" =~ [[:cntrl:]] ]]; then
  say "SY_MODEL must be a nonempty public route ID without ASCII control characters." >&2
  exit 2
fi
CODEX_PROFILE_CONFIG="$CODEX_DIR/$SY_PROFILE.config.toml"
