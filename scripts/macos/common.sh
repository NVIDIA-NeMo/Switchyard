# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paths, markers, and helpers shared by install.sh and uninstall.sh.
# The markers must match between the two, which is why they live here.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SERVER_LABEL="com.nvidia.switchyard.server"
MENUBAR_LABEL="com.nvidia.switchyard.menubar"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
CODEX_CONFIG="$CODEX_DIR/config.toml"
# Codex reads `--profile sy` from its own file next to config.toml. A
# [profiles.sy] table in config.toml is rejected outright as legacy config.
# Codex.app ignores profiles entirely, so it gets a whole config.toml with the
# routing inlined. Keep the generated file beside the original for reference.
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
