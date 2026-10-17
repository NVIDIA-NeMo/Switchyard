# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SY_HOME="${SY_HOME:-$HOME/.switchyard}"
SY_PORT="${SY_PORT:-4123}"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
# Both installers use Codex's standalone profile file for the CLI.
CODEX_PROFILE_CONFIG="$CODEX_DIR/sy.config.toml"
ALIAS_START="# >>> switchyard codex alias >>>"
ALIAS_END="# <<< switchyard codex alias <<<"

say() { printf '%s\n' "$*"; }
step() { printf '\n==> %s\n' "$*"; }

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

run() {
  if (( DRY_RUN )); then
    say "  would run: $*"
  else
    "$@"
  fi
}

write_once() {
  local path="$1"
  if [[ -f "$path" ]]; then
    say "  keeping existing $path"
    cat >/dev/null
    return
  fi
  if (( DRY_RUN )); then
    say "  would create $path"
    cat >/dev/null
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    say "  created $path"
  fi
}

write_with_backup() {
  local path="$1"
  if (( DRY_RUN )); then
    [[ -f "$path" ]] && say "  would back up $path"
    say "  would write $path"
    cat >/dev/null
    return
  fi
  mkdir -p "$(dirname "$path")"
  local incoming
  incoming="$(mktemp)"
  cat > "$incoming"
  if [[ -f "$path" ]] && cmp -s "$incoming" "$path"; then
    say "  $path is already up to date"
    rm -f "$incoming"
    return
  fi
  if [[ -f "$path" ]]; then
    local backup
    backup="$path.switchyard-backup.$(date +%Y%m%d%H%M%S)"
    cp "$path" "$backup"
    say "  backed up $path to $backup"
  fi
  cat "$incoming" > "$path"
  rm -f "$incoming"
  say "  wrote $path"
}

write_always() {
  local path="$1"
  if (( DRY_RUN )); then
    say "  would write $path"
    cat >/dev/null
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    say "  wrote $path"
  fi
}

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
