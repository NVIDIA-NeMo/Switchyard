#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
thinking_level=medium
pi_args=(--model switchyard/compare)
while (( $# )); do
  case "$1" in
    --) pi_args+=("$@"); break ;;
    --thinking)
      if (( $# < 2 )); then
        printf '%s\n' 'Use --thinking medium or --thinking max.' >&2
        exit 2
      fi
      thinking_level="$2"; shift 2 ;;
    --thinking=*) thinking_level="${1#*=}"; shift ;;
    *) pi_args+=("$1"); shift ;;
  esac
done
case "$thinking_level" in
  medium|max) ;;
  *) printf '%s\n' 'Use --thinking medium or --thinking max.' >&2; exit 2 ;;
esac

agent_root="${SY_FORECAST_AGENT_ROOT:-${XDG_CONFIG_HOME:-$HOME/.config}/switchyard-pi-forecast}"
export SY_FORECAST_CONFIG="${SY_FORECAST_CONFIG:-$agent_root/switchyard-forecast.json}"
export PI_CODING_AGENT_DIR="$agent_root/$thinking_level"
export PI_TELEMETRY="${PI_TELEMETRY:-0}"
mkdir -p "$PI_CODING_AGENT_DIR"
exec "$script_dir/node_modules/.bin/pi" -e "$script_dir/switchyard-compare.ts" \
  --thinking "$thinking_level" "${pi_args[@]}"
