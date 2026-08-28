#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

HARBOR_LAB="${HARBOR_LAB:-$(command -v harbor-lab || true)}"

usage() {
  echo "Usage: $0 JOB_DIR [OUT_JSON]"
  echo
  echo "JOB_DIR is the Harbor job directory, e.g. tmp/swebench-results/2026-04-27__12-22-41"
  echo "OUT_JSON defaults to JOB_DIR/harbor-lab-summary.json"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

JOB_DIR="${1%/}"
OUT_JSON="${2:-$JOB_DIR/harbor-lab-summary.json}"

if [[ ! -x "$HARBOR_LAB" ]]; then
  echo "ERROR: harbor-lab not executable at: $HARBOR_LAB" >&2
  echo "Set HARBOR_LAB=/path/to/harbor-lab to override." >&2
  exit 1
fi

if [[ ! -d "$JOB_DIR" ]]; then
  echo "ERROR: job directory not found: $JOB_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

run_json() {
  local name="$1"
  shift
  "$HARBOR_LAB" "$@" "$JOB_DIR/" --format json > "$TMP_DIR/$name.json"
}

run_json errors errors
run_json edits edits
run_json edits_verbose edits --verbose
run_json tool_sequence tool-sequence --tail 10 --text
run_json metrics metrics

python - "$JOB_DIR" "$HARBOR_LAB" "$TMP_DIR" > "$OUT_JSON" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

job_dir = sys.argv[1]
harbor_lab = sys.argv[2]
tmp_dir = Path(sys.argv[3])


def load_section(name: str):
    with (tmp_dir / f"{name}.json").open() as f:
        return json.load(f)


summary = {
    "job_dir": job_dir,
    "harbor_lab": harbor_lab,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "sections": {
        "errors": load_section("errors"),
        "edits": load_section("edits"),
        "edits_verbose": load_section("edits_verbose"),
        "tool_sequence": load_section("tool_sequence"),
        "metrics": load_section("metrics"),
    },
}

json.dump(summary, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY

echo "Wrote $OUT_JSON"
