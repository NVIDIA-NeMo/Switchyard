#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary F2P + P2P reward gate for a planning task.

reward = 1.0 iff every FAIL_TO_PASS test passes AND every PASS_TO_PASS test passes.
Otherwise 0.0. Fractional F2P / P2P counts stay in /logs/verifier/results.json
for analysis; the harbor task reward itself is strict binary.
"""

import json
import re
from pathlib import Path

log_dir = Path("/logs/verifier")
verify_output = (log_dir / "verify_full_output.txt").read_text()

all_passed = set()
all_failed = set()
for line in verify_output.split("\n"):
    m = re.match(r"^\s+(\S+::\S+)\s+(PASSED|FAILED)", line)
    if m:
        test_name, status = m.group(1), m.group(2)
        (all_passed if status == "PASSED" else all_failed).add(test_name)


def load_test_list(path):
    p = Path(path)
    if not p.exists():
        return set()
    return set(line.strip() for line in p.read_text().strip().split("\n") if line.strip())


f2p_list = load_test_list("/tests/fail_to_pass.txt")
p2p_list = load_test_list("/tests/pass_to_pass.txt")

f2p_passed = f2p_list & all_passed
f2p_failed = f2p_list - all_passed
p2p_passed = p2p_list & all_passed
p2p_failed = p2p_list - all_passed

f2p_total = len(f2p_list)
p2p_total = len(p2p_list)
f2p_score = len(f2p_passed) / f2p_total if f2p_total > 0 else 0.0
p2p_score = len(p2p_passed) / p2p_total if p2p_total > 0 else 1.0

binary_reward = 1.0 if (f2p_total > 0 and not f2p_failed and not p2p_failed) else 0.0

results = {
    "reward": binary_reward,
    "f2p": {
        "passed": len(f2p_passed),
        "total": f2p_total,
        "score": round(f2p_score, 3),
        "passed_tests": sorted(f2p_passed),
        "failed_tests": sorted(f2p_failed),
    },
    "p2p": {
        "passed": len(p2p_passed),
        "total": p2p_total,
        "score": round(p2p_score, 3),
        "failed_tests": sorted(p2p_failed),
    },
}

print(f"F2P: {len(f2p_passed)}/{f2p_total} ({f2p_score:.1%})")
if p2p_total > 0:
    print(f"P2P: {len(p2p_passed)}/{p2p_total} ({p2p_score:.1%})")
if f2p_failed:
    print(f"F2P still failing: {sorted(f2p_failed)}")
if p2p_failed:
    print(f"P2P regressions: {sorted(p2p_failed)}")
print(f"Binary reward: {binary_reward:.1f}")

(log_dir / "reward.txt").write_text(f"{binary_reward:.1f}\n")
json.dump(results, open(log_dir / "results.json", "w"), indent=2)
