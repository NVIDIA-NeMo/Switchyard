# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the evidence gate with synthetic outcomes, never task success evidence."""

import copy
import unittest

from evaluate import evaluate, validate


def fixture():
    plan = {
        "task_ids": [f"task-{i}" for i in range(200)], "development_task_ids": [],
        "protocol_sha256": "a" * 64, "pricing_sha256": "b" * 64,
        "candidate": "neural", "arm_sha256": {"neural": "c" * 64, "control": "d" * 64},
        "baselines": [{"name": "control", "mixture": {"control": 1.0}}],
        "replicates": 2, "cost_tolerance": 0,
    }
    rows = []
    for task in plan["task_ids"]:
        for repeat in range(2):
            for arm in plan["arm_sha256"]:
                rows.append({
                    "task_id": task, "replicate": repeat, "arm": arm,
                    "arm_sha256": plan["arm_sha256"][arm],
                    "protocol_sha256": plan["protocol_sha256"], "pricing_sha256": plan["pricing_sha256"],
                    "split": "test", "terminal": True, "cost_complete": True,
                    "reward": int(arm == "neural"), "total_cost_usd": 1 if arm == "neural" else 2,
                })
    return plan, rows


class EvidenceGate(unittest.TestCase):
    def test_detects_success_at_lower_cost_but_rejects_higher_cost(self):
        plan, rows = fixture()
        self.assertEqual(evaluate(plan, rows, draws=4000)["status"], "criteria_met")
        for row in rows:
            if row["arm"] == "neural":
                row["total_cost_usd"] = 3
        self.assertEqual(evaluate(plan, rows, draws=4000)["status"], "not_established")

    def test_equal_success_is_not_superiority(self):
        plan, rows = fixture()
        for row in rows:
            row["reward"] = 1
        self.assertEqual(evaluate(plan, rows, draws=4000)["status"], "not_established")

    def test_rejects_missing_cost_pairing_and_identity(self):
        for key, value in [("cost_complete", False), ("total_cost_usd", None),
                           ("total_cost_usd", float("nan")), ("arm_sha256", "e" * 64),
                           ("split", "development"), ("terminal", False)]:
            plan, rows = fixture()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(plan, rows)
        plan, rows = fixture()
        with self.assertRaises(ValueError):
            validate(plan, rows[:-1])
        with self.assertRaises(ValueError):
            validate(plan, rows + [copy.deepcopy(rows[0])])
        plan["development_task_ids"] = [plan["task_ids"][0]]
        with self.assertRaises(ValueError):
            validate(plan, rows)


if __name__ == "__main__":
    unittest.main()
