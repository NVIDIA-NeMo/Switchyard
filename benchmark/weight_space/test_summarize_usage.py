# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check capture reconciliation and distinguish target changes from cache misses."""

import json
import unittest

from price_usage import price
from summarize_usage import summarize


class UsageCaptureTests(unittest.TestCase):
    def fixture(self):
        calls = []
        for model, tier, cached in [("a", "capable", 0), ("judge", "classifier", 0),
                                    ("b", "efficient", 20), ("a", "capable", 80)]:
            calls.append({"session_id": "session", "model": model, "target": model,
                          "tier": tier, "prompt_tokens": 100, "completion_tokens": 10,
                          "cached_tokens": cached, "cache_creation_tokens": 0})
        log = "\n".join("DEBUG Routing usage recorded record=" + json.dumps(call) for call in calls)
        stats = {"sessions": {"session": {"total_calls": 4, "total_prompt_tokens": 400,
                 "total_completion_tokens": 40, "total_cached_tokens": 100,
                 "total_cache_creation_tokens": 0}}}
        return log, stats

    def test_return_can_reuse_cache_and_judge_does_not_change_answer_target(self):
        log, stats = self.fixture()
        result = summarize(log, stats, "session")
        self.assertEqual(result["answer_target_changes"], 2)
        self.assertIsNone(result["calls"][1]["answer_target_changed"])
        self.assertTrue(result["calls"][3]["return_to_previously_served_target"])
        self.assertEqual(result["calls"][3]["cache_read_fraction"], .8)

    def test_missing_or_duplicate_capture_is_rejected(self):
        log, stats = self.fixture()
        with self.assertRaises(ValueError):
            summarize("\n".join(log.splitlines()[:-1]), stats, "session")
        with self.assertRaises(ValueError):
            summarize(log + "\n" + log.splitlines()[0], stats, "session")

    def test_native_omitted_zero_cache_duration_can_be_priced(self):
        log, stats = self.fixture()
        result = summarize(log, stats, "session")
        rates = {model: {"input": 1, "output": 1, "cache_read": 1, "cache_write": 1}
                 for model in ["a", "b", "judge"]}
        self.assertEqual(price(result, rates)["call_count"], 4)
        self.assertTrue(all(call["cache_creation_1h_tokens"] == 0 for call in result["calls"]))

    def test_reported_cache_duration_is_preserved(self):
        log, stats = self.fixture()
        prefix, payload = log.splitlines()[0].split("record=", 1)
        call = json.loads(payload)
        call.update(cache_creation_tokens=10, cache_creation_1h_tokens=7)
        log = "\n".join([prefix + "record=" + json.dumps(call), *log.splitlines()[1:]])
        stats["sessions"]["session"]["total_cache_creation_tokens"] = 10
        self.assertEqual(summarize(log, stats, "session")["calls"][0]["cache_creation_1h_tokens"], 7)

    def test_inconsistent_token_totals_are_rejected(self):
        log, stats = self.fixture()
        stats["sessions"]["session"]["total_cached_tokens"] = 99
        with self.assertRaises(ValueError):
            summarize(log, stats, "session")


if __name__ == "__main__":
    unittest.main()
