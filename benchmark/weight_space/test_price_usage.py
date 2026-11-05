# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from price_usage import price


class PricingTests(unittest.TestCase):
    def setUp(self):
        self.call = {
            "model": "gateway/worker", "prompt_tokens": 1_000_000,
            "cached_tokens": 800_000, "cache_creation_tokens": 100_000,
            "cache_creation_1h_tokens": 0, "completion_tokens": 10_000,
            "tier": "classifier", "cost": None,
        }
        self.capture = {"status": "captured_usage_reconciles_with_session_stats",
                        "session_id": "s", "call_count": 1, "calls": [self.call]}
        self.rates = {"gateway/worker": {"input": 10, "output": 50,
                                        "cache_read": 1, "cache_write": 12.5}}

    def test_cache_and_judge_costs_do_not_replace_missing_provider_charges(self):
        result = price(self.capture, self.rates)
        self.assertAlmostEqual(result["token_snapshot_total_usd"], 3.55)
        self.assertAlmostEqual(result["token_snapshot_routing_usd"], 3.55)
        self.assertEqual(result["calls_without_provider_charge"], 1)
        self.assertIsNone(result["provider_reported_total_usd"])
        self.call["cost"] = {"source": "provider_reported", "usd": 1.775}
        result = price(self.capture, self.rates)
        self.assertAlmostEqual(result["token_snapshot_total_usd"], 3.55)
        self.assertEqual(result["provider_reported_total_usd"], 1.775)

    def test_missing_model_or_cache_information_cannot_be_free(self):
        with self.assertRaises(ValueError):
            price(self.capture, {})
        del self.call["cache_creation_tokens"]
        with self.assertRaises(ValueError):
            price(self.capture, self.rates)

    def test_one_hour_cache_writes_require_their_own_rate(self):
        self.call["cache_creation_1h_tokens"] = 100_000
        with self.assertRaises(ValueError):
            price(self.capture, self.rates)
        self.rates["gateway/worker"]["cache_write_1h"] = 20
        self.assertAlmostEqual(price(self.capture, self.rates)["token_snapshot_total_usd"], 4.3)
        self.call["cached_tokens"] = 1_000_000
        with self.assertRaises(ValueError):
            price(self.capture, self.rates)


if __name__ == "__main__":
    unittest.main()
