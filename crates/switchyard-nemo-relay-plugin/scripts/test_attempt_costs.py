# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check USD accounting with Relay's response codec, without loading a plugin."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from attempt_costs import summarize  # noqa: E402


def report(*costs):
    events = [
        {"kind": "scope", "uuid": "workflow", "scope_category": "end"},
        {
            "kind": "scope", "uuid": "call", "parent_uuid": "workflow",
            "name": "libsy.client_call", "scope_category": "end",
            "metadata": {
                "switchyard.accounting.unit": "summary",
                "switchyard.accounting.attempts": len(costs),
                "switchyard.accounting.attempt_records": len(costs),
            },
        },
    ]
    for ordinal, cost in enumerate(costs, 1):
        events.append({
            "kind": "scope", "uuid": f"attempt-{ordinal}", "parent_uuid": "call",
            "scope_category": "end",
            "metadata": {
                "attempt": ordinal, "switchyard.accounting.unit": "completion_attempt",
                "switchyard.accounting.usage_status": "complete", "outcome": "ok",
            },
            "data": {
                "type": "message", "role": "assistant", "content": [], "model": "test/model",
                "usage": {"input_tokens": 1, "output_tokens": 1, "cost": cost},
            },
        })
    return summarize(events, "workflow")


class CurrencyTests(unittest.TestCase):
    def test_usd_and_missing_currency_keep_complete_total(self):
        for currency in ["USD", "usd", None]:
            with self.subTest(currency=currency):
                cost = {"total": 0.02}
                if currency is not None:
                    cost["currency"] = currency
                result = report(cost)
                self.assertEqual(result["known_cost_subtotal_usd"], 0.02)
                self.assertEqual(result["total_cost_usd"], 0.02)
                self.assertEqual(result["cost_status"], "complete")

    def test_non_usd_amounts_do_not_contribute_to_usd_totals(self):
        for currency, amount in [("EUR", 0.02), ("JPY", 3), ("EUR", 0.0)]:
            with self.subTest(currency=currency, amount=amount):
                result = report({"total": amount, "currency": currency})
                self.assertTrue(result["coverage_complete"])
                self.assertEqual(result["known_cost_subtotal_usd"], 0.0)
                self.assertIsNone(result["total_cost_usd"])
                self.assertEqual(result["cost_status"], "incomplete")
                self.assertFalse(result["attempts"][0]["cost_complete"])

    def test_mixed_currencies_keep_only_the_usd_subtotal(self):
        result = report({"total": 0.03, "currency": "USD"}, {"total": 0.02, "currency": "EUR"})
        self.assertEqual(result["known_cost_subtotal_usd"], 0.03)
        self.assertIsNone(result["total_cost_usd"])
        self.assertEqual(result["cost_status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
