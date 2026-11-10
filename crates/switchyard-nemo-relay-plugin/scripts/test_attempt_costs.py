# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check USD accounting with Relay's response codec, without loading a plugin."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from attempt_costs import summarize  # noqa: E402


def event_fixture(*costs):
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
    return events


def report(*costs):
    return summarize(event_fixture(*costs), "workflow")


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


class DecisionCoverageTests(unittest.TestCase):
    def events(self):
        events = event_fixture({"total": 0.03, "currency": "USD"})
        events.extend([
            {"kind": "scope", "uuid": "decision", "parent_uuid": "workflow", "scope_category": "end",
             "metadata": {"switchyard.accounting.unit": "decision_call", "switchyard.accounting.usage_status": "complete"},
             "data": {"type": "message", "role": "assistant", "model": "decision/model", "content": [],
                      "usage": {"input_tokens": 1, "output_tokens": 1, "cost": {"total": 0.01, "currency": "USD"}}}},
            {"kind": "mark", "uuid": "decision-mark", "parent_uuid": "workflow", "name": "switchyard.routing.decision_call"},
        ])
        return events

    def test_completion_and_decision_costs_are_counted_once(self):
        events = self.events()
        events[-2]["metadata"]["switchyard.accounting.schema_version"] = 1
        result = summarize(events, "workflow")
        self.assertEqual(result["total_cost_usd"], 0.04)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(len(result["decisions"]), 1)
        self.assertEqual(summarize(events + events, "workflow"), result)

    def test_missing_decision_or_coverage_mark_keeps_total_incomplete(self):
        for dropped in ["decision", "decision-mark"]:
            with self.subTest(dropped=dropped):
                events = [e for e in self.events() if e["uuid"] != dropped]
                result = summarize(events, "workflow")
                self.assertFalse(result["coverage_complete"])
                self.assertIsNone(result["total_cost_usd"])

    def test_wrong_parent_cannot_be_replaced_by_matching_session_metadata(self):
        events = self.events()
        decision = next(e for e in events if e["uuid"] == "decision")
        decision["parent_uuid"] = "different-workflow"
        decision["metadata"]["session_id"] = "workflow"
        result = summarize(events, "workflow")
        self.assertEqual(result["known_cost_subtotal_usd"], 0.03)
        self.assertEqual(result["decisions"], [])
        self.assertIsNone(result["total_cost_usd"])

    def test_unattributed_decision_is_not_assigned_to_the_workflow(self):
        events = self.events()
        decision = next(e for e in events if e["uuid"] == "decision")
        decision["parent_uuid"] = "unknown-root"
        events = [e for e in events if e["uuid"] != "decision-mark"]
        result = summarize(events, "workflow")
        self.assertEqual(result["known_cost_subtotal_usd"], 0.03)
        self.assertEqual(result["decisions"], [])
        self.assertEqual(result["unattributed_decisions"], [{"uuid": "decision", "context_status": "unattributed"}])
        self.assertIsNone(result["total_cost_usd"])


class CompatibilityTests(unittest.TestCase):
    def test_versioned_and_legacy_attempts_keep_accounting_results(self):
        events = event_fixture({"total": 0.02}, {"total": 0.03})
        expected = summarize(events, "workflow")
        for event in [events[1], events[3]]:
            event["metadata"].update({
                "switchyard.accounting.schema_version": 1, "future_field": "ignored",
            })
        events.append(events[3].copy())
        events.append({
            **events[2], "uuid": "outer", "name": "outer", "parent_uuid": "workflow",
            "metadata": {"switchyard.accounting.unit": "summary"},
        })
        result = summarize(events, "workflow")
        self.assertEqual(result, expected)
        self.assertEqual(set(result), {
            "workflow_uuid", "attempts", "decisions", "unattributed_decisions",
            "coverage_complete", "cost_status",
            "known_cost_subtotal_usd", "total_cost_usd",
        })
        self.assertEqual(result["total_cost_usd"], 0.05)
        self.assertEqual([r["uuid"] for r in result["attempts"]], ["attempt-1", "attempt-2"])

    def test_missing_consumption_and_reported_zero_remain_distinct(self):
        events = event_fixture({})
        attempt = events[-1]
        attempt["metadata"]["switchyard.accounting.usage_status"] = "missing"
        attempt["data"].pop("usage")
        missing = summarize(events, "workflow")
        self.assertTrue(missing["coverage_complete"])
        self.assertEqual(missing["known_cost_subtotal_usd"], 0.0)
        self.assertIsNone(missing["total_cost_usd"])
        attempt["data"]["usage"] = {"cost": {
            "total": 0.0, "currency": "USD", "source": "provider_reported",
        }}
        zero = summarize(events, "workflow")
        self.assertEqual(zero["total_cost_usd"], 0.0)
        self.assertEqual(zero["attempts"][0]["usage_status"], "missing")
        self.assertEqual(zero["cost_status"], "complete")


if __name__ == "__main__":
    unittest.main()
