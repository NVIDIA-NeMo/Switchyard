# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Price reconciled calls under one token-rate snapshot, retaining billed costs separately."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def price(capture: dict, rates: dict) -> dict:
    """Require explicit model rates and cache counts for every captured call."""
    if capture.get("status") != "captured_usage_reconciles_with_session_stats":
        raise ValueError("usage capture must reconcile before pricing")
    calls = capture["calls"]
    if len(calls) != capture["call_count"]:
        raise ValueError("capture count differs from its calls")
    normalized = []
    charges = []
    for call in calls:
        model = call["model"]
        if model not in rates:
            raise ValueError(f"no explicit rates for model {model}")
        rate = rates[model]
        fields = ["prompt_tokens", "completion_tokens", "cached_tokens", "cache_creation_tokens",
                  "cache_creation_1h_tokens"]
        values = [call.get(field) for field in fields]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("missing or invalid token counts")
        prompt, output, read, written, written_1h = values
        fresh = prompt - read - written
        if fresh < 0 or written_1h > written:
            raise ValueError("inconsistent cache token counts")
        components = {"input": fresh, "output": output, "cache_read": read,
                      "cache_write": written - written_1h}
        if written_1h:
            components["cache_write_1h"] = written_1h
        for field in components:
            value = rate.get(field)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"missing or invalid {field} rate for {model}")
        usd = math.fsum(count * rate[field] for field, count in components.items()) / 1_000_000
        if not math.isfinite(usd):
            raise ValueError("nonfinite token cost")
        normalized.append(usd)
        cost = call.get("cost")
        if cost is not None and cost.get("source") == "provider_reported":
            charge = cost.get("usd")
            if type(charge) not in (int, float) or not math.isfinite(charge) or charge < 0:
                raise ValueError("invalid provider charge")
            charges.append(charge)
    return {
        "session_id": capture["session_id"],
        "call_count": len(calls),
        "token_snapshot_total_usd": math.fsum(normalized),
        "token_snapshot_routing_usd": math.fsum(
            usd for usd, call in zip(normalized, calls, strict=True) if call.get("tier") == "classifier"
        ),
        "provider_reported_subtotal_usd": math.fsum(charges),
        "calls_without_provider_charge": len(calls) - len(charges),
        "provider_reported_total_usd": math.fsum(charges) if len(charges) == len(calls) else None,
        "scope": "captured calls only",
        "limits": [
            "Token snapshot cost and provider charges use separate accounting definitions.",
            "This calculation cannot establish capture of all billed attempts or missing upstream cache detail.",
            "Do not mark an evaluation cost_complete from this calculation alone.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("rates", type=Path)
    args = parser.parse_args()
    capture_bytes, rate_bytes = args.capture.read_bytes(), args.rates.read_bytes()
    result = price(json.loads(capture_bytes), json.loads(rate_bytes))
    result["capture_sha256"] = hashlib.sha256(capture_bytes).hexdigest()
    result["pricing_sha256"] = hashlib.sha256(rate_bytes).hexdigest()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
