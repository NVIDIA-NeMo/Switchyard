<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Offline TypeSafe/Jev Replay

Use the replay tool to evaluate recorded TypeSafe routing evidence without credentials, network
access, or new model calls:

```bash
uv run --no-sync python benchmark/typesafe_replay.py \
  benchmark/typesafe/fixtures/synthetic-routing.json
```

Pass `--output report.json` to save the stable, machine-readable report. Replaying the same fixture
always produces the same bytes.

## Fixture format

Every fixture is one JSON document with `schema_version: 1`. It records:

- a suite name and pinned Jev model identifier;
- two or more candidate labels and descriptions;
- the confidence threshold and fallback target;
- one or more unique candidate orders per case, with a complete probability distribution for each;
- measured quality in `[0, 1]` and nonnegative cost for every candidate outcome.

The replay averages the recorded distributions, normalizes the average, and chooses the largest
probability. Configured candidate order breaks an exact tie. Confidence is the selected
probability's improvement over a uniform distribution, scaled to `[0, 1]`. A result below
`base_threshold` selects `default_target`.

`best_fixed_target` is the candidate with the highest total quality when used for every case. Lower
cost and then configured candidate order break ties. The report compares the routed totals with
that fixed baseline. Positive `quality_regret_vs_best_fixed` means routing lost quality; a negative
value means it beat every fixed target. `cost_delta_vs_best_fixed` uses the same sign convention.
`order_sensitive_case_count` counts cases whose per-order top candidate changes, while
`maximum_probability_movement` reports the largest probability shift seen across orders.

## Sanitization boundary

The schema deliberately excludes request text, credentials, response bodies, and provider error
details. Use opaque case IDs. Candidate descriptions and model identifiers can still contain
deployment information, so replace them with non-sensitive equivalents before committing a
fixture. The checked-in fixture is entirely synthetic and does not claim production routing
quality.

This is an offline evaluator, not a provider collector. Collection of live evidence must remain an
explicit, separately reviewed workflow.
