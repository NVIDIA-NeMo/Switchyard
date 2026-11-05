# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print a native route fragment for an existing Switchyard TOML configuration."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--route-id", default="weight-space")
    parser.add_argument("--cost-weight", type=float, default=0.75)
    args = parser.parse_args()
    weights = json.loads(args.weights.read_text())
    if len(args.targets) != len(weights["output_bias"]) or len(set(args.targets)) != len(args.targets):
        parser.error("targets must be unique and match checkpoint output order")
    if not math.isfinite(args.cost_weight) or not 0 <= args.cost_weight <= 1_000_000:
        parser.error("cost weight must be finite and between zero and 1000000")
    key = json.dumps(args.route_id)
    print(f"[routes.{key}]")
    print(f"id = {key}")
    print('type = "weight_space"')
    print(f"targets = {json.dumps(args.targets)}")
    print(f"cost_weight = {args.cost_weight}")
    print(f"[routes.{key}.weights]")
    for name, value in weights.items():
        print(f"{name} = {json.dumps(value, allow_nan=False)}")


if __name__ == "__main__":
    main()
