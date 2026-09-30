# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A thin command-line interface to the task evaluation library."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from switchyard import __version__
from switchyard.runner import Runner

from . import Dataset, Result, evaluate, load_harbor
from .evaluate import _aliases, _validate_options, _validate_route


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except RuntimeError as error:
        # Python < 3.13 reports symlink loops as RuntimeError rather than OSError.
        raise ValueError(f"cannot resolve path: {path}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Switchyard task routing on recorded Harbor runs."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Native Switchyard deployment TOML"
    )
    parser.add_argument("--route", required=True, help="Configured route model ID")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="TARGET=PATH",
        help="Recorded Harbor run for a configured target; repeat for each target",
    )
    parser.add_argument(
        "--input-target", required=True, help="Target whose initial agent input the router sees"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New output directory outside all recorded runs"
    )
    parser.add_argument("--dataset", help="Optional task namespace")
    parser.add_argument("--reward-key", default="reward")
    parser.add_argument(
        "--model-alias",
        action="append",
        default=[],
        metavar="RECORDED=CONFIGURED",
        help="Explicit equivalence between recorded and configured model IDs",
    )
    parser.add_argument("--cost-source", choices=("result", "trajectory"), default="result")
    parser.add_argument(
        "--intersection",
        action="store_true",
        help="Explicitly evaluate only tasks valid for every target",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Record rejected inputs; requires --intersection to evaluate remaining tasks",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--timeout", type=float, default=60, help="Whole-decision deadline in seconds"
    )
    args = parser.parse_args(argv)
    paths = {}
    for value in args.run:
        target, separator, path = value.partition("=")
        if not separator or not target or not path or target in paths:
            parser.error("each --run must be a unique TARGET=PATH")
        paths[target] = Path(path)
    aliases = {}
    for value in args.model_alias:
        recorded, separator, configured = value.partition("=")
        if not separator or not recorded or not configured or recorded in aliases:
            parser.error("each --model-alias must be a unique RECORDED=CONFIGURED")
        aliases[recorded] = configured
    try:
        _validate_options(args.concurrency, args.timeout)
        paths = {target: _resolve_path(path) for target, path in paths.items()}
        if args.output.is_symlink():
            raise FileExistsError(f"output already exists: {args.output}")
        output = _resolve_path(args.output)
        if any(output.is_relative_to(path) for path in paths.values()):
            raise ValueError("output directory must be outside every recorded run directory")
        runs = {
            target: load_harbor(
                path,
                target=target,
                dataset=args.dataset,
                reward_key=args.reward_key,
                cost_source=args.cost_source,
                on_error="record" if args.skip_invalid else "raise",
            )
            for target, path in paths.items()
        }
        dataset = Dataset.from_runs(
            runs, input_target=args.input_target, intersection=args.intersection
        )
        del runs  # Dataset owns included trials; release excluded inputs before routing.
        config_bytes = args.config.read_bytes()
        runner = Runner.from_toml(config_bytes.decode("utf-8"))
        _validate_route(dataset, runner, args.route, _aliases(aliases))
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "switchyard_version": __version__,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "route": args.route,
            "runs": {target: str(path) for target, path in paths.items()},
            "input_target": args.input_target,
            "reward_key": args.reward_key,
            "cost_source": args.cost_source,
            "model_aliases": aliases,
            "concurrency": args.concurrency,
            "timeout_seconds": args.timeout,
            "coverage": dict(dataset.coverage),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        with (output / "results.jsonl").open("x", encoding="utf-8") as stream:

            def save(result: Result) -> None:
                stream.write(json.dumps(asdict(result), allow_nan=False) + "\n")
                stream.flush()

            report = asyncio.run(
                evaluate(
                    dataset,
                    runner,
                    route=args.route,
                    concurrency=args.concurrency,
                    timeout=args.timeout,
                    on_result=save,
                    model_aliases=aliases,
                )
            )
        (output / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
        )
    except (OSError, ValueError) as error:
        print(f"switchyard.sim: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "switchyard.sim: interrupted; completed result rows remain in the output directory",
            file=sys.stderr,
        )
        return 130
    print(report.format_text())
    return 0 if report.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
