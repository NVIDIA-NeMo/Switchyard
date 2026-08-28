# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI dispatcher for Harbor task converters.

Usage::

    craft-taskgen-convert --adapter search-native \\
        --candidates-dir tasks/candidates/search/ \\
        --manifest repos/manifest.json \\
        --output-dir harbor-tasks/craft-search/ \\
        --limit 10

Register a new adapter by:
1. Creating `adapters/<name>/converter.py` with a `run_convert(...)` function.
2. Adding an entry to ``_ADAPTERS`` below that describes its CLI args and how
   to call ``run_convert``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AdapterSpec:
    """Describes an adapter's CLI surface and entry point."""

    name: str
    help: str
    add_args: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], Any]


def _search_native_add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidates-dir",
        required=True,
        help="Root dir with per-repo candidate subdirs ({repo}/{uuid}.json)",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to repos/manifest.json (repo_name -> {url, commit})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Harbor task dirs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many tasks (0 = all)",
    )


def _search_native_run(args: argparse.Namespace) -> Any:
    from craft_taskgen.adapters.search_native.converter import run_convert

    return run_convert(
        candidates_dir=args.candidates_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        limit=args.limit,
    )


def _planning_add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidates-dir",
        required=True,
        help="Directory of bootstrapped planning candidate JSONs ({task_name}.json each)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for Harbor planning task dirs",
    )
    parser.add_argument(
        "--repo-cache",
        default=None,
        help="Local git clone cache (default: /tmp/craft-taskgen-repos)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many tasks (0 = all)",
    )


def _planning_run(args: argparse.Namespace) -> Any:
    from craft_taskgen.adapters.planning.converter import run_convert

    return run_convert(
        candidates_dir=args.candidates_dir,
        output_dir=args.output_dir,
        repo_cache=args.repo_cache,
        limit=args.limit,
    )


_ADAPTERS: dict[str, AdapterSpec] = {
    "search-native": AdapterSpec(
        name="search-native",
        help="Native Search tasks (builds fresh Dockerfile per task)",
        add_args=_search_native_add_args,
        run=_search_native_run,
    ),
    "planning": AdapterSpec(
        name="planning",
        help="Iterative planning track tasks with binary F2P+P2P reward",
        add_args=_planning_add_args,
        run=_planning_run,
    ),
}


def _top_level_help(adapters: dict[str, AdapterSpec]) -> str:
    lines = [
        "usage: craft-taskgen-convert --adapter <name> [adapter args ...]",
        "",
        "Convert task candidates into Harbor task directories.",
        "",
        "Adapters:",
    ]
    for name in sorted(adapters):
        lines.append(f"  {name:16s} {adapters[name].help}")
    lines.append("")
    lines.append("Use --adapter <name> --help to see adapter-specific arguments.")
    return "\n".join(lines)


def main() -> None:
    # Top-level parser: disable its own -h so --help can reach the adapter parser
    # once an adapter is selected. If no --adapter is given, print our own help.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--adapter",
        choices=sorted(_ADAPTERS.keys()),
        help="Which adapter to run",
    )
    args, remaining = parser.parse_known_args()

    if args.adapter is None:
        print(_top_level_help(_ADAPTERS))
        return

    spec = _ADAPTERS[args.adapter]
    sub_parser = argparse.ArgumentParser(
        prog=f"craft-taskgen-convert --adapter {args.adapter}",
        description=spec.help,
    )
    spec.add_args(sub_parser)
    sub_args = sub_parser.parse_args(remaining)
    spec.run(sub_args)


if __name__ == "__main__":
    main()
