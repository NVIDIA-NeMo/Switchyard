# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a Codex model catalog for Harbor benchmark containers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from codex_model_catalog_lib import (
    _build_codex_model_catalog,
    _codex_model_display_name,
)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codex_model_catalog")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--codex-bin", default="codex")
    ns = parser.parse_args(argv)

    entries = [
        (
            model,
            f"{_codex_model_display_name(model)} (Switchyard)",
            f"Routed through Switchyard to {model}.",
        )
        for model in _unique(ns.model)
    ]
    catalog = _build_codex_model_catalog(ns.codex_bin, entries)
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(catalog, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
