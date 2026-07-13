# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate static redirects from the former GitHub Pages site to Fern."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

CANONICAL_BASE = "https://docs.nvidia.com/nemo/switchyard"

LEGACY_REDIRECTS = {
    "": "/home",
    "getting_started": "/get-started/getting-started",
    "known_issues": "/get-started/known-issues-in-0-1-0",
    "guides/agent_launchers": "/guides/agent-launchers",
    "skill_distillation": "/guides/skill-distillation",
    "core_concepts": "/concepts/core-concepts",
    "architecture": "/concepts/architecture",
    "routing_algorithms/overview": "/routing/overview",
    "routing_algorithms/random_routing": "/routing/random-routing",
    "routing_algorithms/llm_classifier_routing": "/routing/llm-classifier-routing",
    "routing_algorithms/sticky_routing": "/routing/sticky-routing",
    "routing_algorithms/stage_router_routing": "/routing/stage-router-routing",
    "operations/context_window": "/operations/context-window-handling",
    "cli_reference": "/reference/cli-reference",
}


def render_redirect_page(destination: str) -> str:
    """Return a static redirect page for one canonical Fern destination."""
    target = f"{CANONICAL_BASE}{destination}"
    escaped_target = html.escape(target, quote=True)
    javascript_target = json.dumps(target)
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="refresh" content="0; url={escaped_target}" />
    <link rel="canonical" href="{escaped_target}" />
    <title>NeMo Switchyard documentation moved</title>
    <script>
      window.location.replace(
        {javascript_target} + window.location.search + window.location.hash
      );
    </script>
  </head>
  <body>
    <p>This documentation moved to <a href="{escaped_target}">{escaped_target}</a>.</p>
  </body>
</html>
"""


def generate_redirect_site(output: Path) -> None:
    """Write one GitHub Pages redirect document for every former MkDocs route."""
    output.mkdir(parents=True, exist_ok=True)
    (output / ".nojekyll").touch()
    for legacy_path, destination in LEGACY_REDIRECTS.items():
        page = output / legacy_path / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_redirect_page(destination), encoding="utf-8")


def main() -> None:
    """Parse CLI arguments and generate the redirect-only GitHub Pages site."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_redirect_site(args.output)


if __name__ == "__main__":
    main()
