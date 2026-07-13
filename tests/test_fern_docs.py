# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Fern navigation, links, metadata, and legacy redirects."""

from __future__ import annotations

import re
import runpy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
FERN_ROOT = DOCS_ROOT / "fern"
NAVIGATION_PATH = FERN_ROOT / "versions" / "nightly.yml"
DOCS_CONFIG_PATH = FERN_ROOT / "docs.yml"
GENERATOR_PATH = FERN_ROOT / "generate_legacy_redirect_site.py"
BASE_PATH = "/nemo/switchyard"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping used by the Fern project."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _slug(value: str) -> str:
    """Return the Fern-style slug for the navigation labels used in this site."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _published_routes() -> set[str]:
    """Derive unversioned published routes from nightly navigation."""
    navigation = _load_yaml(NAVIGATION_PATH)["navigation"]
    routes: set[str] = set()
    for item in navigation:
        if "page" in item:
            routes.add(f"/{_slug(item['page'])}")
            continue

        assert "section" in item
        assert "path" not in item, "section paths must not duplicate child page paths"
        section_slug = _slug(item["section"])
        for child in item["contents"]:
            assert "page" in child
            routes.add(f"/{section_slug}/{_slug(child['page'])}")
    return routes


def test_internal_mdx_links_resolve_to_navigation_routes() -> None:
    """Every root-relative MDX link must name a route that Fern publishes."""
    routes = _published_routes()
    broken: list[str] = []
    patterns = (r"\]\((/[^)\s]+)", r'href="(/[^"]+)"')
    for path in sorted(DOCS_ROOT.rglob("*.mdx")):
        if FERN_ROOT in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            for target in re.findall(pattern, text):
                route = target.split("#", 1)[0]
                if route not in routes:
                    broken.append(f"{path.relative_to(REPO_ROOT)}: {target}")
    assert not broken, "broken Fern links:\n" + "\n".join(broken)


def test_published_pages_have_descriptions_and_titled_callouts() -> None:
    """Published pages must retain SEO descriptions and converted admonition titles."""
    failures: list[str] = []
    for path in sorted(DOCS_ROOT.rglob("*.mdx")):
        if FERN_ROOT in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        metadata = yaml.safe_load(frontmatter)
        if not metadata.get("description"):
            failures.append(f"{path.relative_to(REPO_ROOT)}: missing description")
        for tag in re.findall(r"<(?:Note|Warning)(?:\s+[^>]*)?>", text):
            if "title=" not in tag:
                failures.append(f"{path.relative_to(REPO_ROOT)}: untitled {tag}")
    assert not failures, "\n".join(failures)


def test_fern_redirects_cover_every_github_pages_route() -> None:
    """Fern must map every former MkDocs route to its current navigation route."""
    generator = runpy.run_path(str(GENERATOR_PATH))
    legacy_redirects = generator["LEGACY_REDIRECTS"]
    configured = {
        item["source"]: item["destination"]
        for item in _load_yaml(DOCS_CONFIG_PATH)["redirects"]
    }
    routes = _published_routes()

    for legacy_path, destination in legacy_redirects.items():
        assert destination in routes
        full_destination = f"{BASE_PATH}{destination}"
        if not legacy_path:
            assert configured[f"{BASE_PATH}/index.html"] == full_destination
            continue
        source = f"{BASE_PATH}/{legacy_path}"
        assert configured[source] == full_destination
        assert configured[f"{source}/index.html"] == full_destination


def test_legacy_github_pages_site_generation(tmp_path: Path) -> None:
    """The generated GitHub Pages site must redirect every legacy route to Fern."""
    generator = runpy.run_path(str(GENERATOR_PATH))
    generator["generate_redirect_site"](tmp_path)

    assert (tmp_path / ".nojekyll").is_file()
    for legacy_path, destination in generator["LEGACY_REDIRECTS"].items():
        page = tmp_path / legacy_path / "index.html"
        target = f"{generator['CANONICAL_BASE']}{destination}"
        assert page.is_file()
        content = page.read_text(encoding="utf-8")
        assert f'<link rel="canonical" href="{target}" />' in content
        assert "window.location.search + window.location.hash" in content
