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
WORKFLOW_ROOT = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW_PATH = WORKFLOW_ROOT / "ci.yml"
FERN_CI_WORKFLOW_PATH = WORKFLOW_ROOT / "fern-docs-ci.yml"
FERN_PREVIEW_BUILD_PATH = WORKFLOW_ROOT / "fern-docs-preview-build.yml"
FERN_PREVIEW_COMMENT_PATH = WORKFLOW_ROOT / "fern-docs-preview-comment.yml"
FERN_PUBLISH_PATH = WORKFLOW_ROOT / "publish-fern-docs.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping used by the Fern project."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _load_workflow(path: Path) -> dict[str, Any]:
    """Load a GitHub workflow without treating the `on` key as a YAML boolean."""
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
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
        if not metadata.get("title"):
            failures.append(f"{path.relative_to(REPO_ROOT)}: missing title")
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


def test_fern_validation_is_part_of_required_ci_success() -> None:
    """The required CI aggregate must fail when reusable Fern validation fails."""
    ci = _load_workflow(CI_WORKFLOW_PATH)
    jobs = ci["jobs"]

    assert jobs["fern-docs"]["uses"] == "./.github/workflows/fern-docs-ci.yml"
    assert "fern-docs" in jobs["ci-success"]["needs"]

    fern_ci = _load_workflow(FERN_CI_WORKFLOW_PATH)
    assert set(fern_ci["on"]) == {"workflow_call"}


def test_fern_preview_uses_trusted_identity_and_pr_scoped_concurrency() -> None:
    """Preview artifacts must not choose the target PR, preview identifier, or tool version."""
    build = _load_workflow(FERN_PREVIEW_BUILD_PATH)
    comment = _load_workflow(FERN_PREVIEW_COMMENT_PATH)
    build_text = FERN_PREVIEW_BUILD_PATH.read_text(encoding="utf-8")
    comment_text = FERN_PREVIEW_COMMENT_PATH.read_text(encoding="utf-8")

    assert build["concurrency"] == {
        "group": "fern-docs-preview-${{ github.event.pull_request.number }}",
        "cancel-in-progress": "true",
    }
    assert comment["concurrency"] == {
        "group": "fern-docs-preview-${{ github.event.workflow_run.pull_requests[0].number }}",
        "cancel-in-progress": "true",
    }
    assert "github.event.workflow_run.pull_requests[0].number" in comment_text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in build_text
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in comment_text
    assert "github.event.repository.default_branch" in comment_text
    assert "working-directory: ./preview-source/docs/fern" in comment_text
    assert '--id "pr-$PR_NUMBER"' in comment_text
    assert ".preview-metadata/pr_number" not in comment_text
    assert ".preview-metadata/head_ref" not in comment_text
    assert '.user.login == "github-actions[bot]"' in comment_text


def test_fern_publish_serializes_runs_and_isolates_write_permission() -> None:
    """Only the redirect deployment may receive repository write permission."""
    workflow = _load_workflow(FERN_PUBLISH_PATH)
    jobs = workflow["jobs"]

    assert workflow["permissions"] == {}
    assert workflow["concurrency"] == {
        "group": "fern-docs-website",
        "cancel-in-progress": "false",
    }
    assert jobs["publish"]["permissions"] == {"contents": "read"}
    assert jobs["redirects"]["permissions"] == {
        "actions": "read",
        "contents": "write",
    }
    assert "DOCS_FERN_TOKEN" in yaml.safe_dump(jobs["publish"])
    assert "DOCS_FERN_TOKEN" not in yaml.safe_dump(jobs["redirects"])


def test_fern_workflows_pin_actions_and_use_supported_node() -> None:
    """Secret-bearing and docs validation workflows use immutable action pins and Node 24."""
    paths = (
        FERN_CI_WORKFLOW_PATH,
        FERN_PREVIEW_BUILD_PATH,
        FERN_PREVIEW_COMMENT_PATH,
        FERN_PUBLISH_PATH,
    )
    action_pattern = re.compile(r"^\s*uses:\s+([^#\s]+)", re.MULTILINE)

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for action in action_pattern.findall(text):
            if action.startswith("./"):
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), f"{path}: {action}"

    for path in (FERN_CI_WORKFLOW_PATH, FERN_PREVIEW_COMMENT_PATH, FERN_PUBLISH_PATH):
        assert 'node-version: "24"' in path.read_text(encoding="utf-8")
