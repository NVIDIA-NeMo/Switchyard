# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drift checks for the Getting Started guide."""

from pathlib import Path


def test_getting_started_documents_current_paths() -> None:
    root = Path(__file__).resolve().parents[2]
    guide = (root / "docs" / "getting_started.md").read_text()
    readme = (root / "README.md").read_text()
    server = (root / "docs" / "operations" / "local_evaluation.md").read_text()

    assert (
        guide.index("## Route a request")
        < guide.index("## Call your model")
        < guide.index("## Next steps")
    )
    assert guide.split("```python\n", 1)[1].split("```", 1)[0] == (
        readme.split("```python\n", 1)[1].split("```", 1)[0]
    )
    assert 'uv add "nemo-switchyard==0.3.0"' in guide
    assert 'messages=outcome.request["messages"]' in guide
    assert "Selected: fast" in guide
    assert "Critical tool error: strong" in guide
    assert "operations/local_evaluation.md" in guide
    assert "switchyard launch" not in guide
    assert "nemo-switchyard[cli]" not in guide
    assert "cargo install --locked switchyard-server" not in guide
    assert "cargo install --locked switchyard-server --version 0.3.0" in server
    assert "switchyard-server --config routes.toml --dry-run" in server
    assert "not a supported production deployment path" in server
    assert (
        server.index("## Run the server")
        < server.index("## Verify routing is active")
        < server.index("## Compare configurations")
    )
    assert "x-model-router-selected-model" in server
    assert "/v1/stats" in server
    assert "--routing-log-file" in server

    for legacy_command in (
        "switchyard configure",
        "switchyard serve",
        "switchyard verify",
        "switchyard launch",
        "--routing-profiles",
        "routes.yaml",
    ):
        assert legacy_command not in guide + server


# TODO: Execute the published-package examples in isolated CI environments.
