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
        < guide.index("## Use libsy with your client")
        < guide.index("## Next steps")
    )
    for step in ("CallModel", "CallDecision", "Done"):
        assert f"Step::{step}(" in readme
        for example in guide.split("```rust\n")[1:3]:
            assert f"Step::{step}(" in example.split("```", 1)[0]
    assert "cargo add switchyard-libsy switchyard-protocol --git" in guide
    assert "--rev 5415db205c1f4db7e079731e7bf33a6609c67c8e" in guide
    assert "call.respond(client.call(request)).await?" in guide
    assert "struct OpenRouterClient" in guide
    assert '.post("https://openrouter.ai/api/v1/chat/completions")' in guide
    assert "You do not have to use Switchyard's HTTP client." in guide
    assert "cargo add switchyard-llm-client" not in guide
    assert "client.call(outcome.request).await?" in guide
    assert "```python" not in guide + readme
    assert "Selected: Efficient" in guide
    assert "Selected: Capable" in guide
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


# TODO: Execute the documented Rust examples in isolated CI environments.
