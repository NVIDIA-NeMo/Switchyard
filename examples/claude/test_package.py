# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the packaged native helper: python test_package.py <plugin directory>."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def tool(call_id, name, text, failed=False):
    return [
        {"role": "assistant", "content": [
            {"type": "tool_call", "id": call_id, "name": name, "arguments": {}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_call_id": call_id,
             "content": [{"type": "text", "text": text}], "is_error": failed}
        ]},
    ]


def main():
    plugin = Path(sys.argv[1]).resolve()
    config = plugin / "switchyard.toml"

    def run(messages, config_path=config):
        return subprocess.run(
            [plugin / "bin/switchyard-claude", config_path],
            input=messages, text=True, capture_output=True, timeout=3, check=False,
            cwd="/tmp", env={"PATH": "/usr/bin:/bin"},
        )

    checkpoint = None
    step = 0

    def selected(messages):
        nonlocal checkpoint, step
        payload = {"version": 1, "session_id": "package-test", "decision_id": str(step),
                   "messages": messages, "checkpoint": checkpoint}
        result = run(json.dumps(payload))
        assert result.returncode == 0, result.stderr
        decision = json.loads(result.stdout)
        checkpoint = decision["checkpoint"]
        payload["checkpoint"] = checkpoint
        repeated = run(json.dumps(payload))
        assert repeated.returncode == 0, repeated.stderr
        assert json.loads(repeated.stdout)["checkpoint"] == checkpoint
        step += 1
        return decision["model"]

    messages = [{"role": "user", "content": [{"type": "text", "text": "Fix the tests"}]}]
    assert selected(messages) == "claude-sonnet-5-5"
    for cycle in range(3):
        messages.extend(tool(f"failure-{cycle}", "Bash", "out of memory", True))
        assert selected(messages) == "claude-opus-5-5"
        for i in range(3):
            messages.extend(tool(f"write-{cycle}-{i}", "Write", "wrote file"))
        assert selected(messages) == "claude-opus-5-5"  # First held recovery request.
        assert selected(messages) == "claude-opus-5-5"  # Second held recovery request.
        assert selected(messages) == "claude-sonnet-5-5"
    assert run("not json").returncode != 0
    assert run("[]", plugin / "missing.toml").returncode != 0
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "invalid.toml"
        invalid.write_text(config.read_text().replace('id = "switchyard"', 'id = "other"'))
        result = run(json.dumps({"version": 1, "session_id": "test", "decision_id": "0",
                                 "messages": [], "checkpoint": None}), invalid)
        assert result.returncode != 0
        assert "Missing switchyard route" in result.stderr
    print("Packaged helper: three stateful escalation/recovery cycles, replay, and failure checks passed")


if __name__ == "__main__":
    main()
