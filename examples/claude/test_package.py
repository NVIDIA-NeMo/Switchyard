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

    def selected(messages):
        result = run(json.dumps(messages))
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)["model"]

    messages = [{"role": "user", "content": [{"type": "text", "text": "Fix the tests"}]}]
    assert selected(messages) == "claude-sonnet-5-5"
    messages.extend(tool("failure", "Bash", "out of memory", True))
    assert selected(messages) == "claude-opus-5-5"
    for i in range(3):
        messages.extend(tool(f"write-{i}", "Write", "wrote file"))
    assert selected(messages) == "claude-sonnet-5-5"
    assert run("not json").returncode != 0
    assert run("[]", plugin / "missing.toml").returncode != 0
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "invalid.toml"
        invalid.write_text(config.read_text().replace('id = "switchyard"', 'id = "other"'))
        result = run("[]", invalid)
        assert result.returncode != 0
        assert "Missing switchyard route" in result.stderr
    print("Packaged helper: selection, escalation, recovery, and failure checks passed")


if __name__ == "__main__":
    main()
