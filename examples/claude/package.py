# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained plugin and local marketplace. Does not publish it."""

import json
import platform
import shutil
import subprocess
from pathlib import Path

PLUGIN = Path(__file__).resolve().parent
ROOT = PLUGIN.parents[1]


def main():
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise SystemExit("The prebuilt plugin currently supports Linux x86_64 only")
    subprocess.run(
        ["cargo", "build", "--locked", "--release", "-p", "switchyard-runner", "--example", "switchyard-claude"],
        cwd=ROOT,
        check=True,
    )
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=ROOT, text=True,
    ))
    binary = Path(metadata["target_directory"]) / "release/examples/switchyard-claude"
    marketplace = ROOT / "dist" / "switchyard-claude-linux-x86_64"
    if marketplace.exists():
        shutil.rmtree(marketplace)
    destination = marketplace / "switchyard"
    destination.mkdir(parents=True, exist_ok=True)
    for relative in [".claude-plugin/plugin.json", "hooks/hooks.json", "hooks/register.mjs", "switchyard.toml", "README.md"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLUGIN / relative, target)
    (destination / "bin").mkdir(exist_ok=True)
    shutil.copy2(binary, destination / "bin/switchyard-claude")
    for name in ["LICENSE", "NOTICE"]:
        shutil.copy2(ROOT / name, destination / name)
    manifest = marketplace / ".claude-plugin" / "marketplace.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "name": "switchyard-local",
        "owner": {"name": "NVIDIA"},
        "description": "Switchyard native routing for Claude Code",
        "plugins": [{"name": "switchyard", "source": "./switchyard"}],
    }, indent=2) + "\n")
    # Tar preserves the helper's executable bit when users extract the marketplace.
    archive = shutil.make_archive(str(marketplace), "gztar", marketplace.parent, marketplace.name)
    print(archive)


if __name__ == "__main__":
    main()
