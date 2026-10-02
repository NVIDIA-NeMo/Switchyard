# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    scripts = tmp_path / "repo" / "scripts" / "macos"
    shutil.copytree(REPO / "scripts" / "macos", scripts)
    shutil.copy(REPO / "scripts" / "common.sh", scripts.parent / "common.sh")
    shutil.copytree(REPO / "scripts" / "config", scripts.parent / "config")
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    switchyard_home = home / "Switchyard & Routing <local>"
    env = {
        **os.environ,
        "HOME": str(home),
        "SY_HOME": str(switchyard_home),
        "CODEX_HOME": str(home / ".codex"),
        "TMPDIR": str(tmp_path),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
    }
    stubs = {
        "cargo": "exit 0\n",
        "uname": "echo Darwin\n",
        "install": 'printf "#!/bin/sh\\nexit 0\\n" > "$4"\nchmod +x "$4"\n',
        "launchctl": '[[ "$1" != print ]]\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/bash\nset -eu\n" + body)
        stub.chmod(0o755)
    return scripts, home, switchyard_home, env


def run(setup, script):
    scripts, _, _, env = setup
    return subprocess.run(
        ["bash", str(scripts / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def read_config(path):
    return path.read_text()


def test_install_escapes_switchyard_path_in_launch_agent(setup):
    _, home, switchyard_home, _ = setup
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    plist = home / "Library" / "LaunchAgents" / "com.nvidia.switchyard.server.plist"
    root = ET.parse(plist).getroot()
    entries = list(root.find("dict"))
    values = {entries[index].text: entries[index + 1] for index in range(0, len(entries), 2)}
    program_arguments = [element.text for element in values["ProgramArguments"]]
    assert program_arguments[0] == str(switchyard_home / "bin" / "switchyard-server")
    assert program_arguments[2] == str(switchyard_home / "composite.toml")
    assert program_arguments[8] == str(switchyard_home / "routing.jsonl")
    assert values["StandardOutPath"].text == str(switchyard_home / "logs" / "server.log")
    assert values["StandardErrorPath"].text == str(switchyard_home / "logs" / "server.err.log")


def test_missing_codex_config_creates_routed_config_and_empty_backup(setup):
    _, home, _, _ = setup
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    codex = home / ".codex"
    assert 'model_provider = "sy"' in read_config(codex / "config.sy.toml")
    assert read_config(codex / "config.toml") == read_config(codex / "config.sy.toml")
    assert (codex / "config.toml.direct").read_text() == ""


def test_install_prints_profile_usage_without_editing_shell_files(setup):
    _, home, _, _ = setup
    zshrc = home / ".zshrc"
    bashrc = home / ".bashrc"
    zshrc.write_text("zsh settings\n")
    bashrc.write_text("bash settings\n")

    result = run(setup, "install.sh")

    assert result.returncode == 0, result.stderr
    assert "Use it with: codex -p sy" in result.stdout
    assert zshrc.read_text() == "zsh settings\n"
    assert bashrc.read_text() == "bash settings\n"


def test_provider_table_with_comment_is_replaced_and_shared_template_is_used(setup):
    _, home, switchyard_home, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    config = codex / "config.toml"
    original = '''theme = "dark"

[model_providers.sy] # old provider
name = "Old"
base_url = "http://old"

[other]
model_provider = "sy"
'''
    config.write_text(original)
    (codex / "config.toml.direct").write_text(original)
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    generated = read_config(codex / "config.sy.toml")
    assert generated.count("[model_providers.sy]") == 1
    assert 'name = "Old"' not in generated
    assert '[other]\nmodel_provider = "sy"' in generated
    assert read_config(config) == generated
    assert read_config(switchyard_home / "composite.toml") == read_config(
        REPO / "scripts" / "config" / "composite.toml"
    )
    assert read_config(codex / "config.toml.direct") == original
    backups = list(codex.glob("config.toml.switchyard-backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original


def test_provider_table_marker_keeps_existing_snapshot(setup):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    (codex / "config.toml").write_text('[model_providers.sy]\nname = "Switchyard"\n')
    snapshot = codex / "config.toml.direct"
    snapshot.write_text("original direct config\n")
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert snapshot.read_text() == "original direct config\n"
    assert (codex / "config.sy.toml").is_file()


def test_provider_setting_without_table_gets_snapshotted(setup):
    _, _, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    original = 'model_provider = "sy"\n'
    (codex / "config.toml").write_text(original)
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert (codex / "config.toml.direct").read_text() == original


def test_uninstall_preserves_routed_config_before_restoring_snapshot(setup):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    current = '[model_providers.sy]\nname = "Switchyard"\nuser_setting = "keep me"\n'
    (codex / "config.toml").write_text(current)
    (codex / "config.toml.direct").write_text("model = \"original\"\n")
    result = run(setup, "uninstall.sh")
    assert result.returncode == 0, result.stderr
    assert (codex / "config.toml").read_text() == 'model = "original"\n'
    backups = list(codex.glob("config.toml.switchyard-current.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == current
