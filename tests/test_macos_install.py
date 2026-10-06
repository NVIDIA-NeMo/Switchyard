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


def test_missing_codex_config_creates_only_standalone_profile(setup):
    _, _, switchyard_home, env = setup
    env["SY_PORT"] = "5123"
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    codex = Path(env["CODEX_HOME"])
    assert sorted(path.name for path in codex.iterdir()) == ["sy.config.toml"]
    expected = read_config(REPO / "scripts" / "config" / "codex.sy.toml").replace(
        "@SY_PORT@", "5123"
    )
    assert read_config(codex / "sy.config.toml") == expected
    assert read_config(switchyard_home / "composite.toml") == read_config(
        REPO / "scripts" / "config" / "composite.toml"
    )


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


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh"])
@pytest.mark.parametrize(
    "original",
    [
        'model_provider = "sy"\n[model_providers."sy"]\nname = "Old"\n',
        'developer_instructions = """\nmodel = "example"\n'
        "# >>> switchyard sy profile >>>\n[model_providers.sy]\n"
        '# <<< switchyard sy profile <<<\n"""\n'
        '# >>> switchyard sy profile >>>\n[profiles.sy]\nmodel_provider = "sy"\n'
        "# <<< switchyard sy profile <<<\n",
        "invalid TOML that must be left alone\n",
    ],
)
def test_scripts_leave_main_config_and_legacy_files_untouched(setup, script, original):
    _, _, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    files = {
        "config.toml": original,
        "config.toml.direct": 'model = "direct"\n',
        "config.sy.toml": 'model = "previous routed config"\n',
    }
    for name, content in files.items():
        (codex / name).write_text(content)
    (codex / "sy.config.toml").write_text('model = "old profile"\n')

    result = run(setup, script)

    assert result.returncode == 0, result.stderr
    for name, content in files.items():
        assert (codex / name).read_text() == content
    assert not list(codex.glob("config.toml.switchyard-*"))
    if script == "uninstall.sh":
        assert not (codex / "sy.config.toml").exists()
    else:
        assert (codex / "sy.config.toml").read_text() == read_config(
            REPO / "scripts" / "config" / "codex.sy.toml"
        ).replace("@SY_PORT@", env.get("SY_PORT", "4123"))
        backups = list(codex.glob("sy.config.toml.switchyard-backup.*"))
        assert len(backups) == 1
        assert backups[0].read_text() == 'model = "old profile"\n'


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh"])
def test_scripts_dry_run_does_not_create_files(setup, script):
    scripts, home, _, env = setup
    result = subprocess.run(
        ["bash", str(scripts / script), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "would" in result.stdout
    assert list(home.iterdir()) == []


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh"])
@pytest.mark.parametrize(
    "args",
    [
        ["--dryrun"],
        ["unknown"],
        [""],
        ["--dry-run", "extra"],
        ["--dry-run", "--dry-run"],
    ],
)
def test_scripts_reject_unknown_arguments_before_any_work(setup, script, args):
    scripts, home, _, env = setup
    profile = Path(env["CODEX_HOME"]) / "sy.config.toml"
    plist = home / "Library" / "LaunchAgents" / "com.nvidia.switchyard.server.plist"
    for path in (profile, plist):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("existing file\n")
    before = {
        path.relative_to(home): path.read_bytes() for path in home.rglob("*") if path.is_file()
    }
    command_log = home.parent / "commands.log"
    env["SY_TEST_COMMAND_LOG"] = str(command_log)
    for stub in Path(env["PATH"].split(":")[0]).iterdir():
        stub.write_text(
            stub.read_text().replace(
                "set -eu\n", 'set -eu\nprintf "%s\\n" "$0" >> "$SY_TEST_COMMAND_LOG"\n', 1
            )
        )

    result = subprocess.run(
        ["bash", str(scripts / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert result.stdout == ""
    assert not command_log.exists()
    after = {
        path.relative_to(home): path.read_bytes() for path in home.rglob("*") if path.is_file()
    }
    assert after == before
