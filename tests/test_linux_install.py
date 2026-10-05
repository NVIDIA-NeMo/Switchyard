# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
START = "# >>> switchyard codex alias >>>"
END = "# <<< switchyard codex alias <<<"
RC = f"before\n{START}\nalias codex='codex -p sy'\n{END}\nafter\n"


@pytest.fixture
def setup(tmp_path):
    scripts = tmp_path / "repo" / "scripts" / "linux"
    shutil.copytree(REPO / "scripts" / "linux", scripts)
    shutil.copytree(REPO / "scripts" / "config", scripts.parent / "config")
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "SY_HOME": str(home / ".switchyard"),
        "SY_PORT": "4123",
        "XDG_CONFIG_HOME": str(home / ".config"),
        "CODEX_HOME": str(home / ".codex"),
        "TMPDIR": str(tmp_path),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SYSTEMCTL_LOG": str(tmp_path / "systemctl.log"),
        "FAIL_SYSTEMCTL": "",
    }
    stubs = {
        "cargo": "exit 0\n",
        "uname": "echo Linux\n",
        "sleep": "exit 0\n",
        "install": 'printf "#!/bin/sh\\nexit 0\\n" > "$4"\nchmod +x "$4"\n',
        "systemctl": 'echo "$*" >> "$SYSTEMCTL_LOG"\n'
        '[[ "$*" != *"$FAIL_SYSTEMCTL"* || -z "$FAIL_SYSTEMCTL" ]]\n',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text("#!/bin/bash\nset -eu\n" + body)
        stub.chmod(0o755)
    return scripts, home, bin_dir, env


def run(setup, script, *args):
    scripts, _, _, env = setup
    return subprocess.run(
        ["bash", str(scripts / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh"])
@pytest.mark.parametrize("args", [["--dryrun"], ["--help"], ["-n"], ["--dry-run", "extra"], [""]])
def test_bad_arguments_leave_files_unchanged(setup, script, args):
    _, home, _, env = setup
    rc = home / ".bashrc"
    rc.write_text(RC)
    result = run(setup, script, *args)
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert rc.read_text() == RC
    assert not Path(env["SYSTEMCTL_LOG"]).exists()


@pytest.mark.parametrize("failure", ["mktemp", "awk", "missing-end", "second-missing-end"])
def test_uninstall_failure_preserves_shell_files(setup, failure):
    _, home, bin_dir, _ = setup
    contents = RC
    if failure == "missing-end":
        contents = RC.replace(END + "\n", "")
    elif failure == "second-missing-end":
        contents += f"{START}\nalias codex='codex -p sy'\ntail\n"
    else:
        (bin_dir / failure).write_text("#!/bin/bash\nexit 1\n")
        (bin_dir / failure).chmod(0o755)
    for name in [".zshrc", ".bashrc"]:
        (home / name).write_text(contents)
    result = run(setup, "uninstall.sh")
    assert result.returncode != 0
    for name in [".zshrc", ".bashrc"]:
        assert (home / name).read_text() == contents


def test_uninstall_cleans_profile_and_aliases_before_systemctl_failure(setup):
    _, home, _, env = setup
    env["FAIL_SYSTEMCTL"] = "--user"
    profile = Path(env["CODEX_HOME"]) / "sy.config.toml"
    profile.parent.mkdir()
    profile.write_text("old profile\n")
    for name in [".zshrc", ".bashrc"]:
        (home / name).write_text(RC)
    result = run(setup, "uninstall.sh")
    assert result.returncode != 0
    assert not profile.exists()
    for name in [".zshrc", ".bashrc"]:
        assert (home / name).read_text() == "before\nafter\n"


@pytest.mark.parametrize("script", ["install.sh", "uninstall.sh"])
def test_dry_run_leaves_files_unchanged(setup, script):
    _, home, _, env = setup
    (home / ".bashrc").write_text(RC)
    result = run(setup, script, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert (home / ".bashrc").read_text() == RC
    assert sorted(path.name for path in home.iterdir()) == [".bashrc"]
    assert not Path(env["SYSTEMCTL_LOG"]).exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("SY_HOME", "/tmp/with space"),
        ("SY_HOME", "/tmp/line\nbreak"),
        ("SY_HOME", "/tmp/control\x01"),
        ("SY_HOME", "/tmp/backslash\\"),
        ("SY_PORT", "bad"),
        ("SY_PORT", "4123\n"),
    ],
)
def test_invalid_unit_values_fail_before_install(setup, key, value):
    _, home, _, env = setup
    env[key] = value
    result = run(setup, "install.sh")
    assert result.returncode != 0
    assert key in result.stderr
    assert not list(home.iterdir())


def test_reinstall_restarts_service_and_keeps_user_config(setup):
    _, home, _, env = setup
    (home / ".bashrc").write_text("user settings\n")
    assert run(setup, "install.sh").returncode == 0
    config = Path(env["SY_HOME"]) / "composite.toml"
    config.write_text("user config\n")
    env["SY_PORT"] = "5000"
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert config.read_text() == "user config\n"
    assert (home / ".bashrc").read_text() == "user settings\n"
    assert not (home / ".zshrc").exists()
    profile = Path(env["CODEX_HOME"]) / "sy.config.toml"
    assert "127.0.0.1:5000/v1" in profile.read_text()
    backups = list(profile.parent.glob("sy.config.toml.switchyard-backup.*"))
    assert len(backups) == 1
    assert "127.0.0.1:4123/v1" in backups[0].read_text()
    calls = Path(env["SYSTEMCTL_LOG"]).read_text().splitlines()
    assert (
        calls
        == [
            "--user daemon-reload",
            "--user enable switchyard.service",
            "--user restart switchyard.service",
            "--user is-active --quiet switchyard.service",
        ]
        * 2
    )


def test_failed_start_leaves_existing_profile_unchanged(setup):
    _, _, _, env = setup
    env["FAIL_SYSTEMCTL"] = "is-active"
    profile = Path(env["CODEX_HOME"]) / "sy.config.toml"
    profile.parent.mkdir()
    profile.write_text("user profile\n")
    result = run(setup, "install.sh")
    assert result.returncode != 0
    assert "journalctl" in result.stderr
    assert profile.read_text() == "user profile\n"


def test_default_make_only_prints_help(setup):
    _, home, _, env = setup
    result = subprocess.run(
        ["make", "-f", str(REPO / "Makefile")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "install-linux-dry-run" in result.stdout
    assert not list(home.iterdir())
