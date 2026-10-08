# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
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
        "install": 'if [[ "$3" == */switchyard-menubar ]]; then printf "#!/bin/bash\\n[[ \\\"${FAIL_TOML_VALIDATION:-0}\\\" != 1 ]]\\n" > "$4"; else printf "#!/bin/sh\\nexit 0\\n" > "$4"; fi\nchmod +x "$4"\n',
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

    menubar = home / "Library" / "LaunchAgents" / "com.nvidia.switchyard.menubar.plist"
    menubar_root = ET.parse(menubar).getroot()
    menubar_entries = list(menubar_root.find("dict"))
    menubar_values = {
        menubar_entries[index].text: menubar_entries[index + 1]
        for index in range(0, len(menubar_entries), 2)
    }
    menubar_arguments = [element.text for element in menubar_values["ProgramArguments"]]
    assert menubar_arguments == [
        str(switchyard_home / "bin" / "switchyard-menubar"),
        str(switchyard_home / "menubar.toml"),
    ]


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


def test_menu_bar_settings_escape_sy_home_as_toml_strings(setup):
    _, home, _, env = setup
    switchyard_home = home / 'Switchyard "quoted" \\ folder'
    env["SY_HOME"] = str(switchyard_home)

    result = run(setup, "install.sh")

    assert result.returncode == 0, result.stderr
    settings = tomllib.loads((switchyard_home / "menubar.toml").read_text())
    assert settings["routing_log"] == str(switchyard_home / "routing.jsonl")
    assert settings["config_file"] == str(switchyard_home / "composite.toml")


def test_invalid_generated_codex_toml_does_not_replace_active_config(setup):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    original = 'model = "gpt-5.6-sol"\n'
    (codex / "config.toml").write_text(original)
    env["FAIL_TOML_VALIDATION"] = "1"

    result = run(setup, "install.sh")

    assert result.returncode != 0
    assert "leaving" in result.stderr
    assert (codex / "config.toml").read_text() == original


@pytest.mark.parametrize(
    "provider_header",
    [
        "[model_providers.\"sy\"]",
        "[model_providers . 'sy']",
        '[ "model_providers" . "sy" ]',
    ],
)
def test_provider_table_with_quoted_key_is_replaced_and_shared_template_is_used(
    setup, provider_header
):
    _, home, switchyard_home, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    config = codex / "config.toml"
    original = f'''theme = "dark"

{provider_header} # old provider
name = "Old"
base_url = "http://old"

[other]
model_provider = "sy"
'''
    config.write_text(original)
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    generated = read_config(codex / "config.sy.toml")
    parsed = tomllib.loads(generated)
    assert generated.count("[model_providers.sy]") == 1
    assert 'name = "Old"' not in generated
    assert '[other]\nmodel_provider = "sy"' in generated
    assert parsed["model_providers"]["sy"]["name"] == "Switchyard"
    assert read_config(config) == generated
    assert read_config(switchyard_home / "composite.toml") == read_config(
        REPO / "scripts" / "config" / "composite.toml"
    )
    assert read_config(codex / "config.toml.direct") == original


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_top_level_provider_keeps_existing_snapshot(setup, quote):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    (codex / "config.toml").write_text(f"model_provider = {quote}sy{quote} # routed\n")
    snapshot = codex / "config.toml.direct"
    snapshot.write_text("original direct config\n")
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert snapshot.read_text() == "original direct config\n"
    assert (codex / "config.sy.toml").is_file()


def test_uninstall_preserves_routed_config_before_restoring_snapshot(setup):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    current = "model_provider = 'sy' # routed\nuser_setting = \"keep me\"\n"
    (codex / "config.toml").write_text(current)
    (codex / "config.toml.direct").write_text("model = \"original\"\n")
    result = run(setup, "uninstall.sh")
    assert result.returncode == 0, result.stderr
    assert (codex / "config.toml").read_text() == 'model = "original"\n'
    backups = list(codex.glob("config.toml.switchyard-current.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == current


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
