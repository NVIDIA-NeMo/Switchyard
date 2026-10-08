# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
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
    shutil.copy(REPO / "Cargo.toml", scripts.parents[1] / "Cargo.toml")
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
        "codesign": "exit 0\n",
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
        str(home / "Applications" / "Switchyard.app" / "Contents" / "MacOS" / "Switchyard"),
        str(switchyard_home / "menubar.toml"),
    ]


def test_missing_codex_config_creates_only_standalone_profile(setup):
    _, home, _, _ = setup
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    codex = home / ".codex"
    expected = (
        (REPO / "scripts" / "config" / "codex.sy.toml").read_text().replace("@SY_PORT@", "4123")
    )
    assert (codex / "sy.config.toml").read_text() == expected
    assert {path.name for path in codex.iterdir()} == {"sy.config.toml"}


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


def test_reinstall_backs_up_profile_and_keeps_user_settings(setup):
    _, _, switchyard_home, env = setup
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    profile = Path(env["CODEX_HOME"]) / "sy.config.toml"
    # Older installs used this route ID; reinstall must update the profile and back it up.
    original = profile.read_text().replace("composite-gpt-6-sol-gpt-6-luna", "switchyard")
    profile.write_text(original)
    server_config = switchyard_home / "composite.toml"
    menu_settings = switchyard_home / "menubar.toml"
    server_config.write_text("user server settings\n")
    menu_settings.write_text("user menu settings\n")
    env["SY_PORT"] = "5000"

    result = run(setup, "install.sh")

    assert result.returncode == 0, result.stderr
    assert "127.0.0.1:5000/v1" in profile.read_text()
    assert tomllib.loads(profile.read_text())["model"] == "composite-gpt-6-sol-gpt-6-luna"
    backups = list(profile.parent.glob("sy.config.toml.switchyard-backup.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original
    assert server_config.read_text() == "user server settings\n"
    assert menu_settings.read_text() == "user menu settings\n"


@pytest.mark.parametrize(
    "provider_header",
    [
        '[model_providers."sy"]',
        "[model_providers . 'sy']",
        '[ "model_providers" . "sy" ]',
    ],
)
def test_existing_codex_config_is_preserved_and_shared_templates_are_used(setup, provider_header):
    _, home, switchyard_home, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    config = codex / "config.toml"
    original = f"""theme = "dark"

{provider_header} # old provider
name = "Old"
base_url = "http://old"

[other]
model_provider = "sy"
"""
    config.write_text(original)
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert config.read_text() == original
    profile = (codex / "sy.config.toml").read_text()
    expected = (
        (REPO / "scripts" / "config" / "codex.sy.toml").read_text().replace("@SY_PORT@", "4123")
    )
    assert profile == expected
    assert tomllib.loads(profile)["model_providers"]["sy"]["name"] == "Switchyard"
    assert not (codex / "config.sy.toml").exists()
    assert (switchyard_home / "composite.toml").read_text() == (
        REPO / "scripts" / "config" / "composite.toml"
    ).read_text()
    assert not (codex / "config.toml.direct").exists()


def test_existing_snapshot_and_routed_config_are_preserved(setup):
    quote = "'"
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    (codex / "config.toml").write_text(f"model_provider = {quote}sy{quote} # routed\n")
    snapshot = codex / "config.toml.direct"
    snapshot.write_text("original direct config\n")
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    assert snapshot.read_text() == "original direct config\n"
    assert (codex / "config.toml").read_text() == f"model_provider = {quote}sy{quote} # routed\n"
    assert not (codex / "config.sy.toml").exists()
    assert (codex / "sy.config.toml").is_file()


def test_uninstall_preserves_routed_config_before_restoring_snapshot(setup):
    _, home, _, env = setup
    codex = Path(env["CODEX_HOME"])
    codex.mkdir()
    current = "model_provider = 'sy' # routed\nuser_setting = \"keep me\"\n"
    (codex / "config.toml").write_text(current)
    (codex / "config.toml.direct").write_text('model = "original"\n')
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


def test_app_bundle_has_a_launcher_and_source_update_command(setup):
    _, home, switchyard_home, _ = setup
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    contents = home / "Applications" / "Switchyard.app" / "Contents"
    root = ET.parse(contents / "Info.plist").getroot()
    entries = list(root.find("dict"))
    values = {entries[i].text: entries[i + 1] for i in range(0, len(entries), 2)}
    assert values["CFBundleIdentifier"].text == "com.nvidia.switchyard"
    assert values["CFBundleExecutable"].text == "Switchyard"
    assert os.access(contents / "MacOS" / "Switchyard", os.X_OK)
    update = contents / "Resources" / "Update.command"
    assert os.access(update, os.X_OK)
    home_line = next(
        line for line in update.read_text().splitlines() if line.startswith("export SY_HOME=")
    )
    assert shlex.split(home_line.split("=", 1)[1])[0] == str(switchyard_home)
    for path in [update, contents / "MacOS" / "Switchyard"]:
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


def test_app_launcher_preserves_quotes_and_shell_metacharacters(setup):
    """The launcher must pass shell syntax in the settings path as literal text."""
    _, home, _, env = setup
    switchyard_home = home / "Switchyard ' $(touch unsafe)"
    env["SY_HOME"] = str(switchyard_home)
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    executable = (
        home / "Applications" / "Switchyard.app" / "Contents" / "MacOS" / "switchyard-menubar"
    )
    output = home / "arguments"
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > ' + shlex.quote(str(output)) + "\n")
    executable.chmod(0o755)
    launcher = executable.with_name("Switchyard")
    result = subprocess.run(["bash", str(launcher)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().strip() == str(switchyard_home / "menubar.toml")
    assert not (home / "unsafe").exists()


def test_uninstall_removes_the_installed_app_bundle(setup):
    _, home, _, _ = setup
    assert run(setup, "install.sh").returncode == 0
    app = home / "Applications" / "Switchyard.app"
    assert app.is_dir()
    result = run(setup, "uninstall.sh")
    assert result.returncode == 0, result.stderr
    assert not app.exists()


def test_installer_builds_in_source_checkout_with_an_explicit_target_directory(setup):
    """The build must use the checkout target directory even if CARGO_TARGET_DIR differs."""
    scripts, home, _, env = setup
    recorded = home / "build.txt"
    cargo = Path(env["PATH"].split(":")[0]) / "cargo"
    cargo.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$PWD" "$@" > ' + shlex.quote(str(recorded)) + "\n"
    )
    cargo.chmod(0o755)
    env["CARGO_TARGET_DIR"] = str(home / "unrelated-target")
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    lines = recorded.read_text().splitlines()
    assert lines[0] == str(scripts.parents[1])
    target = lines.index("--target-dir")
    assert lines[target + 1] == str(scripts.parents[1] / "target")


def test_named_profile_keeps_route_and_update_settings(setup):
    _, home, _, env = setup
    env["SY_PROFILE"] = "stage-gpt-sonnet"
    env["SY_MODEL"] = 'stage/provider-"model"'
    result = run(setup, "install.sh")
    assert result.returncode == 0, result.stderr
    profile = home / ".codex" / "stage-gpt-sonnet.config.toml"
    assert tomllib.loads(profile.read_text())["model"] == env["SY_MODEL"]
    bundle = home / "Applications" / "Switchyard.app" / "Contents"
    assert (bundle / "MacOS" / "switchyard-server").is_file()
    update = (bundle / "Resources" / "Update.command").read_text()
    for key in ["SY_PROFILE", "SY_MODEL"]:
        line = next(line for line in update.splitlines() if line.startswith(f"export {key}="))
        assert shlex.split(line.split("=", 1)[1])[0] == env[key]
    assert run(setup, "uninstall.sh").returncode == 0
    assert not profile.exists()


@pytest.mark.parametrize("name", ["../escape", "", "a/b", "a\nname", "x" * 129])
def test_invalid_profile_name_rejects_before_install(setup, name):
    _, home, switchyard_home, env = setup
    env["SY_PROFILE"] = name
    result = run(setup, "install.sh")
    assert result.returncode == 2
    assert "SY_PROFILE" in result.stderr
    assert not switchyard_home.exists()
    assert not (home / "Applications").exists()


@pytest.mark.parametrize(
    "model", ["", "model\nname", "model\rname", "model\x01name", "model\x7fname"]
)
def test_invalid_model_rejects_before_install(setup, model):
    _, home, switchyard_home, env = setup
    env["SY_MODEL"] = model
    result = run(setup, "install.sh")
    assert result.returncode == 2
    assert "SY_MODEL" in result.stderr
    assert not switchyard_home.exists()
    assert not (home / "Applications").exists()
