# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that CLI constants and runtime strings use the `switchyard` command name.

The PyPI distribution is `nemo-switchyard`, but user-facing CLI names,
configuration directories, and runtime model descriptions stay `switchyard`.
"""

from __future__ import annotations

import os
from unittest.mock import patch


def test_config_dir_env_var_name() -> None:
    from switchyard.cli.config.user_config import CONFIG_DIR_ENV_VAR
    assert CONFIG_DIR_ENV_VAR == "SWITCHYARD_CONFIG_DIR", (
        f"Expected SWITCHYARD_CONFIG_DIR, got {CONFIG_DIR_ENV_VAR}"
    )


def test_no_tui_env_var_name() -> None:
    from switchyard.cli.tui.terminal_capabilities import NO_TUI_ENV_VAR
    assert NO_TUI_ENV_VAR == "SWITCHYARD_NO_TUI", (
        f"Expected SWITCHYARD_NO_TUI, got {NO_TUI_ENV_VAR}"
    )


def test_config_dir_path_uses_switchyard() -> None:
    from switchyard.cli.config.user_config import get_user_config_dir

    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in ("XDG_CONFIG_HOME", "NEMO_SWITCHYARD_CONFIG_DIR", "SWITCHYARD_CONFIG_DIR")
    }
    with patch.dict(os.environ, clean_env, clear=True):
        result = get_user_config_dir()

    assert result.name == "switchyard", (
        f"Config dir should end with 'switchyard', got '{result.name}'"
    )
    assert "nemo-switchyard" not in str(result), (
        f"Config dir path should not contain 'nemo-switchyard', got '{result}'"
    )


def test_argparse_prog_is_switchyard() -> None:
    # The parser prog= determines the command name shown in --help and errors.

    # Import the CLI module and find the parser builder
    # The ArgumentParser is built inside main() / build_parser() — search for it
    import inspect

    import switchyard.cli.switchyard_cli as cli_module
    source = inspect.getsource(cli_module)
    assert 'prog="nemo-switchyard"' not in source, (
        "argparse prog still set to 'nemo-switchyard'; should be 'switchyard'"
    )


def test_claude_code_launcher_model_description_uses_switchyard() -> None:
    import inspect

    import switchyard.cli.launchers.claude_code_launcher as launcher_module
    source = inspect.getsource(launcher_module)
    assert "nemo-switchyard" not in source or "Routed via nemo-switchyard" not in source, (
        "claude_code_launcher still uses 'nemo-switchyard' in user-visible model description"
    )
    # More precise check
    assert "Routed via nemo-switchyard" not in source, (
        "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION still says 'nemo-switchyard'"
    )
