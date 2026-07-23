# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Components-v2 profile config loading and shared request input bindings.

Exposes two Rust-owned surfaces:

* the erased serving API (``ProfileConfigDocument``, ``ProfileConfigPlan``,
  ``Profile``) for loading config files and running any profile through
  ``run``; and
* shared request input classes (``ProfileInput`` and
  ``ProfileRequestMetadata``) used by both Rust and Python profile runtimes.

Names are resolved lazily from the native extension on first access.
"""

from switchyard_rust.core import _load_native

_PROFILE_EXPORTS = (
    # Erased serving surface.
    "ProfileConfigDocument",
    "ProfileConfigPlan",
    "Profile",
    "load_profile_config",
    "parse_profile_config_path",
    "parse_profile_config_str",
    "ProfileInput",
    "ProfileRequestMetadata",
    "is_subagent_request",
)
__all__ = _PROFILE_EXPORTS


def __getattr__(name: str) -> object:
    if name in _PROFILE_EXPORTS:
        return getattr(_load_native(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
