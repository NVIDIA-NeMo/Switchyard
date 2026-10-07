# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Paths shared by install.sh and uninstall.sh.

# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SERVER_LABEL="com.nvidia.switchyard.server"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
