# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.PHONY: install-linux install-linux-dry-run uninstall-linux

## Install the Switchyard background server as a systemd user service.
install-linux:
	@scripts/linux/install.sh

## Print what install-linux would do, without changing anything.
install-linux-dry-run:
	@scripts/linux/install.sh --dry-run

## Remove the systemd user service, the sy Codex profile, and the codex alias.
uninstall-linux:
	@scripts/linux/uninstall.sh
