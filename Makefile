# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

.DEFAULT_GOAL := help
.PHONY: help install-linux install-linux-dry-run uninstall-linux

help:
	@echo "install-linux          Install the systemd user service and Codex profile"
	@echo "install-linux-dry-run  Preview installation without changes"
	@echo "uninstall-linux        Remove the service and Codex profile"

## Install the Switchyard background server as a systemd user service.
install-linux:
	@scripts/linux/install.sh

## Print what install-linux would do, without changing anything.
install-linux-dry-run:
	@scripts/linux/install.sh --dry-run

## Remove the systemd user service, the sy Codex profile, and the codex alias.
uninstall-linux:
	@scripts/linux/uninstall.sh
