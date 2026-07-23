# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG RUST_VERSION=1.96.1
FROM rust:${RUST_VERSION}-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /usr/local/bin/

WORKDIR /opt/switchyard
# Hosted runtimes can override the container UID, so keep Python and package imports
# independent of root's home and the copied source tree.
ENV PATH="/opt/switchyard/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/switchyard/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

COPY pyproject.toml uv.lock README.md Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY switchyard ./switchyard
COPY switchyard_rust ./switchyard_rust

RUN uv sync --frozen --no-dev --extra server --extra cli --no-editable

ENTRYPOINT ["switchyard"]
