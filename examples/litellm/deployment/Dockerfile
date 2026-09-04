# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM rust:1.96.1-slim-bookworm AS switchyard-builder

RUN apt-get update \
    && apt-get install --yes --no-install-recommends patchelf python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/maturin \
    && /opt/maturin/bin/pip install --no-cache-dir "maturin==1.14.1"

WORKDIR /src
COPY . .
RUN /opt/maturin/bin/maturin build \
    --release \
    --locked \
    --compatibility manylinux_2_34 \
    --out /wheels \
    --interpreter /usr/bin/python3

FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM ghcr.io/berriai/litellm:v1.97.0

COPY --from=uv /uv /usr/local/bin/uv
RUN --mount=type=bind,from=switchyard-builder,source=/wheels,target=/wheels \
    uv pip install --python /app/.venv/bin/python --no-deps /wheels/*.whl

COPY examples/litellm/src /app/switchyard-plugin/src
ENV PYTHONPATH=/app/switchyard-plugin/src

RUN python -c "from importlib.metadata import version; \
from switchyard.libsy import algorithms; \
from switchyard_litellm import SwitchyardRoutingPlugin; \
assert version('litellm') == '1.97.0'; \
SwitchyardRoutingPlugin(algorithms.random(['smoke']))"

RUN addgroup -S -g 10001 litellm \
    && adduser -S -D -u 10001 -G litellm -h /home/litellm litellm
USER litellm
