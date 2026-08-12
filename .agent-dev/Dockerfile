# syntax=docker/dockerfile:1.7

FROM --platform=$TARGETPLATFORM rust:1.96.1-bookworm AS rust-toolchain

FROM --platform=$TARGETPLATFORM python:3.12.13-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG UV_VERSION=0.11.8

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/home/sandbox/workspace/target \
    VIRTUAL_ENV=/home/sandbox/workspace/.venv \
    UV_PROJECT_ENVIRONMENT=/home/sandbox/workspace/.venv \
    UV_CACHE_DIR=/opt/switchyard/uv-cache \
    UV_LINK_MODE=copy \
    PATH="/home/sandbox/workspace/.venv/bin:/usr/local/cargo/bin:${PATH}"

COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    iproute2 \
    libssl-dev \
    netcat-openbsd \
    openssh-client \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && rustup component add clippy rustfmt rust-analyzer \
    && cargo --version | grep 'cargo 1.96.1' \
    && uv --version | grep "uv ${UV_VERSION}"

WORKDIR /home/sandbox/workspace
COPY . .

RUN git rev-parse --is-inside-work-tree \
    && git remote get-url origin \
    && test -z "$(git status --porcelain)" \
    && cargo fetch --locked \
    && uv sync --locked \
    && cargo fmt --all --check

RUN useradd -m -s /bin/bash sandbox \
    && mkdir -p /home/sandbox/job /home/sandbox/.cache /sandbox/job /opt/switchyard \
    && chown -R sandbox:sandbox \
        /home/sandbox \
        /sandbox \
        /opt/switchyard \
        /usr/local/cargo \
        /usr/local/rustup

ENV HOME=/home/sandbox

USER sandbox
WORKDIR /home/sandbox/workspace
