# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# syntax=docker/dockerfile:1.7

# Production image for the standalone `switchyard-server` proxy.
#
# The Dockerfiles under `benchmark/` build the Python launcher and an
# unoptimised server for benchmark harnesses. This one builds only the release
# proxy and ships it on a slim runtime with no toolchain attached.
#
#   docker build -t switchyard-server:0.2.0 .
#   docker run --rm -p 4000:4000 \
#     -v "$PWD/routes.toml:/etc/switchyard/routes.toml:ro" \
#     -e OPENROUTER_API_KEY \
#     switchyard-server:0.2.0 --config /etc/switchyard/routes.toml
#
# CPU baseline: `.cargo/config.toml` compiles x86_64 with `-C
# target-cpu=x86-64-v3`, so the resulting binary needs an AVX2-class CPU
# (Haswell 2013+), and aarch64 with `-C target-cpu=neoverse-n1`. This matches
# the published wheels documented in INSTALLATION.md.

ARG RUST_VERSION=1.96.1
ARG DEBIAN_RELEASE=bookworm

########################################
# Build stage
########################################
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder

# `aws-lc-rs`, pulled in by rustls, builds native code and needs cmake plus a
# libclang for its bindgen step. Everything else in the dependency graph is
# pure Rust: reqwest is configured for rustls, so no OpenSSL headers.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake \
        clang \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy only what the server's dependency graph needs to resolve. Cargo parses
# every workspace manifest even for `-p switchyard-server`, so all of `crates`
# comes along; the Python package and test corpus do not.
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY .cargo ./.cargo
COPY crates ./crates

# The cache mounts make incremental rebuilds cheap. The binary is copied out of
# the mounted target directory in the same layer, because cache mounts are not
# present in the resulting image.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --locked --release -p switchyard-server \
    && install -Dm0755 target/release/switchyard-server /out/switchyard-server

########################################
# Runtime stage
########################################
FROM debian:${DEBIAN_RELEASE}-slim AS runtime

ARG SWITCHYARD_VERSION=0.2.0

LABEL org.opencontainers.image.title="switchyard-server" \
      org.opencontainers.image.description="Rust proxy for LLM traffic: routing, translation and metrics" \
      org.opencontainers.image.version="${SWITCHYARD_VERSION}" \
      org.opencontainers.image.source="https://github.com/NVIDIA-NeMo/Switchyard" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="NVIDIA Corporation"

# ca-certificates is required to reach HTTPS upstreams through rustls.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 switchyard \
    && useradd --system --uid 65532 --gid switchyard --no-create-home switchyard

COPY --from=builder /out/switchyard-server /usr/local/bin/switchyard-server

# A read-only root filesystem is the intended deployment posture, so keep the
# only writable expectation on /tmp.
ENV HOME=/tmp \
    RUST_LOG=switchyard_server=info,libsy=info

USER 65532:65532
EXPOSE 4000

# The server traps SIGTERM and drains in-flight requests for --shutdown-timeout
# (30s default), which lines up with the Kubernetes termination grace period.
ENTRYPOINT ["switchyard-server"]
CMD ["--config", "/etc/switchyard/routes.toml"]
