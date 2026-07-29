# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG RUST_VERSION=1.96.1
FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /opt/switchyard

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN cargo build --locked --release -p switchyard-server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/switchyard/target/release/switchyard-server \
    /usr/local/bin/switchyard-server

ENTRYPOINT ["switchyard-server"]
