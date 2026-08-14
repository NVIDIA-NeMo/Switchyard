# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the standalone switchyard-server binary against the workspace lockfile.
FROM rust:1.96-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
RUN cargo build --release --locked --package switchyard-server

# Runtime image: binary only, runs unprivileged. Mount a TOML deployment at
# /etc/switchyard/config.toml and pass provider API keys as env vars.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 switchyard
COPY --from=builder /app/target/release/switchyard-server /usr/local/bin/switchyard-server
USER switchyard
EXPOSE 4000
ENTRYPOINT ["switchyard-server"]
CMD ["--config", "/etc/switchyard/config.toml"]
