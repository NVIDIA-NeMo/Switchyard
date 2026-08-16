# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage image for the standalone switchyard-server binary.
# Builds from this workspace (not crates.io) so the image matches the
# checked-out commit. Prefer this over cargo install for reproducible deploys.
#
#   docker build -t switchyard-server:local .
#   docker run --rm -p 4000:4000 \
#     -e API_KEY \
#     -v "$PWD/deploy/examples/routes.passthrough.toml:/etc/switchyard/routes.toml:ro" \
#     switchyard-server:local

ARG RUST_VERSION=1.96.1

FROM rust:${RUST_VERSION}-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --locked --release -p switchyard-server \
    && strip /src/target/release/switchyard-server

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 switchyard
COPY --from=build /src/target/release/switchyard-server /usr/local/bin/switchyard-server
USER switchyard
WORKDIR /home/switchyard
EXPOSE 4000
ENTRYPOINT ["/usr/local/bin/switchyard-server"]
CMD ["--config", "/etc/switchyard/routes.toml", "--host", "0.0.0.0", "--port", "4000"]
