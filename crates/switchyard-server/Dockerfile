# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG RUST_VERSION=1.96.1
FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --locked --release -p switchyard-server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/release/switchyard-server /usr/local/bin/switchyard-server

ARG IMAGE_VERSION=local
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/NVIDIA-NeMo/Switchyard" \
      org.opencontainers.image.description="Switchyard's libsy-based HTTP server" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

USER 65532:65532
EXPOSE 4000
ENTRYPOINT ["/usr/local/bin/switchyard-server"]
