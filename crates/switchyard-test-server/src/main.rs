// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs the mock LLM server as a standalone process.

use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use switchyard_test_server::MockLlmServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let default_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
    let addr = match std::env::args().nth(1) {
        Some(value) => value.parse()?,
        None => default_addr,
    };
    let server = MockLlmServer::builder().bind_addr(addr).start().await?;
    println!("{}", server.base_url());
    tokio::signal::ctrl_c().await?;
    Ok(())
}
