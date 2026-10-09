// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module checks the local server's `/health` endpoint.
//!
//! The request is written by hand over TCP. The server is always on loopback
//! over plain HTTP, so a full HTTP client would buy nothing here.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// TIMEOUT limits how long a health check can delay a usage refresh.
const TIMEOUT: Duration = Duration::from_millis(500);

/// ServerStatus records whether the server answered its health check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerStatus {
    Running,
    Stopped,
}

/// This function probes `GET /health`, treating any failure as a stopped server.
pub fn probe(server_url: &str) -> ServerStatus {
    match request(server_url) {
        Ok(true) => ServerStatus::Running,
        _ => ServerStatus::Stopped,
    }
}

fn request(server_url: &str) -> std::io::Result<bool> {
    let authority = authority(server_url);
    let address = authority
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::other(format!("no address for {authority}")))?;
    let mut stream = TcpStream::connect_timeout(&address, TIMEOUT)?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    stream.write_all(b"GET /health HTTP/1.0\r\n\r\n")?;
    let mut status_line = String::new();
    BufReader::new(stream).read_line(&mut status_line)?;
    Ok(status_line.contains(" 200"))
}

fn authority(server_url: &str) -> String {
    let rest = server_url.rsplit("://").next().unwrap_or(server_url);
    let mut authority = rest.split('/').next().unwrap_or(rest).to_string();
    let host_end = authority.rfind(']').map_or(0, |index| index + 1);
    if !authority[host_end..].contains(':') {
        authority.push_str(":4123");
    }

    authority
}
