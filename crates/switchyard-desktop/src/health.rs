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
    let url = validate_url(server_url).map_err(std::io::Error::other)?;
    let host = url
        .host_str()
        .ok_or_else(|| std::io::Error::other("Missing server host"))?;
    let authority = match url.host() {
        Some(url::Host::Ipv6(address)) => format!("[{address}]:{}", url.port().unwrap_or(4123)),
        _ => format!("{host}:{}", url.port().unwrap_or(4123)),
    };
    let address = authority
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::other(format!("no address for {authority}")))?;
    let mut stream = TcpStream::connect_timeout(&address, TIMEOUT)?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    stream.write_all(format!("GET /health HTTP/1.0\r\nHost: {authority}\r\n\r\n").as_bytes())?;
    let mut status_line = String::new();
    BufReader::new(stream).read_line(&mut status_line)?;
    Ok(status_line.split_whitespace().nth(1) == Some("200"))
}

pub fn validate_url(server_url: &str) -> Result<url::Url, String> {
    if server_url.trim() != server_url
        || server_url.chars().any(char::is_control)
        || !server_url
            .split_once("://")
            .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("http"))
    {
        return Err(
            "Server URL must start with http:// and contain no whitespace or control characters."
                .into(),
        );
    }
    let url =
        url::Url::parse(server_url).map_err(|error| format!("Invalid server URL: {error}"))?;
    if url.scheme() != "http"
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        return Err("Server URL must be an HTTP base URL with a host and optional port, without credentials, a path, query, or fragment.".into());
    }
    Ok(url)
}

pub fn details(server_url: &str) -> serde_json::Value {
    match request(server_url) {
        Ok(true) => serde_json::json!({"state":"running","reason":null}),
        Ok(false) => {
            serde_json::json!({"state":"unreachable","reason":"The health endpoint did not return HTTP 200."})
        }
        Err(error) => serde_json::json!({"state":"unreachable","reason":error.to_string()}),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn health_urls_match_the_supported_probe_forms() {
        for value in [
            "http://127.0.0.1:4123",
            "http://localhost",
            "http://[::1]:4123/",
        ] {
            assert!(validate_url(value).is_ok(), "{value}");
        }
        for value in [
            "https://localhost",
            "localhost:4123",
            "http://user@localhost",
            "http://localhost/v1",
            "http://localhost?x=1",
            "http://localhost#x",
            " http://localhost",
            "http://local\nhost",
            "http:/localhost",
        ] {
            assert!(validate_url(value).is_err(), "{value}");
        }
    }
}
