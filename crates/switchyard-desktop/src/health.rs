// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module checks the configured server's `/health` endpoint.
//!
//! The health check supports plain HTTP base URLs.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// TIMEOUT limits each connection, read, and write operation.
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
    let authority = authority(server_url).map_err(std::io::Error::other)?;
    // `localhost` can resolve to `::1` before `127.0.0.1`, so try every address.
    let mut stream = Err(std::io::Error::other(format!("no address for {authority}")));
    for address in authority.to_socket_addrs()? {
        stream = TcpStream::connect_timeout(&address, TIMEOUT);
        if stream.is_ok() {
            break;
        }
    }
    let mut stream = stream?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    stream.write_all(format!("GET /health HTTP/1.0\r\nHost: {authority}\r\n\r\n").as_bytes())?;
    let mut status_line = String::new();
    BufReader::new(stream).read_line(&mut status_line)?;
    Ok(status_line.split_whitespace().nth(1) == Some("200"))
}

fn authority(server_url: &str) -> Result<String, String> {
    let url = validate_url(server_url)?;
    // Url removes an explicit HTTP port of 80, so a missing parsed port still means 80.
    let port = url.port_or_known_default().ok_or("Missing server port")?;
    Ok(format!(
        "{}:{port}",
        &url[url::Position::BeforeHost..url::Position::AfterHost]
    ))
}

pub(crate) fn validate_url(server_url: &str) -> Result<url::Url, String> {
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
        return Err(
            "Server URL must be an HTTP base URL with a host and optional port, \
             without credentials, a path, query, or fragment."
                .into(),
        );
    }
    Ok(url)
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

    #[test]
    fn health_checks_use_http_ports_including_explicit_port_80() {
        for (url, expected) in [
            ("http://127.0.0.1:80", "127.0.0.1:80"),
            ("http://localhost", "localhost:80"),
            ("http://localhost:4123", "localhost:4123"),
            ("http://[::1]:80/", "[::1]:80"),
            ("http://[::1]:4567/", "[::1]:4567"),
        ] {
            assert_eq!(authority(url).expect("valid URL"), expected);
        }
    }

    // The server listens on IPv4 only, while `localhost` often resolves to `::1` first.
    #[test]
    fn probe_reaches_an_ipv4_server_through_localhost() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("address").port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut request = [0; 256];
            let _ = std::io::Read::read(&mut stream, &mut request);
            stream
                .write_all(b"HTTP/1.0 200 OK\r\n\r\n")
                .expect("respond");
        });
        assert_eq!(
            probe(&format!("http://localhost:{port}")),
            ServerStatus::Running
        );
        server.join().expect("server");
    }
}
