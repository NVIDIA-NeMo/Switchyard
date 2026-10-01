// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Liveness probe against the local server's `/health` endpoint.
//!
//! The request is written by hand over TCP. The server is always on loopback
//! over plain HTTP, so a full HTTP client would buy nothing here.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// How long the probe waits before calling the server unreachable. Short,
/// because it runs on the thread that draws the menu.
const TIMEOUT: Duration = Duration::from_millis(500);

/// Whether the server answered its health check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerStatus {
    Running,
    Stopped,
}

/// Probes `GET /health`, treating any failure as a stopped server.
pub fn probe(server_url: &str) -> ServerStatus {
    match request(server_url) {
        Ok(true) => ServerStatus::Running,
        _ => ServerStatus::Stopped,
    }
}

fn request(server_url: &str) -> std::io::Result<bool> {
    // Strip the scheme and any path, then default the port.
    let rest = server_url.rsplit("://").next().unwrap_or(server_url);
    let mut authority = rest.split('/').next().unwrap_or(rest).to_string();
    if !authority.contains(':') {
        authority.push_str(":4123");
    }

    let address = authority
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::other(format!("no address for {authority}")))?;
    let mut stream = TcpStream::connect_timeout(&address, TIMEOUT)?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    // HTTP/1.0 needs no Host header and closes the connection for us.
    stream.write_all(b"GET /health HTTP/1.0\r\n\r\n")?;

    let mut status_line = String::new();
    BufReader::new(stream).read_line(&mut status_line)?;
    Ok(status_line.contains(" 200"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Serves one canned response, then closes.
    fn serve(response: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    #[test]
    fn reports_running_only_on_a_200() {
        let ok = serve("HTTP/1.1 200 OK\r\n\r\n");
        let down = serve("HTTP/1.1 503 Service Unavailable\r\n\r\n");

        assert_eq!(probe(&ok), ServerStatus::Running);
        assert_eq!(probe(&down), ServerStatus::Stopped);
    }

    #[test]
    fn reports_stopped_when_nothing_is_listening() {
        // Binding then dropping yields a port with no listener.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        drop(listener);

        assert_eq!(
            probe(&format!("http://127.0.0.1:{port}")),
            ServerStatus::Stopped
        );
    }

    #[test]
    fn accepts_a_url_with_a_path_or_no_port() {
        let ok = serve("HTTP/1.1 200 OK\r\n\r\n");

        assert_eq!(probe(&format!("{ok}/v1")), ServerStatus::Running);
        assert_eq!(probe("http://localhost"), ServerStatus::Stopped);
    }
}
