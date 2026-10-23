// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module checks the configured server's `/health` endpoint.
//!
//! The health check supports plain HTTP base URLs.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::{OnceLock, mpsc};
use std::time::{Duration, Instant};

/// TIMEOUT limits the whole probe, including DNS and every connection attempt.
const TIMEOUT: Duration = Duration::from_millis(500);
const MAX_STATUS_LINE: usize = 1024;

/// ServerStatus records whether the server answered its health check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServerStatus {
    Running,
    Stopped,
}

/// Callers retain one Probe so repeated checks share one worker and at most one waiting request.
/// A stuck OS name lookup occupies the worker, but callers still stop waiting after 500 ms.
/// Dropping the Probe closes the queue without waiting for the worker to finish.
#[derive(Default)]
pub struct Probe {
    worker: OnceLock<Result<mpsc::SyncSender<Request>, String>>,
}

struct Request {
    authority: String,
    deadline: Instant,
    reply: mpsc::SyncSender<std::io::Result<bool>>,
}

impl Probe {
    /// This method checks `/health`, treating any failure as a stopped server.
    pub fn probe(&self, server_url: &str) -> ServerStatus {
        match self.request(server_url) {
            Ok(true) => ServerStatus::Running,
            _ => ServerStatus::Stopped,
        }
    }

    /// This method includes the reason when a health check cannot confirm HTTP 200.
    pub fn details(&self, server_url: &str) -> serde_json::Value {
        match self.request(server_url) {
            Ok(true) => serde_json::json!({"state":"running","reason":null}),
            Ok(false) => {
                serde_json::json!({"state":"unreachable","reason":"The health endpoint did not return HTTP 200."})
            }
            Err(error) => serde_json::json!({"state":"unreachable","reason":error.to_string()}),
        }
    }

    fn request(&self, server_url: &str) -> std::io::Result<bool> {
        let deadline = Instant::now() + TIMEOUT;
        let authority = authority(server_url).map_err(std::io::Error::other)?;
        let worker = self
            .worker
            .get_or_init(|| {
                start_worker(|authority| authority.to_socket_addrs().map(Iterator::collect))
            })
            .as_ref()
            .map_err(|error| std::io::Error::other(error.clone()))?;
        let (reply, receiver) = mpsc::sync_channel(1);
        worker
            .try_send(Request {
                authority,
                deadline,
                reply,
            })
            .map_err(|_| std::io::Error::other("A previous health check is still in progress."))?;
        let result = receiver
            .recv_timeout(remaining(deadline)?)
            .map_err(|_| timeout())?;
        remaining(deadline)?;
        result
    }
}

// The queue retains at most one waiting request. Expired requests never start DNS or connect.
fn start_worker(
    resolve: fn(&str) -> std::io::Result<Vec<SocketAddr>>,
) -> Result<mpsc::SyncSender<Request>, String> {
    let (sender, receiver) = mpsc::sync_channel::<Request>(1);
    std::thread::Builder::new()
        .name("switchyard-health".into())
        .spawn(move || {
            while let Ok(request) = receiver.recv() {
                let result = remaining(request.deadline)
                    .and_then(|_| request_status(&request.authority, request.deadline, resolve));
                let _ = request.reply.send(result);
            }
        })
        .map_err(|error| error.to_string())?;
    Ok(sender)
}

fn timeout() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "The health check exceeded its 500 ms deadline.",
    )
}

fn remaining(deadline: Instant) -> std::io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|duration| !duration.is_zero())
        .ok_or_else(timeout)
}

fn request_status(
    authority: &str,
    deadline: Instant,
    resolve: fn(&str) -> std::io::Result<Vec<SocketAddr>>,
) -> std::io::Result<bool> {
    remaining(deadline)?;
    let addresses = resolve(authority)?;
    // localhost can resolve to IPv6 first, so the check tries each address within one deadline.
    let mut stream = Err(std::io::Error::other(format!("no address for {authority}")));
    for address in addresses {
        stream = TcpStream::connect_timeout(&address, remaining(deadline)?);
        if stream.is_ok() {
            break;
        }
    }
    let mut stream = stream?;
    let request = format!("GET /health HTTP/1.0\r\nHost: {authority}\r\n\r\n");
    let mut pending = request.as_bytes();
    while !pending.is_empty() {
        stream.set_write_timeout(Some(remaining(deadline)?))?;
        match stream.write(pending) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "Could not write the health request.",
                ));
            }
            Ok(count) => pending = &pending[count..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    let mut bytes = [0; MAX_STATUS_LINE];
    let mut used = 0;
    while used < bytes.len() {
        stream.set_read_timeout(Some(remaining(deadline)?))?;
        let count = match stream.read(&mut bytes[used..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "The HTTP status line has no newline.",
                ));
            }
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if let Some(end) = bytes[used..used + count]
            .iter()
            .position(|byte| *byte == b'\n')
        {
            remaining(deadline)?;
            return parse_status_line(&bytes[..used + end]);
        }
        used += count;
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        "The HTTP status line exceeds 1,024 bytes or has no newline.",
    ))
}

fn parse_status_line(bytes: &[u8]) -> std::io::Result<bool> {
    let line = std::str::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let mut fields = line.split_ascii_whitespace();
    if !matches!(fields.next(), Some("HTTP/1.0" | "HTTP/1.1")) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "The HTTP status line has an invalid version.",
        ));
    }
    let code = fields.next().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "The HTTP status line has no status code.",
        )
    })?;
    if code.len() != 3 || !code.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "The HTTP status line has an invalid status code.",
        ));
    }
    Ok(code == "200")
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
    use super::{Probe, start_worker};
    use std::io::Write;
    use std::net::SocketAddr;
    use std::sync::OnceLock;
    use std::time::{Duration, Instant};

    // The listener stops waiting after two seconds if the probe never connects.
    fn response(bytes: Vec<u8>, delay: Duration) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
        listener.set_nonblocking(true).expect("nonblocking");
        let url = format!("http://{}", listener.local_addr().expect("address"));
        let thread = std::thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(2);
            while Instant::now() < until {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        use std::io::Read;
                        stream
                            .set_read_timeout(Some(Duration::from_millis(500)))
                            .expect("read timeout");
                        let _ = stream.read(&mut [0; 1024]);
                        stream
                            .set_write_timeout(Some(Duration::from_millis(200)))
                            .expect("timeout");
                        for byte in bytes {
                            if stream.write_all(&[byte]).is_err() {
                                break;
                            }
                            std::thread::sleep(delay);
                        }
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accept: {error}"),
                }
            }
        });
        (url, thread)
    }

    #[test]
    fn dns_uses_the_same_deadline_and_a_bounded_worker() {
        static LOOKUPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        // This lookup outlasts the caller's deadline so later checks must reuse the occupied worker.
        fn slow_dns(_: &str) -> std::io::Result<Vec<SocketAddr>> {
            LOOKUPS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(1600));
            Ok(vec!["127.0.0.1:1".parse().expect("address")])
        }
        let probe = Probe {
            worker: OnceLock::from(start_worker(slow_dns)),
        };
        let start = Instant::now();
        assert_eq!(probe.details("http://slow.test")["state"], "unreachable");
        assert!(start.elapsed() < Duration::from_millis(900));
        assert_eq!(probe.details("http://slow.test")["state"], "unreachable");
        let start = Instant::now();
        assert_eq!(probe.details("http://slow.test")["state"], "unreachable");
        assert!(
            start.elapsed() < Duration::from_millis(200),
            "a full queue must not wait"
        );
        assert_eq!(LOOKUPS.load(std::sync::atomic::Ordering::SeqCst), 1);
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(
            LOOKUPS.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "expired requests must not resolve"
        );
    }

    #[test]
    fn trickled_status_line_uses_one_deadline() {
        let (url, thread) = response(b"HTTP/1.1 200 OK\r\n".to_vec(), Duration::from_millis(100));
        let start = Instant::now();
        let result = Probe::default().details(&url);
        let elapsed = start.elapsed();
        thread.join().expect("server");
        assert_eq!(result["state"], "unreachable");
        assert!(
            elapsed < Duration::from_millis(900),
            "probe took {elapsed:?}"
        );
    }

    #[test]
    fn status_lines_require_a_terminator_and_fit_the_limit() {
        for (line, expected) in [
            (b"HTTP/1.1 200 OK\r\n".to_vec(), "running"),
            (b"HTTP/1.0 200 OK\r\n".to_vec(), "running"),
            (
                [b"HTTP/1.1 200 ".as_slice(), &[b'A'; 1009], b"\r\n"].concat(),
                "running",
            ),
            (
                [b"HTTP/1.1 200 ".as_slice(), &[b'A'; 1010], b"\r\n"].concat(),
                "unreachable",
            ),
            (b"HTTP/2 200 OK\r\n".to_vec(), "unreachable"),
            (b"HTTP/1.1 20x OK\r\n".to_vec(), "unreachable"),
            (b"HTTP/1.1\r\n".to_vec(), "unreachable"),
            (b"HTTP/1.1 200 \xff\r\n".to_vec(), "unreachable"),
            (
                b"HTTP/1.1 200 OK\r\nignored headers and body".to_vec(),
                "running",
            ),
            (b"HTTP/1.0 503 Unavailable\r\n".to_vec(), "unreachable"),
            (b"HTTP/1.1 200 OK".to_vec(), "unreachable"),
            (
                [b"HTTP/1.1 200 ".as_slice(), &[b'A'; 2048], b"\r\n"].concat(),
                "unreachable",
            ),
        ] {
            let (url, thread) = response(line, Duration::ZERO);
            let result = Probe::default().details(&url);
            thread.join().expect("server");
            assert_eq!(result["state"], expected);
        }
    }
}
