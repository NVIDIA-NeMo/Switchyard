// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Lightweight HTTP test servers shared by backend and config tests.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::Value;
use switchyard_core::{Result, SwitchyardError};

pub mod config;
pub mod intake;

/// One HTTP request captured by a local mock server.
#[derive(Debug)]
pub struct CapturedRequest {
    /// Request method.
    pub method: String,
    /// Request path including query.
    pub path: String,
    /// Lowercased request headers.
    pub headers: BTreeMap<String, String>,
    /// JSON request body.
    pub body: Value,
    /// Exact request body bytes received on the socket.
    pub raw_body: Vec<u8>,
}

impl CapturedRequest {
    /// Returns a header value using case-insensitive matching.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

/// Mock server that accepts exactly one request.
pub struct OneShotServer {
    /// Base URL clients can call.
    base_url: String,
    /// Captured request result from the server thread.
    receiver: Receiver<Result<CapturedRequest>>,
    /// Server thread handle joined when captured output is read.
    handle: Option<JoinHandle<()>>,
}

/// Mock server that accepts a fixed sequence of requests.
pub struct SequenceServer {
    /// Base URL clients can call.
    base_url: String,
    /// Captured request sequence from the server thread.
    receiver: Receiver<Result<Vec<CapturedRequest>>>,
    /// Server thread handle joined when captured output is read.
    handle: Option<JoinHandle<()>>,
}

impl OneShotServer {
    /// Creates a one-shot server returning JSON.
    pub fn json(status: u16, body: Value) -> Result<Self> {
        Self::raw(status, "application/json", body.to_string())
    }

    /// Creates a one-shot server returning raw SSE.
    #[allow(dead_code)]
    pub fn sse(body: impl Into<String>) -> Result<Self> {
        Self::raw(200, "text/event-stream", body.into())
    }

    /// Returns the base URL for this server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Waits for and returns the captured request.
    pub fn captured(mut self) -> Result<CapturedRequest> {
        let request = self.receiver.recv_timeout(Duration::from_secs(5));
        match &request {
            Ok(_) | Err(RecvTimeoutError::Disconnected) => {
                if let Some(handle) = self.handle.take() {
                    if handle.join().is_err() {
                        return Err(SwitchyardError::Other("server thread panicked".to_string()));
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        request.map_err(|error| {
            SwitchyardError::Other(format!("server should capture one request: {error}"))
        })?
    }

    /// Creates a one-shot server returning an arbitrary content type and body.
    fn raw(status: u16, content_type: &'static str, body: String) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| SwitchyardError::Other(format!("bind test server: {error}")))?;
        let address = listener.local_addr().map_err(|error| {
            SwitchyardError::Other(format!("read test server address: {error}"))
        })?;
        let base_url = format!("http://{address}");
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = handle_one_request(listener, status, content_type, body);
            let _ignored = sender.send(result);
        });

        Ok(Self {
            base_url,
            receiver,
            handle: Some(handle),
        })
    }
}

impl SequenceServer {
    /// Creates a sequence server returning one JSON response per request.
    pub fn json(responses: Vec<(u16, Value)>) -> Result<Self> {
        Self::raw(
            responses
                .into_iter()
                .map(|(status, body)| (status, "application/json", body.to_string()))
                .collect(),
        )
    }

    /// Creates a sequence server returning one SSE response per request.
    pub fn sse(responses: Vec<String>) -> Result<Self> {
        Self::raw(
            responses
                .into_iter()
                .map(|body| (200, "text/event-stream", body))
                .collect(),
        )
    }

    fn raw(responses: Vec<(u16, &'static str, String)>) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .map_err(|error| SwitchyardError::Other(format!("bind test server: {error}")))?;
        let address = listener.local_addr().map_err(|error| {
            SwitchyardError::Other(format!("read test server address: {error}"))
        })?;
        let base_url = format!("http://{address}");
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            let result = handle_request_sequence(listener, responses);
            let _ignored = sender.send(result);
        });

        Ok(Self {
            base_url,
            receiver,
            handle: Some(handle),
        })
    }

    /// Returns the base URL for this server.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Waits for and returns every captured request.
    pub fn captured(mut self) -> Result<Vec<CapturedRequest>> {
        let requests = self.receiver.recv_timeout(Duration::from_secs(5));
        match &requests {
            Ok(_) | Err(RecvTimeoutError::Disconnected) => {
                if let Some(handle) = self.handle.take() {
                    if handle.join().is_err() {
                        return Err(SwitchyardError::Other("server thread panicked".to_string()));
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
        }
        requests.map_err(|error| {
            SwitchyardError::Other(format!("server should capture requests: {error}"))
        })?
    }
}

/// Handles one HTTP request and writes a fixed response.
fn handle_one_request(
    listener: TcpListener,
    status: u16,
    content_type: &'static str,
    body: String,
) -> Result<CapturedRequest> {
    let (mut stream, _) = listener
        .accept()
        .map_err(|error| SwitchyardError::Other(format!("accept one request: {error}")))?;
    let request = read_request(&mut stream)?;

    let response = format!(
        "HTTP/1.1 {status} OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .map_err(|error| SwitchyardError::Other(format!("write test response: {error}")))?;
    Ok(request)
}

/// Handles a fixed sequence of HTTP requests with matching responses.
fn handle_request_sequence(
    listener: TcpListener,
    responses: Vec<(u16, &'static str, String)>,
) -> Result<Vec<CapturedRequest>> {
    let mut requests = Vec::with_capacity(responses.len());
    for (status, content_type, body) in responses {
        requests.push(handle_one_request(
            listener
                .try_clone()
                .map_err(|error| SwitchyardError::Other(format!("clone test listener: {error}")))?,
            status,
            content_type,
            body,
        )?);
    }
    Ok(requests)
}

/// Reads one HTTP/1.1 request from a blocking test socket.
fn read_request(stream: &mut std::net::TcpStream) -> Result<CapturedRequest> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| SwitchyardError::Other(format!("read request: {error}")))?;
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&chunk[..read]);
        if let Some((header_end, content_length)) = request_shape(&bytes) {
            let body_start = header_end + 4;
            if bytes.len() >= body_start + content_length {
                break;
            }
        }
    }

    let (header_end, content_length) = request_shape(&bytes)
        .ok_or_else(|| SwitchyardError::Other("request should contain HTTP headers".to_string()))?;
    let header_text = std::str::from_utf8(&bytes[..header_end]).map_err(|error| {
        SwitchyardError::Other(format!("headers should be valid UTF-8: {error}"))
    })?;
    let body_start = header_end + 4;
    let body_end = body_start + content_length;
    let raw_body = bytes[body_start..body_end].to_vec();
    let raw_body_text = std::str::from_utf8(&raw_body)
        .map_err(|error| SwitchyardError::Other(format!("body should be valid UTF-8: {error}")))?
        .to_string();

    let mut lines = header_text.lines();
    let start_line = lines
        .next()
        .ok_or_else(|| SwitchyardError::Other("request line".to_string()))?;
    let mut start_parts = start_line.split_whitespace();
    let method = start_parts.next().unwrap_or_default().to_string();
    let path = start_parts.next().unwrap_or_default().to_string();

    let mut headers = BTreeMap::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        headers.insert(name.to_ascii_lowercase(), value.trim().to_string());
    }

    let body = if raw_body_text.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&raw_body_text).map_err(|error| {
            SwitchyardError::Other(format!("request body should be JSON: {error}"))
        })?
    };

    Ok(CapturedRequest {
        method,
        path,
        headers,
        body,
        raw_body,
    })
}

/// Returns the header/body split and content length once headers are complete.
fn request_shape(bytes: &[u8]) -> Option<(usize, usize)> {
    let header_end = find_bytes(bytes, b"\r\n\r\n")?;
    let headers = std::str::from_utf8(&bytes[..header_end]).ok()?;
    let content_length = headers
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| {
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    Some((header_end, content_length))
}

/// Finds a byte sequence without pulling in memchr for tests.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
