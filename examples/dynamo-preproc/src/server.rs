// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    pin::{Pin, pin},
    sync::Arc,
    time::Duration,
};

use anyhow::{Context, ensure};
use futures_util::{Stream, StreamExt};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    task::JoinSet,
    time::{Instant, timeout, timeout_at},
};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    preprocessor::{MAX_BODY, headers},
    proto::envoy::{
        config::core::v3::{HeaderValue, HeaderValueOption},
        service::ext_proc::v3::{
            self as ext,
            external_processor_server::{ExternalProcessor, ExternalProcessorServer},
            processing_request::Request as Input,
            processing_response::Response as Output,
        },
    },
    router::{MODEL_HEADER, Router},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);
const CHUNK_SIZE: usize = 64 * 1024;
const MAX_UPLOADS: usize = 8;

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct Error {
    pub status_code: u16,
    pub message: String,
}

impl Error {
    pub fn new(status_code: u16, message: impl Into<String>) -> Self {
        Self {
            status_code,
            message: message.into(),
        }
    }
}

#[derive(Clone)]
pub struct Server {
    router: Arc<Router>,
    requests: Arc<Semaphore>,
    streams: Arc<Semaphore>,
    direct_streams: Arc<Semaphore>,
}

struct Prepared {
    body: Vec<u8>,
    headers: ext::HeaderMutation,
    has_trailers: bool,
    inflight: Option<crate::metrics::InFlight>,
    stream_permit: OwnedSemaphorePermit,
}

impl Server {
    pub fn new(router: Arc<Router>, max_active_streams: usize) -> Self {
        Self {
            router,
            requests: Arc::new(Semaphore::new(MAX_UPLOADS)),
            streams: Arc::new(Semaphore::new(max_active_streams)),
            direct_streams: Arc::new(Semaphore::new(max_active_streams)),
        }
    }

    pub fn into_service(self) -> ExternalProcessorServer<Self> {
        ExternalProcessorServer::new(self)
    }

    async fn prepare(
        &self,
        input: &mut Streaming<ext::ProcessingRequest>,
        request_permit: OwnedSemaphorePermit,
    ) -> anyhow::Result<Prepared> {
        let first = input.message().await?.context("missing request headers")?;
        let Some(Input::RequestHeaders(first)) = first.request else {
            anyhow::bail!("request headers must arrive first");
        };
        ensure!(
            !first.end_of_stream,
            "chat completion requires a request body"
        );
        let values = first
            .headers
            .unwrap_or_default()
            .headers
            .into_iter()
            .map(|h| {
                let value = if h.raw_value.is_empty() {
                    h.value
                } else {
                    String::from_utf8(h.raw_value)?
                };
                Ok((h.key, value))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let (headers, mut remove_headers) = headers(&values)?;
        let mut body = Vec::new();
        let mut has_trailers = false;
        loop {
            let next = input.message().await?.context("incomplete request body")?;
            match next.request {
                Some(Input::RequestBody(chunk)) => {
                    ensure!(
                        !has_trailers || (chunk.body.is_empty() && chunk.end_of_stream),
                        "expected body end after request trailers"
                    );
                    if chunk.body.len() > MAX_BODY - body.len() {
                        return Err(Error::new(413, "request body exceeds 2 MiB").into());
                    }
                    body.extend_from_slice(&chunk.body);
                    if chunk.end_of_stream {
                        break;
                    }
                }
                // Stock agentgateway 1.0 sends an empty body EOS after trailers.
                Some(Input::RequestTrailers(_)) if !has_trailers => has_trailers = true,
                _ => anyhow::bail!("expected request body"),
            }
        }
        // Loopback judge calls need headroom while the parent holds its stream slot.
        let streams = if self.router.is_passthrough(&body) {
            &self.direct_streams
        } else {
            &self.streams
        };
        let stream_permit = streams
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::new(503, "processor stream limit reached"))?;
        drop(request_permit);
        let (body, model, inflight) = self.router.decide(&body, &headers).await?;
        if body.len() > MAX_BODY {
            return Err(Error::new(413, "rewritten request body exceeds 2 MiB").into());
        }
        remove_headers.push("content-length".into());
        Ok(Prepared {
            body,
            headers: ext::HeaderMutation {
                set_headers: vec![set_header(MODEL_HEADER, &model)],
                remove_headers,
            },
            has_trailers,
            inflight,
            stream_permit,
        })
    }
}

#[tonic::async_trait]
impl ExternalProcessor for Server {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ext::ProcessingResponse, Status>> + Send>>;

    async fn process(
        &self,
        request: Request<Streaming<ext::ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let request_permit = self.requests.clone().try_acquire_owned();
        let server = self.clone();
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        let mut input = request.into_inner();
        let output = async_stream::stream! {
            let Ok(request_permit) = request_permit else {
                yield Ok(reject(Error::new(503, "request upload limit reached")));
                return;
            };
            let result = timeout_at(deadline, server.prepare(&mut input, request_permit)).await;
            let prepared = match result {
                Ok(Ok(prepared)) => prepared,
                Ok(Err(error)) => {
                    tracing::warn!(%error, "preprocessing rejected request");
                    yield Ok(reject(error.downcast::<Error>()
                        .unwrap_or_else(|e| Error::new(400, e.to_string()))));
                    return;
                }
                Err(_) => {
                    yield Ok(reject(Error::new(504, "preprocessing deadline exceeded")));
                    return;
                }
            };
            // Both guards live through response completion, cancellation and backpressure.
            let _stream_permit = prepared.stream_permit;
            let _inflight = prepared.inflight;
            let mut response_deadline = Instant::now() + RESPONSE_TIMEOUT;
            // v1.0 selects the route when this header reply arrives, before reading the body.
            yield Ok(reply(Output::RequestHeaders(ext::HeadersResponse {
                response: Some(ext::CommonResponse {
                    header_mutation: Some(prepared.headers),
                    ..Default::default()
                }),
            })));
            if prepared.has_trailers {
                yield Ok(reply(Output::RequestTrailers(ext::TrailersResponse::default())));
            }
            for (i, chunk) in prepared.body.chunks(CHUNK_SIZE).enumerate() {
                if Instant::now() >= response_deadline {
                    yield Err(Status::deadline_exceeded("request output timed out"));
                    return;
                }
                yield Ok(reply(Output::RequestBody(body_reply(
                    chunk.to_vec(), (i + 1) * CHUNK_SIZE >= prepared.body.len(),
                ))));
            }
            response_deadline = Instant::now() + RESPONSE_TIMEOUT;
            let mut has_headers = false;
            let mut has_trailers = false;
            loop {
                let next = match timeout_at(response_deadline, input.message()).await {
                    Ok(Ok(Some(next))) => next,
                    Ok(Ok(None)) => {
                        yield Err(Status::failed_precondition("incomplete response stream"));
                        return;
                    }
                    Ok(Err(error)) => { yield Err(error); return; }
                    Err(_) => {
                        yield Err(Status::deadline_exceeded("response processing timed out"));
                        return;
                    }
                };
                response_deadline = Instant::now() + RESPONSE_TIMEOUT;
                let (output, is_done) = match next.request {
                    Some(Input::ResponseHeaders(h)) if !has_headers => {
                        has_headers = true;
                        (Output::ResponseHeaders(ext::HeadersResponse {
                            response: Some(ext::CommonResponse::default()),
                        }), h.end_of_stream)
                    }
                    Some(Input::ResponseBody(b)) if has_headers
                        && (!has_trailers || (b.body.is_empty() && b.end_of_stream)) => {
                        let is_done = b.end_of_stream;
                        (Output::ResponseBody(body_reply(b.body, is_done)), is_done)
                    }
                    Some(Input::ResponseTrailers(_)) if has_headers && !has_trailers => {
                        has_trailers = true;
                        // ACK trailers, then consume agentgateway's synthetic body EOS.
                        (Output::ResponseTrailers(ext::TrailersResponse::default()), false)
                    }
                    _ => {
                        yield Err(Status::invalid_argument("unexpected response processing phase"));
                        return;
                    }
                };
                yield Ok(reply(output));
                if is_done { return; }
            }
        };
        Ok(Response::new(Box::pin(with_output_timeout(output))))
    }
}

fn with_output_timeout(
    output: impl Stream<Item = Result<ext::ProcessingResponse, Status>> + Send + 'static,
) -> impl Stream<Item = Result<ext::ProcessingResponse, Status>> + Send {
    let (tx, mut rx) = mpsc::channel(1);
    let mut tasks = JoinSet::new();
    // Poll independently of tonic so a stalled reader cannot retain admission permits.
    tasks.spawn(async move {
        let mut output = pin!(output);
        while let Some(response) = output.next().await {
            if timeout(RESPONSE_TIMEOUT, tx.send(response))
                .await
                .map_err(|_| Status::deadline_exceeded("processor output stalled"))?
                .is_err()
            {
                break;
            }
        }
        Ok::<_, Status>(())
    });
    async_stream::stream! {
        // Owning the JoinSet also cancels the producer when this stream is dropped.
        while let Some(response) = rx.recv().await {
            yield response;
        }
        match tasks.join_next().await {
            Some(Ok(Err(error))) => yield Err(error),
            Some(Err(_)) => yield Err(Status::internal("processor task failed")),
            _ => {}
        }
    }
}

fn reply(response: Output) -> ext::ProcessingResponse {
    ext::ProcessingResponse {
        response: Some(response),
        ..Default::default()
    }
}

fn body_reply(body: Vec<u8>, is_end_of_stream: bool) -> ext::BodyResponse {
    ext::BodyResponse {
        response: Some(ext::CommonResponse {
            body_mutation: Some(ext::BodyMutation {
                mutation: Some(ext::body_mutation::Mutation::StreamedResponse(
                    ext::StreamedBodyResponse {
                        body,
                        end_of_stream: is_end_of_stream,
                    },
                )),
            }),
            ..Default::default()
        }),
    }
}

fn set_header(key: &str, value: &str) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.into(),
            value: String::new(),
            raw_value: value.as_bytes().to_vec(),
        }),
        append_action: 2,
        keep_empty_value: false,
        ..Default::default()
    }
}

fn reject(error: Error) -> ext::ProcessingResponse {
    reply(Output::ImmediateResponse(ext::ImmediateResponse {
        status: Some(crate::proto::envoy::r#type::v3::HttpStatus {
            code: i32::from(error.status_code),
        }),
        headers: Some(ext::HeaderMutation {
            set_headers: vec![set_header("content-type", "application/json")],
            remove_headers: vec![],
        }),
        body: serde_json::to_vec(&serde_json::json!({"error": {
            "message": error.message, "type": "switchyard_preproc_error"
        }}))
        .expect("JSON error object serializes"),
        details: "switchyard_preproc".into(),
        ..Default::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};

    type Client =
        ext::external_processor_client::ExternalProcessorClient<tonic::transport::Channel>;

    async fn start(
        max_active_streams: usize,
        config: &str,
    ) -> (Client, Arc<Semaphore>, Arc<Router>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let runner = switchyard_runner::Runner::from_toml(config);
        let mut router = Router::new(runner.unwrap());
        router.metrics = Some(crate::metrics::Metrics::empty_for_test());
        let router = Arc::new(router);
        let server = Server::new(router.clone(), max_active_streams);
        let requests = server.requests.clone();
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(server.into_service())
                .serve_with_incoming(TcpListenerStream::new(listener))
                .await
                .unwrap();
        });
        (
            Client::new(
                tonic::transport::Endpoint::from_shared(format!("http://{address}"))
                    .unwrap()
                    .connect()
                    .await
                    .unwrap(),
            ),
            requests,
            router,
        )
    }

    fn input(request: Input) -> ext::ProcessingRequest {
        ext::ProcessingRequest {
            request: Some(request),
            ..Default::default()
        }
    }

    fn body(bytes: &[u8], is_end_of_stream: bool) -> ext::HttpBody {
        ext::HttpBody {
            body: bytes.to_vec(),
            end_of_stream: is_end_of_stream,
        }
    }

    fn headers() -> Input {
        let headers = [
            (":method", "POST"),
            (":path", "/v1/chat/completions"),
            ("content-type", "application/json"),
            (MODEL_HEADER, "forged"),
        ]
        .into_iter()
        .map(|(k, v)| set_header(k, v).header.unwrap())
        .collect();
        Input::RequestHeaders(ext::HttpHeaders {
            headers: Some(crate::proto::envoy::config::core::v3::HeaderMap { headers }),
            end_of_stream: false,
            ..Default::default()
        })
    }

    async fn send(tx: &mpsc::Sender<ext::ProcessingRequest>, request: Input) {
        tx.send(input(request)).await.unwrap();
    }

    async fn next(rx: &mut Streaming<ext::ProcessingResponse>) -> Output {
        rx.message().await.unwrap().unwrap().response.unwrap()
    }

    async fn process_body(client: &mut Client, json: &[u8]) -> Streaming<ext::ProcessingResponse> {
        client
            .process(tokio_stream::iter([
                input(headers()),
                input(Input::RequestBody(body(json, true))),
            ]))
            .await
            .unwrap()
            .into_inner()
    }

    fn streamed(reply: ext::BodyResponse) -> ext::StreamedBodyResponse {
        let Some(ext::body_mutation::Mutation::StreamedResponse(body)) =
            reply.response.unwrap().body_mutation.unwrap().mutation
        else {
            panic!()
        };
        body
    }

    #[tokio::test]
    async fn routes_upload_and_passes_response_through_trailers_to_eos() {
        let (mut client, requests, router) = start(16, include_str!("../config/routes.toml")).await;
        let (tx, receive) = mpsc::channel(8);
        send(&tx, headers()).await;
        let receive = ReceiverStream::new(receive);
        let mut rx = client.process(receive).await.unwrap().into_inner();
        let json = br#"{"model":"auto","messages":[{"role":"user","content":"hello"}]}"#;
        for chunk in json.chunks(40) {
            send(&tx, Input::RequestBody(body(chunk, false))).await;
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(30), rx.message())
                .await
                .is_err()
        );
        send(&tx, Input::RequestTrailers(ext::HttpTrailers::default())).await;
        send(&tx, Input::RequestBody(body(b"", true))).await;
        let Output::RequestHeaders(h) = next(&mut rx).await else {
            panic!()
        };
        let h = h.response.unwrap().header_mutation.unwrap();
        let selected = h.set_headers[0].header.as_ref().unwrap();
        assert_eq!(selected.raw_value, b"Qwen/Qwen3-0.6B");
        assert!(h.remove_headers.contains(&"content-length".to_owned()));
        assert!(matches!(next(&mut rx).await, Output::RequestTrailers(_)));
        let Output::RequestBody(b) = next(&mut rx).await else {
            panic!()
        };
        let b = streamed(b);
        assert!(b.end_of_stream);
        let value: serde_json::Value = serde_json::from_slice(&b.body).unwrap();
        assert_eq!(value["model"], "Qwen/Qwen3-0.6B");
        assert_eq!(requests.available_permits(), MAX_UPLOADS);
        assert_eq!(router.metrics.as_ref().unwrap().inflight_for_test(), 1);
        send(&tx, Input::ResponseHeaders(ext::HttpHeaders::default())).await;
        assert!(matches!(next(&mut rx).await, Output::ResponseHeaders(_)));
        for (bytes, is_done) in [
            (b"data: hello\n\n".as_slice(), false),
            (b"".as_slice(), true),
        ] {
            send(&tx, Input::ResponseBody(body(bytes, is_done))).await;
            let Output::ResponseBody(b) = next(&mut rx).await else {
                panic!()
            };
            let b = streamed(b);
            assert_eq!((b.body.as_slice(), b.end_of_stream), (bytes, is_done));
            if !is_done {
                send(&tx, Input::ResponseTrailers(ext::HttpTrailers::default())).await;
                assert!(matches!(next(&mut rx).await, Output::ResponseTrailers(_)));
            }
        }
        assert!(rx.message().await.unwrap().is_none());
        assert_eq!(router.metrics.as_ref().unwrap().inflight_for_test(), 0);

        let (tx, receive) = mpsc::channel(8);
        send(&tx, headers()).await;
        send(&tx, Input::RequestBody(body(json, true))).await;
        let mut rx = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        assert!(matches!(next(&mut rx).await, Output::RequestHeaders(_)));
        assert!(matches!(next(&mut rx).await, Output::RequestBody(_)));
        assert_eq!(router.metrics.as_ref().unwrap().inflight_for_test(), 1);
        drop(rx);
        drop(tx);
        tokio::time::timeout(Duration::from_secs(1), async {
            while router.metrics.as_ref().unwrap().inflight_for_test() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn response_timeout_tracks_inactivity() {
        async fn elapse(duration: Duration) {
            // Resume before socket I/O so idle network polls cannot auto-advance time.
            tokio::time::pause();
            tokio::time::advance(duration).await;
            tokio::time::resume();
        }

        let (mut client, _, router) = start(1, include_str!("../config/routes.toml")).await;
        let (tx, receive) = mpsc::channel(8);
        send(&tx, headers()).await;
        let mut rx = client
            .process(ReceiverStream::new(receive))
            .await
            .unwrap()
            .into_inner();
        send(
            &tx,
            Input::RequestBody(body(
                br#"{"model":"auto","messages":[{"role":"user","content":"hello"}]}"#,
                true,
            )),
        )
        .await;
        assert!(matches!(next(&mut rx).await, Output::RequestHeaders(_)));
        assert!(matches!(next(&mut rx).await, Output::RequestBody(_)));
        send(&tx, Input::ResponseHeaders(ext::HttpHeaders::default())).await;
        assert!(matches!(next(&mut rx).await, Output::ResponseHeaders(_)));

        let mut rejected = process_body(
            &mut client,
            br#"{"model":"auto","messages":[{"role":"user","content":"hello"}]}"#,
        )
        .await;
        let Output::ImmediateResponse(error) = next(&mut rejected).await else {
            panic!()
        };
        assert_eq!(error.status.unwrap().code, 503);

        for _ in 0..3 {
            elapse(RESPONSE_TIMEOUT / 2).await;
            send(&tx, Input::ResponseBody(body(b"data: hello\n\n", false))).await;
            let Output::ResponseBody(b) = next(&mut rx).await else {
                panic!()
            };
            assert_eq!(streamed(b).body, b"data: hello\n\n");
        }
        assert_eq!(router.metrics.as_ref().unwrap().inflight_for_test(), 1);
        elapse(RESPONSE_TIMEOUT + Duration::from_secs(1)).await;
        assert_eq!(
            rx.message().await.unwrap_err().code(),
            tonic::Code::DeadlineExceeded
        );
        assert_eq!(router.metrics.as_ref().unwrap().inflight_for_test(), 0);
    }

    #[tokio::test]
    async fn rejects_cumulative_body_limit_before_routing_headers() {
        let (mut client, _, _) = start(16, include_str!("../config/routes.toml")).await;
        let messages = tokio_stream::iter([
            input(headers()),
            input(Input::RequestBody(body(&vec![b' '; MAX_BODY], false))),
            input(Input::RequestBody(body(b" ", true))),
        ]);
        let mut rx = client.process(messages).await.unwrap().into_inner();
        let Output::ImmediateResponse(error) = next(&mut rx).await else {
            panic!()
        };
        assert_eq!(error.status.unwrap().code, 413);
        assert!(rx.message().await.unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn loopback_judge_runs_when_task_slots_are_full() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let config = include_str!("../config/task-classifier.toml")
            .replace(
                "http://inference-gateway.agentgateway-system.svc.cluster.local/v1",
                &format!("http://{address}/v1"),
            )
            .replace("fail_open = true", "fail_open = false");
        let (client, _, _) = start(8, &config).await;
        let nested_client = client.clone();
        let barrier = Arc::new(tokio::sync::Barrier::new(8));
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(move || {
            let mut client = nested_client.clone();
            let barrier = barrier.clone();
            async move {
                // Hold all parent slots before their judge calls re-enter PreProc.
                barrier.wait().await;
                let json = br#"{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"judge"}]}"#;
                let mut rx = process_body(&mut client, json).await;
                assert!(matches!(next(&mut rx).await, Output::RequestHeaders(_)));
                assert!(matches!(next(&mut rx).await, Output::RequestBody(_)));
                axum::Json(serde_json::json!({
                    "id":"judge", "object":"chat.completion", "created":0, "model":"Qwen/Qwen3-1.7B",
                    "choices":[{"index":0,"message":{"role":"assistant","content":
                        r#"{"crux":"bounded task","primary_rule":"SUP-1","capability_boundary":"supported","p_solve":0.9}"#
                    },"finish_reason":"stop"}]
                }))
            }
        }));
        let http_task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut tasks = JoinSet::new();
        for _ in 0..8 {
            let mut client = client.clone();
            tasks.spawn(async move {
                let json =
                    br#"{"model":"auto","messages":[{"role":"user","content":"Sort [3,1,2]."}]}"#;
                let mut rx = process_body(&mut client, json).await;
                assert!(matches!(next(&mut rx).await, Output::RequestHeaders(_)));
                let Output::RequestBody(response) = next(&mut rx).await else {
                    panic!("expected routed body")
                };
                let body: serde_json::Value =
                    serde_json::from_slice(&streamed(response).body).unwrap();
                assert_eq!(body["model"], "Qwen/Qwen3-0.6B");
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        http_task.abort();
    }
}
