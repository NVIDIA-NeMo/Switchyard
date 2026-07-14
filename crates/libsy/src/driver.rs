// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # driver — a type-erased promise-over-a-stream request pump
//!
//! [`TypeErasedDriver`] is the generic offload primitive; the crate root builds the
//! libsy-typed `Driver` on top of it. This module has no dependency on the rest of the
//! crate — the coupling is one-directional (`lib.rs` → `driver.rs`).
//!
//! A [`TypeErasedDriver`] lets a *producer* (e.g. a routing algorithm) fulfill
//! arbitrary requests by publishing promises onto a stream that a single *consumer*
//! drains. It is the type-erased generalization of the crate's `run_stream` offload:
//! instead of one fixed request/response shape, a producer calls
//! [`fulfill_request`](TypeErasedDriver::fulfill_request)
//! with *any* `REQ` and awaits *any* `RES`.
//!
//! - [`fulfill_request`](TypeErasedDriver::fulfill_request) enqueues a [`DriverStep::Request`]
//!   carrying a [`DriverRequest`], then awaits the consumer's response.
//! - [`info`](TypeErasedDriver::info) pushes a fire-and-forget [`DriverStep::Info`] — no promise
//!   to await.
//! - [`done`](TypeErasedDriver::done) emits the terminal [`DriverStep::Done`] with a final payload.
//! - [`stream`](TypeErasedDriver::stream) hands the single consumer the [`Stream`] of steps; for
//!   each [`DriverStep::Request`] the consumer downcasts the request, computes a
//!   response, and writes it back with [`DriverRequest::respond`].
//!
//! Payloads are erased to `Box<dyn Any + Send>`, so one `TypeErasedDriver` serves any request
//! type; the consumer downcasts to the concrete type it expects. `TypeErasedDriver` is `Clone`
//! (many producer tasks may call it concurrently — multi-producer), while the stream
//! is single-consumer. Each request rides its own `oneshot`, so concurrent
//! `fulfill_request` calls never cross responses.
//!
//! ## Pacing (bounded step channel)
//!
//! The step channel has capacity 1, so a producer cannot publish its next step until
//! the consumer has pulled the previous one. The consumer therefore *paces* the
//! algorithm: it advances one step for each `.next().await`. Every producer method is
//! `async` because publishing a step awaits channel capacity.
//!
//! ## Termination
//!
//! There is no explicit stop method — the consumer terminates by **dropping the
//! stream** (and any [`DriverRequest`] it is holding). The producer's next publish
//! (`fulfill_request`/`info`/`done`/`fail`) then resolves to `Err`, and a producer
//! awaiting a response sees `Err` once the promise it handed out is dropped. Either
//! way the algorithm unwinds cooperatively at its next driver interaction. Because the
//! producer runs on a task the driver does not own, hard cancellation (e.g. mid-compute
//! that never touches the driver) is the caller's concern — abort the producer task.

use std::{
    any::Any,
    error::Error,
    sync::{Arc, Mutex},
};

use futures::{Stream, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

type BoxErr = Box<dyn Error + Send + Sync>;
type BoxAny = Box<dyn Any + Send>;
type StepResult = Result<DriverStep, BoxErr>;

/// One item on the stream returned by [`TypeErasedDriver::stream`].
pub enum DriverStep {
    /// A request awaiting a response. The consumer downcasts it and fulfills the
    /// paired promise with [`DriverRequest::respond`].
    Request(DriverRequest),
    /// A fire-and-forget payload from a producer; no response is expected.
    Info(BoxAny),
    /// A producer's terminal result. The consumer treats it as the last meaningful
    /// step (the stream itself closes when every [`TypeErasedDriver`] clone drops).
    Done(BoxAny),
}

/// The consumer-facing half of one [`TypeErasedDriver::fulfill_request`] call.
///
/// Yielded inside [`DriverStep::Request`]. The consumer reads the request via
/// [`request`](Self::request), does whatever work it names, and fulfills the promise
/// with [`respond`](Self::respond) — unblocking the producer's `fulfill_request`.
pub struct DriverRequest {
    request: BoxAny,
    // Fulfilled exactly once — `respond` consumes `self` to send, so no `Option`.
    tx: oneshot::Sender<Result<BoxAny, BoxErr>>,
}

impl DriverRequest {
    /// Borrow the request payload as `REQ`. Errors if the producer enqueued a
    /// different type than the consumer expected.
    pub fn request<REQ: Any>(&self) -> Result<&REQ, BoxErr> {
        self.request
            .downcast_ref::<REQ>()
            .ok_or_else(|| "driver: request type mismatch".into())
    }

    /// Fulfill the promise with a typed response, or an `Err` to propagate a failure
    /// back to the producer. Consumes `self`: a promise is fulfilled exactly once.
    pub fn respond<RES: Any + Send>(self, res: Result<RES, BoxErr>) -> Result<(), BoxErr> {
        // Erase the response so the single stream item type can carry any RES; the
        // producer downcasts it back in `fulfill_request`.
        let boxed: Result<BoxAny, BoxErr> = res.map(|r| Box::new(r) as BoxAny);
        self.tx
            .send(boxed)
            .map_err(|_| "driver: response receiver dropped".into())
    }
}

/// Internal shared state: the step channel plus the single, take-once receiver.
struct DriverInner {
    // Multi-producer: cloned into every `TypeErasedDriver`, so many tasks can enqueue steps.
    // Capacity 1: a producer blocks publishing its next step until the consumer pulls
    // the previous one, so the consumer paces the algorithm.
    step_tx: mpsc::Sender<StepResult>,
    // Single-consumer: taken out (once) by `stream`. `None` after the first take.
    step_rx: Mutex<Option<mpsc::Receiver<StepResult>>>,
}

/// A promise-over-a-stream request pump. See the [module docs](self) for the model.
///
/// Cheap to clone (shares one `Arc`); clone it to hand a producer handle to another
/// task. The consumer calls [`stream`](Self::stream) exactly once to drain steps.
#[derive(Clone)]
pub struct TypeErasedDriver {
    inner: Arc<DriverInner>,
}

impl TypeErasedDriver {
    /// Build an empty driver with its step channel ready. Take the consumer stream
    /// with [`stream`](Self::stream); enqueue work with the other methods.
    pub fn new() -> Self {
        let (step_tx, step_rx) = mpsc::channel(1);
        TypeErasedDriver {
            inner: Arc::new(DriverInner {
                step_tx,
                step_rx: Mutex::new(Some(step_rx)),
            }),
        }
    }

    /// Enqueue `req` as a [`DriverStep::Request`], await the consumer's response, and
    /// downcast it to `RES`. Errors if the stream is closed, the promise is dropped
    /// unfulfilled, the consumer responded with `Err`, or the response was not a `RES`.
    pub async fn fulfill_request<REQ, RES>(&self, req: REQ) -> Result<RES, BoxErr>
    where
        REQ: Any + Send + 'static,
        RES: Any + Send + 'static,
    {
        let (tx, rx) = oneshot::channel::<Result<BoxAny, BoxErr>>();
        let promise = DriverRequest {
            request: Box::new(req),
            tx,
        };
        self.inner
            .step_tx
            .send(Ok(DriverStep::Request(promise)))
            .await
            .map_err(|_| "driver: stream closed")?;

        // Outer error: the promise was dropped without a response. Inner error: the
        // consumer fulfilled it with an explicit `Err` — propagate it as-is.
        let response = match rx.await {
            Ok(result) => result?,
            Err(_) => return Err("driver: promise dropped without a response".into()),
        };
        response
            .downcast::<RES>()
            .map(|boxed| *boxed)
            .map_err(|_| "driver: response type mismatch".into())
    }

    /// Push a fire-and-forget [`DriverStep::Info`] payload; there is no promise to
    /// await for a response. Awaits channel capacity (the consumer pacing the stream)
    /// and errors only if the stream is closed.
    pub async fn info<INFO>(&self, info: INFO) -> Result<(), BoxErr>
    where
        INFO: Any + Send + 'static,
    {
        self.inner
            .step_tx
            .send(Ok(DriverStep::Info(Box::new(info))))
            .await
            .map_err(|_| "driver: stream closed".into())
    }

    /// Emit the terminal [`DriverStep::Done`] with a final payload. Does not close the
    /// stream (that happens when every `TypeErasedDriver` clone drops); the consumer treats it
    /// as the last meaningful step. Awaits channel capacity and errors only if the
    /// stream is closed.
    pub async fn done<T>(&self, payload: T) -> Result<(), BoxErr>
    where
        T: Any + Send + 'static,
    {
        self.inner
            .step_tx
            .send(Ok(DriverStep::Done(Box::new(payload))))
            .await
            .map_err(|_| "driver: stream closed".into())
    }

    /// Terminate the stream with an error item — the producer-side way to surface a
    /// failure to the consumer (mirrors how the crate's `run_stream` yields an `Err`
    /// step). Awaits channel capacity and errors only if the stream is
    /// already closed.
    pub async fn fail(&self, err: BoxErr) -> Result<(), BoxErr> {
        self.inner
            .step_tx
            .send(Err(err))
            .await
            .map_err(|_| "driver: stream closed".into())
    }

    /// Take the single consumer stream of [`DriverStep`]s. Callable once: a second
    /// call yields a one-item stream carrying an `Err`, since the receiver is gone.
    pub fn stream(&self) -> impl Stream<Item = Result<DriverStep, BoxErr>> {
        // Take the receiver out; `None` means already taken (or the lock was poisoned).
        let taken = match self.inner.step_rx.lock() {
            Ok(mut guard) => guard.take(),
            Err(_) => None,
        };
        match taken {
            Some(rx) => ReceiverStream::new(rx).left_stream(),
            None => futures::stream::once(async {
                Err::<DriverStep, BoxErr>("driver: stream already taken".into())
            })
            .right_stream(),
        }
    }
}

impl Default for TypeErasedDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn fulfill_request_round_trips_typed_values() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        // Producer asks for a u32 -> String on its own task.
        let producer = driver.clone();
        let handle =
            tokio::spawn(async move { producer.fulfill_request::<u32, String>(7u32).await });

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Request(promise) => {
                let req = *promise.request::<u32>()?;
                assert_eq!(req, 7);
                promise.respond::<String>(Ok(format!("got {req}")))?;
            }
            _ => return Err("expected a Request step".into()),
        }

        assert_eq!(handle.await??, "got 7");
        Ok(())
    }

    #[tokio::test]
    async fn info_pushes_a_typed_payload() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();
        driver.info(42u64).await?;

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Info(payload) => {
                let value = payload.downcast::<u64>().map_err(|_| "wrong info type")?;
                assert_eq!(*value, 42);
            }
            _ => return Err("expected an Info step".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn done_emits_the_terminal_payload() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();
        driver.done("finished".to_string()).await?;

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Done(payload) => {
                let value = payload
                    .downcast::<String>()
                    .map_err(|_| "wrong done type")?;
                assert_eq!(*value, "finished");
            }
            _ => return Err("expected a Done step".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn respond_error_propagates_to_the_producer() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        let producer = driver.clone();
        let handle = tokio::spawn(async move { producer.fulfill_request::<u32, u32>(1u32).await });

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Request(promise) => {
                promise.respond::<u32>(Err("upstream failed".into()))?;
            }
            _ => return Err("expected a Request step".into()),
        }

        match handle.await? {
            Ok(_) => Err("expected the error to propagate".into()),
            Err(err) => {
                assert!(err.to_string().contains("upstream failed"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn response_type_mismatch_errors() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        // Producer expects a String back.
        let producer = driver.clone();
        let handle =
            tokio::spawn(async move { producer.fulfill_request::<u32, String>(1u32).await });

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Request(promise) => {
                // But the consumer responds with a u32.
                promise.respond::<u32>(Ok(99u32))?;
            }
            _ => return Err("expected a Request step".into()),
        }

        match handle.await? {
            Ok(_) => Err("expected a response type mismatch".into()),
            Err(err) => {
                assert!(err.to_string().contains("response type mismatch"));
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn request_downcast_to_wrong_type_errors() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        let producer = driver.clone();
        let handle = tokio::spawn(async move { producer.fulfill_request::<u32, u32>(5u32).await });

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            DriverStep::Request(promise) => {
                assert!(promise.request::<String>().is_err());
                // Unblock the producer so its task can finish.
                promise.respond::<u32>(Ok(5u32))?;
            }
            _ => return Err("expected a Request step".into()),
        }

        assert_eq!(handle.await??, 5);
        Ok(())
    }

    #[tokio::test]
    async fn closed_stream_errors_on_send() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        // Drop the consumer stream (and its receiver) before producing anything.
        drop(driver.stream());

        assert!(driver.fulfill_request::<u32, u32>(1u32).await.is_err());
        assert!(driver.info(1u32).await.is_err());
        assert!(driver.done(1u32).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn promise_dropped_without_response_errors() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        let producer = driver.clone();
        let handle = tokio::spawn(async move { producer.fulfill_request::<u32, u32>(1u32).await });

        tokio::pin!(stream);
        match stream.next().await.ok_or("no step")?? {
            // Drop the promise without responding.
            DriverStep::Request(_promise) => {}
            _ => return Err("expected a Request step".into()),
        }

        assert!(handle.await?.is_err());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_producers_do_not_cross_responses() -> Result<(), BoxErr> {
        const N: usize = 8;
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();

        // N producers each fulfill their own request concurrently.
        let mut handles = Vec::new();
        for i in 0..N {
            let producer = driver.clone();
            handles.push((
                i,
                tokio::spawn(async move { producer.fulfill_request::<usize, usize>(i).await }),
            ));
        }

        // Single consumer responds `req * 10` to each request.
        tokio::pin!(stream);
        let mut served = 0;
        while served < N {
            match stream.next().await.ok_or("stream ended early")?? {
                DriverStep::Request(promise) => {
                    let req = *promise.request::<usize>()?;
                    promise.respond::<usize>(Ok(req * 10))?;
                    served += 1;
                }
                _ => return Err("expected a Request step".into()),
            }
        }

        // Each producer must see exactly its own response, not another's.
        for (i, handle) in handles {
            assert_eq!(handle.await??, i * 10);
        }
        Ok(())
    }

    #[tokio::test]
    async fn stream_taken_twice_yields_an_error_item() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let _first = driver.stream();
        let second = driver.stream();

        tokio::pin!(second);
        match second.next().await.ok_or("expected an item")? {
            Err(err) => {
                assert!(err.to_string().contains("already taken"));
                Ok(())
            }
            Ok(_) => Err("expected an error item".into()),
        }
    }

    #[tokio::test]
    async fn fail_surfaces_an_error_item_on_the_stream() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        let stream = driver.stream();
        driver.fail("kaboom".into()).await?;

        tokio::pin!(stream);
        match stream.next().await.ok_or("no item")? {
            Err(err) => {
                assert!(err.to_string().contains("kaboom"));
                Ok(())
            }
            Ok(_) => Err("expected an error item".into()),
        }
    }

    #[tokio::test]
    async fn dropping_the_stream_terminates_the_producer() -> Result<(), BoxErr> {
        let driver = TypeErasedDriver::new();
        // Box::pin so the stream is owned here and `drop` actually drops the receiver.
        // (`tokio::pin!` would rebind to a `Pin<&mut _>`, making `drop` a no-op.)
        let mut stream = Box::pin(driver.stream());

        // Producer publishes paced steps until the consumer goes away, then reports
        // how many it managed to send.
        let producer = driver.clone();
        let handle = tokio::spawn(async move {
            let mut sent = 0usize;
            while producer.info(sent).await.is_ok() {
                sent += 1;
            }
            sent
        });

        // Pace two steps, then terminate by dropping the stream.
        for _ in 0..2 {
            match stream.next().await.ok_or("stream ended early")?? {
                DriverStep::Info(_) => {}
                _ => return Err("expected an Info step".into()),
            }
        }
        drop(stream);

        // With the consumer gone, the producer's next publish errors and it stops.
        let sent = handle.await?;
        assert!(sent >= 2, "producer should have published the paced steps");
        Ok(())
    }
}
