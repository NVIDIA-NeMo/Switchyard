// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit opt-in for HTTP body capture by a host's tracing subscriber.

use std::fmt;
use std::future::Future;

tokio::task_local! {
    static CAPTURE_ENABLED: ();
}

/// Enables raw HTTP body events while polling `future` on the current task.
///
/// The host must install a subscriber for `switchyard.payload` and apply its
/// own filtering, size limits, and redaction. Spawned tasks do not inherit capture.
pub async fn with_payload_capture<T>(future: impl Future<Output = T>) -> T {
    CAPTURE_ENABLED.scope((), future).await
}

pub(crate) fn record<D: fmt::Display>(phase: &'static str, body: impl FnOnce() -> D) {
    if CAPTURE_ENABLED.try_with(|_| ()).is_ok() {
        tracing::trace!(target: "switchyard.payload", phase, body = %body());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use tracing::field::Field;
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context, SubscriberExt};

    use super::*;

    struct Recorder(Arc<Mutex<Vec<String>>>);

    impl<S: tracing::Subscriber> Layer<S> for Recorder {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            event.record(&mut |field: &Field, value: &dyn fmt::Debug| {
                if field.name() == "body" {
                    self.0.lock().push(format!("{value:?}"));
                }
            });
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn capture_requires_opt_in_and_does_not_leak_between_tasks_or_after_cancellation() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry().with(Recorder(Arc::clone(&events)));
        async {
            record("request", || -> &str {
                panic!("disabled capture must not evaluate the body")
            });
            tokio::join!(
                with_payload_capture(async {
                    tokio::spawn(
                        async {
                            record("request", || "spawned");
                        }
                        .with_current_subscriber(),
                    )
                    .await
                    .unwrap();
                    tokio::task::yield_now().await;
                    record("request", || "enabled");
                }),
                async {
                    tokio::task::yield_now().await;
                    record("request", || "other-task");
                },
            );
            let mut pending = Box::pin(with_payload_capture(std::future::pending::<()>()));
            assert!(futures::poll!(&mut pending).is_pending());
            drop(pending);
            record("response", || "after-cancellation");
        }
        .with_subscriber(subscriber)
        .await;
        assert_eq!(*events.lock(), ["enabled"]);
    }
}
