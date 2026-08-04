// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};

use tokio::runtime::{Builder, Handle};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;

/// Plugin-owned async executor.
///
/// The public native-plugin SDK uses synchronous Rust callbacks and pull-based
/// iterators at the dynamic-library boundary. Switchyard performs provider I/O
/// on this dedicated runtime rather than entering Relay's Tokio runtime from a
/// separately linked cdylib.
#[derive(Clone)]
pub(crate) struct PluginExecutor {
    inner: Arc<ExecutorInner>,
}

struct ExecutorInner {
    handle: Handle,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl PluginExecutor {
    pub(crate) fn new() -> Result<Self, String> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("switchyard-relay-http".into())
            .spawn(move || {
                let runtime = match Builder::new_multi_thread()
                    .worker_threads(2)
                    .thread_name("switchyard-relay-http-worker")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                let (shutdown_tx, shutdown_rx) = oneshot::channel();
                if ready_tx
                    .send(Ok((runtime.handle().clone(), shutdown_tx)))
                    .is_err()
                {
                    return;
                }
                runtime.block_on(async {
                    let _ = shutdown_rx.await;
                });
            })
            .map_err(|error| format!("failed to start Switchyard HTTP runtime: {error}"))?;
        let (handle, shutdown) = ready_rx
            .recv()
            .map_err(|_| "Switchyard HTTP runtime stopped during startup".to_string())??;
        Ok(Self {
            inner: Arc::new(ExecutorInner {
                handle,
                shutdown: Mutex::new(Some(shutdown)),
                thread: Mutex::new(Some(thread)),
            }),
        })
    }

    pub(crate) fn spawn<F>(&self, future: F) -> AbortHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.inner.handle.spawn(future).abort_handle()
    }

    pub(crate) fn run<F>(&self, future: F) -> Result<F::Output, String>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.inner.handle.spawn(async move {
            let _ = sender.send(future.await);
        });
        receiver
            .recv()
            .map_err(|_| "Switchyard HTTP runtime stopped before completing work".to_string())
    }
}

impl Drop for ExecutorInner {
    fn drop(&mut self) {
        if let Some(shutdown) = self
            .shutdown
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self
            .thread
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            if std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("switchyard-relay-http-worker"))
            {
                // The runtime owner will join this worker after the current
                // task returns. Waiting here would deadlock that shutdown.
                drop(thread);
            } else {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_runs_buffered_and_spawned_work() {
        let executor = PluginExecutor::new().unwrap();
        assert_eq!(executor.run(async { 42 }).unwrap(), 42);
        let (sender, receiver) = mpsc::sync_channel(1);
        executor.spawn(async move {
            sender.send("done").unwrap();
        });
        assert_eq!(receiver.recv().unwrap(), "done");
    }

    #[test]
    fn last_reference_can_drop_on_a_worker() {
        let executor = PluginExecutor::new().unwrap();
        let worker_reference = executor.clone();
        let (sender, receiver) = mpsc::sync_channel(1);
        executor.spawn(async move {
            drop(worker_reference);
            sender.send(()).unwrap();
        });
        drop(executor);
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("dropping the executor on its own worker must not deadlock");
    }
}
