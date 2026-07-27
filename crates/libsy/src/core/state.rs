// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use tokio::sync::Mutex;

use crate::ToolSignals;

/// A value in a session's [`State`].
#[derive(Debug, Clone)]
pub enum StateValue {
    String(String),
    Count(u32),
    Int(i32),
    Scalar(f32),
}

/// State maintained by algorithms that use the built-in state model.
#[derive(Debug, Clone, Default)]
pub struct State {
    pub turn_count: u32,
    /// Tool-result signals for the current request, set by the tool-signal
    /// processor. `None` until it runs or when the request has no tool activity,
    /// so routers must treat absence as "no signal yet".
    pub tool_signals: Option<ToolSignals>,
    pub extra: HashMap<String, StateValue>,
}

/// Mutable access to a state value acquired from a [`StateHandle`].
pub trait StateGuard<S>: Send {
    /// Returns the state value guarded for this fall-through run.
    fn get_mut(&mut self) -> &mut S;
}

impl StateGuard<()> for () {
    fn get_mut(&mut self) -> &mut () {
        self
    }
}

impl<S: Send> StateGuard<S> for tokio::sync::MutexGuard<'_, S> {
    fn get_mut(&mut self) -> &mut S {
        self
    }
}

/// Supplies one mutable state value to an algorithm run.
///
/// `()` is the zero-cost stateless handle. [`Shared<S>`] provides state that
/// persists when the same context is reused across turns.
pub trait StateHandle: Clone + Send + Sync + 'static {
    /// State value exposed to processors and classifiers.
    type State: Send + 'static;
    /// Guard retaining exclusive access for the processor/classifier fold.
    type Guard<'a>: StateGuard<Self::State> + Send
    where
        Self: 'a;

    /// Acquires the state value for one algorithm run.
    fn acquire(&self) -> impl Future<Output = Self::Guard<'_>> + Send;
}

impl StateHandle for () {
    type State = ();
    type Guard<'a> = ();

    fn acquire(&self) -> impl Future<Output = Self::Guard<'_>> + Send {
        std::future::ready(())
    }
}

/// Shared, asynchronously locked state stored in an algorithm [`Context`](crate::Context).
#[derive(Debug)]
pub struct Shared<S> {
    inner: Arc<Mutex<S>>,
}

impl<S> Shared<S> {
    /// Creates shared state from an initial value.
    pub fn new(state: S) -> Self {
        Self {
            inner: Arc::new(Mutex::new(state)),
        }
    }

    /// Locks the state for direct inspection or mutation.
    pub async fn lock(&self) -> tokio::sync::MutexGuard<'_, S> {
        self.inner.lock().await
    }
}

impl<S> Clone for Shared<S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S: Default> Default for Shared<S> {
    fn default() -> Self {
        Self::new(S::default())
    }
}

impl<S: Send + 'static> StateHandle for Shared<S> {
    type State = S;
    type Guard<'a>
        = tokio::sync::MutexGuard<'a, S>
    where
        Self: 'a;

    fn acquire(&self) -> impl Future<Output = Self::Guard<'_>> + Send {
        self.lock()
    }
}

/// Compatibility name for the original built-in shared state.
#[deprecated(note = "use Shared<State>")]
pub type SharedState = Shared<State>;
