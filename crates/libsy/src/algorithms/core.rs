// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::{Driver, Signals};
use switchyard_protocol::{AggLlmResponse, Decision, Request};

/// The per-request accumulator shared across an algorithm's components.
///
/// The [`Processor`] chain runs first and *writes* facts into it (keyed by type); the
/// [`Classifier`] cascade runs next and *reads* those facts to decide. One `State` is
/// threaded through a single request's whole processor→classifier run.
#[derive(Default)]
pub struct State {
    items: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl State {
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.items.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    pub fn get_mut<T: Any + Send + Sync>(&mut self) -> Option<&mut T> {
        self.items.get_mut(&TypeId::of::<T>())?.downcast_mut::<T>()
    }

    pub fn insert<T: Any + Send + Sync>(&mut self, item: T) {
        self.items.insert(TypeId::of::<T>(), Box::new(item));
    }

    pub fn entry_or_insert_with<T: Any + Send + Sync>(&mut self, f: impl FnOnce() -> T) -> &mut T {
        match self
            .items
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(f()))
            .downcast_mut::<T>()
        {
            Some(value) => value,
            // The TypeId key guarantees the stored box holds a T.
            None => unreachable!("State key TypeId matches the stored value type"),
        }
    }
}

/// An event fed to the [`Processor`] chain. Every variant is `Send` so the chain can run
/// inside an algorithm's `Send` run task; a response-side processor therefore observes the
/// *buffered* [`AggLlmResponse`], since a live response stream cannot cross that boundary.
pub enum Event<'a> {
    Request(&'a Request),
    Signal(&'a Signals),
    Decision(&'a dyn Decision),
    ModelRequest(&'a Request),
    ModelResponse(&'a AggLlmResponse),
}

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Head-of-algorithm state collection.
///
/// Processors run first, one after another, before any classifier. Each does *lightweight*
/// work — read an [`Event`] and accumulate facts into the shared [`State`] — so later
/// classifiers can key off it. Heavy work (LLM calls, scoring) belongs in a [`Classifier`],
/// not here.
///
/// `process` returns a `Send` future so a processor chain can run inside an algorithm's
/// `Send` run task — which is why every [`Event`] variant is `Send`.
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process an event, accumulating facts into `state`.
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr>;
}

/// One classifier's recommendation of a routing `target`, with a `[0.0, 1.0]` confidence.
pub struct Score {
    /// `[0.0, 1.0]` confidence in `target`.
    pub confidence: f64,
    /// The target (model / tier) being recommended.
    pub target: String,
}

/// A classification stage in the cascade that runs after the [`Processor`] chain.
///
/// Classifiers are tried in turn — ideally cheapest first (e.g. a heuristic `StagedRouter`),
/// falling back to heavier ones (e.g. an LLM-backed classifier). Each [`score`](Self::score)
/// call reads the accumulated [`State`] and the request, does its (possibly expensive)
/// classification, and returns a [`Score`] per recommended `target`. Returning an **empty**
/// vec **abstains**, deferring to the next classifier; the first classifier to return a
/// non-empty result decides.
///
/// `driver` offloads any model call the classifier itself needs (e.g. an LLM-backed
/// classifier's own call). It is `None` when the classifier is scored outside an algorithm
/// run; a classifier that requires it returns an error rather than assuming one is present.
#[async_trait]
pub trait Classifier: Send + Sync {
    async fn score(
        &self,
        state: &mut State,
        request: &Request,
        driver: Option<&Driver>,
    ) -> Result<Vec<Score>, BoxErr>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // A couple of unrelated concrete state items so we can prove the bag keys by type.
    #[derive(Debug, PartialEq)]
    struct Counter(u32);

    #[derive(Debug, PartialEq)]
    struct Label(String);

    #[test]
    fn get_returns_none_when_absent() {
        let state = State::default();
        assert!(state.get::<Counter>().is_none());
    }

    #[test]
    fn insert_then_get_returns_value() {
        let mut state = State::default();
        state.insert(Counter(7));
        assert_eq!(state.get::<Counter>(), Some(&Counter(7)));
    }

    #[test]
    fn insert_replaces_existing_item_of_same_type() {
        let mut state = State::default();
        state.insert(Counter(1));
        state.insert(Counter(2));
        assert_eq!(state.get::<Counter>(), Some(&Counter(2)));
    }

    #[test]
    fn get_mut_mutates_in_place() {
        let mut state = State::default();
        state.insert(Counter(0));
        let Some(counter) = state.get_mut::<Counter>() else {
            panic!("expected a Counter to be present");
        };
        counter.0 += 5;
        assert_eq!(state.get::<Counter>(), Some(&Counter(5)));
    }

    #[test]
    fn get_mut_returns_none_when_absent() {
        let mut state = State::default();
        assert!(state.get_mut::<Counter>().is_none());
    }

    #[test]
    fn distinct_types_coexist_keyed_by_type() {
        let mut state = State::default();
        state.insert(Counter(3));
        state.insert(Label("hello".to_string()));
        assert_eq!(state.get::<Counter>(), Some(&Counter(3)));
        assert_eq!(state.get::<Label>(), Some(&Label("hello".to_string())));
    }

    #[test]
    fn entry_or_insert_with_inserts_default_when_absent() {
        let mut state = State::default();
        let counter = state.entry_or_insert_with(|| Counter(42));
        assert_eq!(*counter, Counter(42));
    }

    #[test]
    fn entry_or_insert_with_returns_existing_without_calling_default() {
        let mut state = State::default();
        state.insert(Counter(1));
        // The default closure must not run when an item is already present.
        let counter =
            state.entry_or_insert_with::<Counter>(|| panic!("default should not be called"));
        assert_eq!(*counter, Counter(1));
    }

    #[test]
    fn entry_or_insert_with_yields_mutable_handle() {
        let mut state = State::default();
        *state.entry_or_insert_with(|| Counter(10)) = Counter(11);
        assert_eq!(state.get::<Counter>(), Some(&Counter(11)));
    }
}
