// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::Signals;
use switchyard_protocol::{Decision, Request, Response};

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

pub enum Event<'a> {
    Request(&'a Request),
    Signal(&'a Signals),
    Decision(&'a dyn Decision),
    ModelRequest(&'a Request),
    ModelResponse(&'a Response),
}

type BoxErr = Box<dyn std::error::Error + Send + Sync>;

/// Processor mutates state given an event
// `Event` can carry a `&Response`, whose streaming body is `!Sync`, so the returned
// future cannot be `Send`; opt out of async_trait's default `Send` bound.
#[async_trait(?Send)]
pub trait Processor: Send + Sync {
    /// Process an event, mutating the state and returning a result.
    async fn process(&self, state: &mut State, event: Event<'_>) -> Result<(), BoxErr>;
}

pub struct Score {
    /// [0.0, 1.0] confidence score
    pub confidence: f64,
    pub target: String,
}

#[async_trait]
pub trait Classifier: Send + Sync {
    async fn score(&self, state: &mut State, request: &Request) -> Result<Vec<Score>, BoxErr>;
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
