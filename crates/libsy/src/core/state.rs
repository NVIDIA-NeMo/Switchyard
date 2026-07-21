// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Holds state for an [`Algorithm`](crate::Algorithm) across turns.
///
/// A type-keyed bag: each Rust type has at most one stored value, so components share
/// state by agreeing on a type rather than a string key. State is accumulated / mutated
/// by [`crate::Processor`]s and referenced by [`crate::Classifier`]s.
#[derive(Default)]
pub struct State {
    items: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl State {
    /// Borrows the stored value of type `T`, or `None` if none is present.
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.items.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }

    /// Mutably borrows the stored value of type `T`, or `None` if none is present.
    pub fn get_mut<T: Any + Send + Sync>(&mut self) -> Option<&mut T> {
        self.items.get_mut(&TypeId::of::<T>())?.downcast_mut::<T>()
    }

    /// Stores `item`, replacing any existing value of the same type.
    pub fn insert<T: Any + Send + Sync>(&mut self, item: T) {
        self.items.insert(TypeId::of::<T>(), Box::new(item));
    }

    /// Returns a mutable handle to the stored `T`, first inserting `f()` if none is present.
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
