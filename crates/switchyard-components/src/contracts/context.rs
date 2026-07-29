// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request context and typed extension storage for compatibility components.

use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::ids::{LlmTargetId, RequestId};
use super::types::ChatRequestType;

/// Mutable state shared across processors and the backend for one request.
#[derive(Default)]
pub struct ProxyContext {
    /// Optional request ID propagated across processors and backends.
    pub request_id: Option<RequestId>,
    /// Optional inbound wire format recorded by endpoint or translation code.
    pub inbound_format: Option<ChatRequestType>,
    /// Optional target selected by a request processor for backend dispatch.
    pub selected_target: Option<LlmTargetId>,
    extensions: Extensions,
}

impl ProxyContext {
    /// Creates an empty request context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context with a known request identifier.
    pub fn with_request_id(request_id: RequestId) -> Self {
        Self {
            request_id: Some(request_id),
            ..Self::default()
        }
    }

    /// Returns read-only access to typed extension values.
    pub fn extensions(&self) -> &Extensions {
        &self.extensions
    }

    /// Returns mutable access to typed extension values.
    pub fn extensions_mut(&mut self) -> &mut Extensions {
        &mut self.extensions
    }

    /// Returns the selected target for backend dispatch.
    pub fn selected_target(&self) -> Option<&LlmTargetId> {
        self.selected_target.as_ref()
    }

    /// Replaces the selected target, returning the previous target.
    pub fn set_selected_target(&mut self, target_id: LlmTargetId) -> Option<LlmTargetId> {
        self.selected_target.replace(target_id)
    }

    /// Clears the selected target.
    pub fn clear_selected_target(&mut self) -> Option<LlmTargetId> {
        self.selected_target.take()
    }

    /// Inserts a typed extension, returning the previous value of the same type.
    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.insert(value)
    }

    /// Gets an immutable typed extension by Rust type.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.get()
    }

    /// Gets a mutable typed extension by Rust type.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.get_mut()
    }

    /// Removes a typed extension and returns it when present.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.extensions.remove()
    }
}

impl fmt::Debug for ProxyContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProxyContext")
            .field("request_id", &self.request_id)
            .field("inbound_format", &self.inbound_format)
            .field("selected_target", &self.selected_target)
            .field("extensions_len", &self.extensions.len())
            .finish()
    }
}

/// Type-indexed storage used for cross-component request metadata.
#[derive(Default)]
pub struct Extensions {
    values: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    /// Creates an empty extension map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores a value under its concrete Rust type.
    pub fn insert<T>(&mut self, value: T) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.values
            .insert(TypeId::of::<T>(), Box::new(value))
            .and_then(|previous| previous.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Gets a value by its concrete Rust type.
    pub fn get<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.values
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref())
    }

    /// Gets a mutable value by its concrete Rust type.
    pub fn get_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.values
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| value.downcast_mut())
    }

    /// Removes a value by its concrete Rust type.
    pub fn remove<T>(&mut self) -> Option<T>
    where
        T: Send + Sync + 'static,
    {
        self.values
            .remove(&TypeId::of::<T>())
            .and_then(|value| value.downcast::<T>().ok())
            .map(|boxed| *boxed)
    }

    /// Returns whether a value of type `T` is present.
    pub fn contains<T>(&self) -> bool
    where
        T: Send + Sync + 'static,
    {
        self.values.contains_key(&TypeId::of::<T>())
    }

    /// Returns the number of stored extension values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the extension map is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl fmt::Debug for Extensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Extensions")
            .field("len", &self.values.len())
            .finish()
    }
}

/// Targets evicted from a routing pool after a `ContextWindowExceeded` failure.
///
/// Stored in `ProxyContext` for compatibility routers that retry the same
/// logical request on a fallback target after an upstream context-window error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictedTargets(HashSet<LlmTargetId>);

impl EvictedTargets {
    /// Returns whether a target has already overflowed on this request.
    pub fn contains(&self, target_id: &LlmTargetId) -> bool {
        self.0.contains(target_id)
    }

    /// Records a target as evicted, returning whether it was newly inserted.
    pub fn insert(&mut self, target_id: LlmTargetId) -> bool {
        self.0.insert(target_id)
    }

    /// Returns whether no targets have been evicted yet.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate the evicted target ids in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &LlmTargetId> {
        self.0.iter()
    }
}
