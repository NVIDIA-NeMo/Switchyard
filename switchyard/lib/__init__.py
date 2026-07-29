# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core library — Rust-backed chat request/response values, routing, and profiles.

This subpackage holds the protocol-agnostic building blocks used across the rest
of the library:

- ``ChatRequest`` — Rust-backed request values (OpenAI, Responses, Anthropic)
- ``chat_response`` — Rust-backed response values plus Python stream adapters
- ``translation`` — pure format-conversion functions and typed translation engines
- ``processors`` — reusable request/response components used by profiles
- ``backends`` — LLM backend implementations (OpenAI, Anthropic, multi-tier routing)
- ``profiles`` — profile-owned runtime and programmatic config abstractions
- ``roles`` — backend role definitions and translation response aliases
"""
