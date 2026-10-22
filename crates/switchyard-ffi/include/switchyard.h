// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWITCHYARD_H
#define SWITCHYARD_H

#include <stdint.h>

// Compatibility generation for C signatures and JSON payloads.
#define SY_ABI_VERSION 1
uint32_t sy_abi_version(void);

typedef struct SyRouter SyRouter;
typedef struct SyRun SyRun;
typedef void (*SyNextCallback)(uintptr_t user_data, char *error, char *step_json);

// Fallible calls return NULL on success or an owned error string.
// JSON inputs are borrowed for the call and must be NUL-terminated.
// Free all output strings, including callback strings, with sy_string_free.
char *sy_router_new(const char *config_json, SyRouter **out);
char *sy_run_new(const SyRouter *router, const char *input_json, SyRun **out);

// One pending read per run; other runs execute independently. Replies and cancellation may overlap.
// NULL accepts one read (one callback, even if cancelled); errors reject it (no callback).
// Callbacks may overlap or run before return. Keep callback state alive until they return.
// Callbacks must return promptly, not unwind, and not call sy_run_respond or free handles.
char *sy_run_next_async(const SyRun *run, SyNextCallback callback, uintptr_t user_data);
char *sy_run_respond(const SyRun *run, uint64_t call_id, const char *reply_json);

char *sy_decode_request(const char *input_json, char **out_json);
char *sy_encode_request(const char *input_json, char **out_json);
char *sy_decode_response(const char *input_json, char **out_json);

// Signals cancellation without waiting for callbacks.
void sy_run_cancel(const SyRun *run);
// Free a run once, after other calls using it return; accepted callbacks may still be active.
// Cancels and waits for those callbacks. Do not call from a callback or Tokio task.
void sy_run_free(SyRun *run);
// Free a router once, with no calls using it. Existing runs retain their engine.
void sy_router_free(SyRouter *router);
void sy_string_free(char *string);

#endif
