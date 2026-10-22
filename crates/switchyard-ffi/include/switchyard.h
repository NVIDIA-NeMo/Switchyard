// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWITCHYARD_H
#define SWITCHYARD_H

#include <stdint.h>

// Compatibility generation for C signatures and JSON payloads.
#define SY_ABI_VERSION 1
uint32_t sy_abi_version(void);

// Fallible calls return NULL on success or an owned error string.
// JSON inputs are borrowed for the call and must be NUL-terminated.
// Free all output strings with sy_string_free.
char *sy_decode_request(const char *input_json, char **out_json);
char *sy_encode_request(const char *input_json, char **out_json);
char *sy_decode_response(const char *input_json, char **out_json);

void sy_string_free(char *string);

#endif
