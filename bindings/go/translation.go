// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

/*
#include <stdlib.h>
#include "switchyard.h"
*/
import "C"

import (
	"encoding/json"
	"unsafe"
)

// DecodeRequest converts a provider wire body to Switchyard's normalized request.
func DecodeRequest(format string, body json.RawMessage) (json.RawMessage, error) {
	return translated(struct {
		Format string          `json:"format"`
		Body   json.RawMessage `json:"body"`
	}{format, body}, func(input *C.char, output **C.char) *C.char {
		return C.sy_decode_request(input, output)
	})
}

// EncodeRequest stamps the target model and encodes the prepared normalized request.
func EncodeRequest(format string, request json.RawMessage, model string) (json.RawMessage, error) {
	return translated(struct {
		Format  string          `json:"format"`
		Request json.RawMessage `json:"request"`
		Model   string          `json:"model"`
	}{format, request, model}, func(input *C.char, output **C.char) *C.char {
		return C.sy_encode_request(input, output)
	})
}

// DecodeResponse converts a buffered provider response to Switchyard's normalized aggregate.
func DecodeResponse(format string, body json.RawMessage) (json.RawMessage, error) {
	return translated(struct {
		Format string          `json:"format"`
		Body   json.RawMessage `json:"body"`
	}{format, body}, func(input *C.char, output **C.char) *C.char {
		return C.sy_decode_response(input, output)
	})
}

func translated(input any, call func(*C.char, **C.char) *C.char) (json.RawMessage, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	value := C.CString(string(data))
	defer C.free(unsafe.Pointer(value))
	var output *C.char
	if err := nativeError(call(value, &output)); err != nil {
		return nil, err
	}
	defer C.sy_string_free(output)
	return json.RawMessage(C.GoString(output)), nil
}
