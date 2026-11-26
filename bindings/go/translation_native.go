// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo && switchyard_native

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

func translate(export string, input []byte) (json.RawMessage, error) {
	value := C.CString(string(input))
	defer C.free(unsafe.Pointer(value))
	var output *C.char
	var failure *C.char
	switch export {
	case "sy_decode_request":
		failure = C.sy_decode_request(value, &output)
	case "sy_encode_request":
		failure = C.sy_encode_request(value, &output)
	case "sy_decode_response":
		failure = C.sy_decode_response(value, &output)
	case "sy_encode_response":
		failure = C.sy_encode_response(value, &output)
	default:
		panic("unknown Switchyard translation export " + export)
	}
	return nativeOutput(failure, output)
}

// nativeOutput returns a call's JSON output or its error, freeing both.
func nativeOutput(failure, output *C.char) ([]byte, error) {
	if err := nativeError(failure); err != nil {
		return nil, err
	}
	defer C.sy_string_free(output)
	return []byte(C.GoString(output)), nil
}

type nativeStream struct {
	ptr *C.SyStream
}

func newStreamHandle(input []byte) (streamHandle, error) {
	value := C.CString(string(input))
	defer C.free(unsafe.Pointer(value))
	var ptr *C.SyStream
	if err := nativeError(C.sy_stream_new(value, &ptr)); err != nil {
		return nil, err
	}
	return &nativeStream{ptr: ptr}, nil
}

func (stream *nativeStream) translate(event []byte) ([]byte, error) {
	value := C.CString(string(event))
	defer C.free(unsafe.Pointer(value))
	var output *C.char
	failure := C.sy_stream_translate(stream.ptr, value, &output)
	return nativeOutput(failure, output)
}

func (stream *nativeStream) finish() ([]byte, error) {
	var output *C.char
	failure := C.sy_stream_finish(stream.ptr, &output)
	return nativeOutput(failure, output)
}

func (stream *nativeStream) free() {
	C.sy_stream_free(stream.ptr)
}
