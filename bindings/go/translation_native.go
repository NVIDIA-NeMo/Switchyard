// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo

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
	default:
		panic("unknown Switchyard translation export " + export)
	}
	if err := nativeError(failure); err != nil {
		return nil, err
	}
	defer C.sy_string_free(output)
	return json.RawMessage(C.GoString(output)), nil
}
