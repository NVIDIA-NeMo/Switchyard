// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

/*
#include "switchyard.h"
*/
import "C"

import "runtime/cgo"

// The handle stays registered until native free has joined every accepted callback.
//
//export syGoNext
func syGoNext(token C.uintptr_t, failure *C.char, output *C.char) {
	result := nextResult{err: nativeError(failure)}
	if output != nil {
		result.data = []byte(C.GoString(output))
		C.sy_string_free(output)
	}
	cgo.Handle(token).Value().(chan nextResult) <- result
}
