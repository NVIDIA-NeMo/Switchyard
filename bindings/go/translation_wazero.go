// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import "encoding/json"

func translate(export string, input []byte) (json.RawMessage, error) {
	return wasmTranslate(export, input)
}
