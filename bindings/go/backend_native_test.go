// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo && switchyard_native

package switchyard

import "testing"

func TestBackend(t *testing.T) {
	if got := Backend(); got != "native" {
		t.Fatalf("Backend() = %q, want native", got)
	}
}
