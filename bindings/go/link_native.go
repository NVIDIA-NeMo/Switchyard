// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo && switchyard_native

package switchyard

// Source builds link the local archive; packaged modules supply package-local paths.

/*
#cgo CFLAGS: -I${SRCDIR}/../../crates/switchyard-ffi/include
#cgo LDFLAGS: ${SRCDIR}/../../target/release/libswitchyard_ffi.a
#cgo linux LDFLAGS: -ldl -lpthread -lm
#cgo darwin LDFLAGS: -framework Security -framework CoreFoundation -liconv
*/
import "C"
