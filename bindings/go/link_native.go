// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo && switchyard_native

package switchyard

/*
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/libswitchyard_ffi_linux_amd64.a
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/libswitchyard_ffi_linux_arm64.a
#cgo linux LDFLAGS: -ldl -lpthread -lm
#cgo darwin LDFLAGS: -framework Security -framework CoreFoundation -liconv
*/
import "C"
