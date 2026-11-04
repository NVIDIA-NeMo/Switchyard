//go:build cgo && switchyard_native

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A C host exercises Go routing linked with the Rust static library.
package main

import "C"

import (
	"context"
	"encoding/json"
	"time"

	switchyard "github.com/NVIDIA-NeMo/Switchyard/bindings/go"
)

//export SwitchyardSmoke
func SwitchyardSmoke() C.int {
	router, err := switchyard.New(json.RawMessage(`{"algorithm":{"type":"random","targets":["target"]},"targets":{"target":"model"}}`))
	if err != nil {
		return 1
	}
	defer router.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	outcome, err := router.Route(ctx, switchyard.Input{
		Request: json.RawMessage(`{"messages":[]}`),
		Models:  map[string][]string{"any": {"model"}},
	}, nil)
	if err != nil || len(outcome.SelectedModelIDs) != 1 || outcome.SelectedModelIDs[0] != "model" {
		return 2
	}
	return 0
}

func main() {}
