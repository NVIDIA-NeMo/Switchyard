// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// The private runtime uses the same JSON codec ABI as the native backend.
func TestWasmTranslation(t *testing.T) {
	request, err := wasmTranslate("sy_decode_request", []byte(`{"format":"openai_chat","body":{"messages":[{"role":"user","content":"hi"}]}}`))
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Messages []struct {
			Role    string
			Content []struct{ Text string }
		}
	}
	if err := json.Unmarshal(request, &decoded); err != nil || len(decoded.Messages) != 1 || decoded.Messages[0].Role != "user" || len(decoded.Messages[0].Content) != 1 || decoded.Messages[0].Content[0].Text != "hi" {
		t.Fatalf("request=%s err=%v", request, err)
	}
}

func TestWasmTrapIncludesGuestPanicMessage(t *testing.T) {
	runtime, err := newWasmRuntime()
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.closeAll()
	guest, err := runtime.newModule(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_, err = guest.call(context.Background(), "sy_alloc", uint64(^uint32(0)))
	if !isWasmTrap(err) || !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("guest panic returned %v", err)
	}
}

func TestWasmTranslatorIsReplacedAfterFailure(t *testing.T) {
	input := []byte(`{"format":"openai_chat","body":{"messages":[{"role":"user","content":"hi"}]}}`)
	if _, err := wasmTranslate("sy_decode_request", input); err != nil {
		t.Fatal(err)
	}
	translator := defaultWasmTranslator.value
	translator.mu.Lock()
	err := translator.guest.module.Close(context.Background())
	translator.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wasmTranslate("sy_decode_request", input); err == nil {
		t.Fatal("translation on a closed instance succeeded")
	}
	if _, err := wasmTranslate("sy_decode_request", input); err != nil {
		t.Fatalf("translator was not replaced: %v", err)
	}
}
