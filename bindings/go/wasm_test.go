// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

// Missing startup exports must return errors instead of panicking.
func TestWasmMissingStartupExport(t *testing.T) {
	runtime, err := newWasmRuntime()
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.closeAll()
	ctx := context.Background()
	// Minimal modules with no exports, or only a no-op _initialize export.
	for name, wasm := range map[string]string{
		"_initialize":    "\x00asm\x01\x00\x00\x00",
		"sy_abi_version": "\x00asm\x01\x00\x00\x00\x01\x04\x01\x60\x00\x00\x03\x02\x01\x00\x07\x0f\x01\x0b_initialize\x00\x00\x0a\x04\x01\x02\x00\x0b",
	} {
		t.Run(name, func(t *testing.T) {
			runtime.compiled, err = runtime.runtime.CompileModule(ctx, []byte(wasm))
			if err != nil {
				t.Fatal(err)
			}
			if guest, err := runtime.newModule(ctx); guest != nil || err == nil || !strings.Contains(err.Error(), "export \""+name+"\" is missing") {
				t.Fatalf("missing %s returned %v", name, err)
			}
		})
	}
}

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

// A caller's expired context must not close an instance that other runs share.
func TestWasmCancelledRunKeepsSharedInstance(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(json.RawMessage(`{"algorithm":`+composite+`,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()

	entered := make(chan struct{})
	release := make(chan struct{})
	result := make(chan error, 1)
	go func() {
		_, err := backend.route(context.Background(), input(), func(ctx context.Context, _ Work) (Reply, error) {
			close(entered)
			select {
			case <-release:
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			case <-ctx.Done():
				return Reply{}, ctx.Err()
			}
		})
		result <- err
	}()
	<-entered

	large := input()
	large.Request = json.RawMessage(`{"messages":[{"role":"user","content":[{"type":"text","text":"` + strings.Repeat("x", 32<<20) + `"}]}]}`)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
	defer cancel()
	if _, err := backend.route(ctx, large, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired run returned %v", err)
	}

	close(release)
	if err := <-result; err != nil {
		t.Fatalf("run sharing the instance failed: %v", err)
	}
}
