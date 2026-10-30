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

	"github.com/tetratelabs/wazero/api"
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

type cancelAfterNextModule struct {
	api.Module
	cancel context.CancelFunc
}

func (module cancelAfterNextModule) ExportedFunction(name string) api.Function {
	function := module.Module.ExportedFunction(name)
	if name == "sy_run_next" {
		return cancelAfterCallFunction{Function: function, cancel: module.cancel}
	}
	return function
}

type cancelAfterCallFunction struct {
	api.Function
	cancel context.CancelFunc
}

func (function cancelAfterCallFunction) Call(ctx context.Context, args ...uint64) ([]uint64, error) {
	values, err := function.Function.Call(ctx, args...)
	if err == nil {
		function.cancel()
	}
	return values, err
}

// Cancellation after the final guest call must take precedence over its outcome.
func TestWasmCancellationBeforeOutcome(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(testConfig(`{"type":"stage_router","picker":"efficient_first",`+stage+`}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	guest := backend.slots[0].gen.guest
	guest.module = cancelAfterNextModule{Module: guest.module, cancel: cancel}
	if _, err := backend.route(ctx, input(), nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled route returned %v", err)
	}
}

// Parent and child identities must share the guest that owns their session state.
func TestWasmSessionAffinity(t *testing.T) {
	backend := &wasmBackend{slots: []*wasmSlot{{}, {}}}
	request := input()
	request.Metadata = &RequestMetadata{SessionID: "shared"}
	parent := backend.slotFor(request)
	request.Request = json.RawMessage(`{"messages":[{"role":"user","content":[{"type":"text","text":"child task"}]}]}`)
	for _, agent := range []string{"a", "b", ""} {
		request.Metadata.IsSubagent = true
		request.Metadata.AgentID = agent
		if backend.slotFor(request) != parent {
			t.Errorf("child %q was assigned outside its parent's session guest", agent)
		}
	}
}

func TestWasmTrapIsolatesAndReplacesInstance(t *testing.T) {
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(input())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	first, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer first.close()
	second, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer second.close()
	other, err := backend.slots[1].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer other.close()

	backend.slots[0].mu.Lock()
	_, allocationErr := first.gen.guest.call(ctx, "sy_alloc", uint64(^uint32(0)))
	backend.slots[0].mu.Unlock()
	if !isWasmTrap(allocationErr) {
		t.Fatalf("forced guest allocation panic returned %v", allocationErr)
	}
	first.ptr = ^uint32(0)
	if _, err := first.next(ctx); err == nil || !strings.Contains(err.Error(), "trapped") {
		t.Fatalf("forced guest trap returned %v", err)
	}
	if _, err := second.next(ctx); err == nil || !strings.Contains(err.Error(), "trapped") {
		t.Fatalf("run sharing the trapped instance returned %v", err)
	}
	step, err := other.next(ctx)
	if err != nil {
		t.Fatalf("unrelated instance failed: %v", err)
	}
	if !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("unrelated run returned %s", step)
	}

	replacement, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatalf("pool did not replace the trapped instance: %v", err)
	}
	defer replacement.close()
	step, err = replacement.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("replacement returned step=%s err=%v", step, err)
	}
}

// A failed replacement must not prevent a later request from recovering the slot.
func TestWasmFailedReplacementIsRetried(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(testConfig(`{"type":"stage_router","picker":"efficient_first",`+stage+`}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	request := input()
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	slot := backend.slots[0]
	old, err := slot.startRun(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	defer old.close()
	slot.mu.Lock()
	config := slot.config
	slot.config = []byte(`{`)
	slot.poisonLocked(old.gen, errors.New("test trap"))
	failed := slot.err != nil
	slot.config = config
	slot.mu.Unlock()
	if !failed {
		t.Fatal("replacement with invalid config succeeded")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := slot.startRun(cancelled, data); !errors.Is(err, context.Canceled) || slot.gen != old.gen || slot.err == nil {
		t.Fatalf("cancelled request attempted replacement: %v", err)
	}
	outcome, err := backend.route(ctx, request, nil)
	if err != nil || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != "weak-model" {
		t.Fatalf("recovered route returned outcome=%+v err=%v", outcome, err)
	}
	if slot.err != nil || slot.gen == old.gen {
		t.Fatal("recovery did not replace the failed generation and clear its error")
	}
	if _, err := old.next(ctx); err == nil {
		t.Fatal("recovery revived a failed run")
	}
}

func TestWasmClosedInstanceIsReplaced(t *testing.T) {
	config := testConfig(`{"type":"stage_router","picker":"efficient_first",` + stage + `}`)
	backend, err := newWasmBackendWithPoolSize(config, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	input, err := json.Marshal(input())
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	closedRun, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer closedRun.close()
	otherRun, err := backend.slots[1].startRun(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	defer otherRun.close()
	if err := closedRun.gen.guest.module.Close(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := closedRun.next(ctx); err == nil {
		t.Fatal("call on a closed instance succeeded")
	}
	step, err := otherRun.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("unrelated instance returned step=%s err=%v", step, err)
	}
	replacement, err := backend.slots[0].startRun(ctx, input)
	if err != nil {
		t.Fatalf("closed instance was not replaced: %v", err)
	}
	defer replacement.close()
	step, err = replacement.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("replacement returned step=%s err=%v", step, err)
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

func TestWasmTrapIncludesGuestPanicMessage(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(testConfig(`{"type":"stage_router","picker":"efficient_first",`+stage+`}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	slot := backend.slots[0]
	slot.mu.Lock()
	_, err = slot.gen.guest.call(context.Background(), "sy_alloc", uint64(^uint32(0)))
	slot.mu.Unlock()
	if !isWasmTrap(err) || !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("guest panic returned %v", err)
	}
}

// Pending calls and same-session waiters yield without blocking other runs.
func TestWasmNextWhileHostCallPending(t *testing.T) {
	backend, err := newWasmBackendWithPoolSize(json.RawMessage(`{"algorithm":`+composite+`,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`), 1)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.close()
	request := input()
	request.Metadata = &RequestMetadata{SessionID: "shared"}
	data, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	run, err := backend.slots[0].startRun(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	defer run.close()
	step, err := run.next(ctx)
	if err != nil || !strings.Contains(string(step), `"type":"call_model"`) {
		t.Fatalf("first step=%s err=%v", step, err)
	}
	if step, err := run.next(ctx); err != nil || step != nil {
		t.Fatalf("read with an unanswered call returned step=%s err=%v", step, err)
	}
	waiting, err := backend.slots[0].startRun(ctx, data)
	if err != nil {
		t.Fatal(err)
	}
	if step, err := waiting.next(ctx); err != nil || step != nil {
		waiting.close()
		t.Fatalf("same-session waiter returned step=%s err=%v", step, err)
	}
	waiting.close()
	var work Work
	if err := json.Unmarshal(step, &work); err != nil {
		t.Fatal(err)
	}
	if err := run.respond(ctx, work.ID, []byte(`{"response":`+judgeAnswer+`}`)); err != nil {
		t.Fatal(err)
	}
	if step, err := run.next(ctx); err != nil || !strings.Contains(string(step), `"type":"done"`) {
		t.Fatalf("surviving run returned step=%s err=%v", step, err)
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
