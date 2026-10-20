// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
)

type wasmBackend struct {
	runtime   *wasmRuntime
	slots     []*wasmSlot
	next      atomic.Uint64
	active    atomic.Int64
	closed    atomic.Bool
	lifecycle sync.Mutex
	finalize  sync.Once
}

type wasmSlot struct {
	mu     sync.Mutex
	owner  *wasmBackend
	config []byte
	gen    *wasmGeneration
	err    error
}

type wasmGeneration struct {
	guest    *guestModule
	router   uint32
	runs     map[*wasmRun]struct{}
	done     chan struct{}
	errMu    sync.RWMutex
	err      error
	poisoned bool
}

func (generation *wasmGeneration) error() error {
	generation.errMu.RLock()
	defer generation.errMu.RUnlock()
	return generation.err
}

func (generation *wasmGeneration) setError(err error) {
	generation.errMu.Lock()
	generation.err = err
	generation.errMu.Unlock()
}

type wasmRun struct {
	slot   *wasmSlot
	gen    *wasmGeneration
	ptr    uint32
	ctx    context.Context
	cancel context.CancelFunc
}

func newWasmBackend(config json.RawMessage) (engineBackend, error) {
	return newWasmBackendWithPoolSize(config, runtime.GOMAXPROCS(0))
}

func newWasmBackendWithPoolSize(config json.RawMessage, size int) (*wasmBackend, error) {
	if size < 1 {
		return nil, errors.New("Switchyard wasm instance pool must have at least one instance")
	}
	runtime, err := newWasmRuntime()
	if err != nil {
		return nil, err
	}
	backend := &wasmBackend{runtime: runtime, slots: make([]*wasmSlot, size)}
	for index := range backend.slots {
		slot := &wasmSlot{owner: backend, config: bytesClone(config)}
		slot.gen, err = backend.newGeneration(context.Background(), slot.config)
		if err != nil {
			backend.close()
			return nil, fmt.Errorf("initialize Switchyard wasm instance %d: %w", index, err)
		}
		backend.slots[index] = slot
	}
	return backend, nil
}

func bytesClone(value []byte) []byte {
	return append([]byte(nil), value...)
}

func (backend *wasmBackend) newGeneration(ctx context.Context, config []byte) (*wasmGeneration, error) {
	guest, err := backend.runtime.newModule(ctx)
	if err != nil {
		return nil, err
	}
	gen := &wasmGeneration{
		guest: guest,
		runs:  make(map[*wasmRun]struct{}),
		done:  make(chan struct{}),
	}
	gen.router, err = gen.newRouter(ctx, config)
	if err != nil {
		_ = guest.module.Close(context.Background())
		return nil, err
	}
	return gen, nil
}

func (generation *wasmGeneration) newRouter(ctx context.Context, config []byte) (uint32, error) {
	inputPtr, inputLen, err := generation.guest.cstring(ctx, config)
	if err != nil {
		return 0, err
	}
	defer generation.guest.free(context.Background(), inputPtr, inputLen)
	outPtr, err := generation.guest.alloc(ctx, make([]byte, 4))
	if err != nil {
		return 0, err
	}
	defer generation.guest.free(context.Background(), outPtr, 4)
	values, err := generation.guest.call(ctx, "sy_router_new", uint64(inputPtr), uint64(outPtr))
	if err != nil {
		return 0, err
	}
	if len(values) != 1 {
		return 0, errors.New("Switchyard wasm router creation returned an invalid result")
	}
	if err := generation.guest.guestError(uint32(values[0])); err != nil {
		return 0, err
	}
	ptr, ok := generation.guest.module.Memory().ReadUint32Le(outPtr)
	if !ok || ptr == 0 {
		return 0, errors.New("Switchyard wasm returned no router handle")
	}
	return ptr, nil
}

func (generation *wasmGeneration) newRun(ctx context.Context, router uint32, input []byte) (uint32, error) {
	inputPtr, inputLen, err := generation.guest.cstring(ctx, input)
	if err != nil {
		return 0, err
	}
	defer generation.guest.free(context.Background(), inputPtr, inputLen)
	outPtr, err := generation.guest.alloc(ctx, make([]byte, 4))
	if err != nil {
		return 0, err
	}
	defer generation.guest.free(context.Background(), outPtr, 4)
	values, err := generation.guest.call(ctx, "sy_run_new", uint64(router), uint64(inputPtr), uint64(outPtr))
	if err != nil {
		return 0, err
	}
	if len(values) != 1 {
		return 0, errors.New("Switchyard wasm run creation returned an invalid result")
	}
	if err := generation.guest.guestError(uint32(values[0])); err != nil {
		return 0, err
	}
	ptr, ok := generation.guest.module.Memory().ReadUint32Le(outPtr)
	if !ok || ptr == 0 {
		return 0, errors.New("Switchyard wasm returned no run handle")
	}
	return ptr, nil
}

func (generation *wasmGeneration) next(ctx context.Context, run uint32) ([]byte, error) {
	outPtr, err := generation.guest.alloc(ctx, make([]byte, 4))
	if err != nil {
		return nil, err
	}
	defer generation.guest.free(context.Background(), outPtr, 4)
	values, err := generation.guest.call(ctx, "sy_run_next", uint64(run), uint64(outPtr))
	if err != nil {
		return nil, err
	}
	if len(values) != 1 {
		return nil, errors.New("Switchyard wasm next returned an invalid result")
	}
	if err := generation.guest.guestError(uint32(values[0])); err != nil {
		return nil, err
	}
	ptr, ok := generation.guest.module.Memory().ReadUint32Le(outPtr)
	if !ok {
		return nil, errors.New("Switchyard wasm next step is outside guest memory")
	}
	if ptr == 0 {
		// The run is waiting on host replies.
		return nil, nil
	}
	output, err := generation.guest.readCString(ptr)
	if err != nil {
		return nil, err
	}
	if _, err := generation.guest.call(context.Background(), "sy_string_free", uint64(ptr)); err != nil {
		return nil, err
	}
	return output, nil
}

func (generation *wasmGeneration) respond(ctx context.Context, run uint32, id uint64, reply []byte) error {
	replyPtr, replyLen, err := generation.guest.cstring(ctx, reply)
	if err != nil {
		return err
	}
	defer generation.guest.free(context.Background(), replyPtr, replyLen)
	values, err := generation.guest.call(ctx, "sy_run_respond", uint64(run), id, uint64(replyPtr))
	if err != nil {
		return err
	}
	if len(values) != 1 {
		return errors.New("Switchyard wasm respond returned an invalid result")
	}
	return generation.guest.guestError(uint32(values[0]))
}

func (backend *wasmBackend) close() {
	backend.lifecycle.Lock()
	if !backend.closed.CompareAndSwap(false, true) {
		backend.lifecycle.Unlock()
		return
	}
	finalize := backend.active.Load() == 0
	backend.lifecycle.Unlock()
	if finalize {
		backend.finalizeClose()
	}
}

func (backend *wasmBackend) finalizeClose() {
	backend.finalize.Do(func() {
		for _, slot := range backend.slots {
			if slot == nil {
				continue
			}
			slot.mu.Lock()
			slot.closeLocked(errors.New("Switchyard router is closed"))
			slot.mu.Unlock()
		}
		backend.runtime.closeAll()
	})
}

func (slot *wasmSlot) closeLocked(cause error) {
	gen := slot.gen
	if gen == nil || gen.poisoned {
		return
	}
	gen.poisoned = true
	gen.setError(cause)
	close(gen.done)
	for run := range gen.runs {
		run.cancel()
	}
	if gen.router != 0 {
		_, _ = gen.guest.call(context.Background(), "sy_router_free", uint64(gen.router))
		gen.router = 0
	}
	_ = gen.guest.module.Close(context.Background())
}

func (slot *wasmSlot) poisonLocked(gen *wasmGeneration, cause error) {
	if slot.gen != gen || gen.poisoned {
		return
	}
	gen.poisoned = true
	state := "trapped"
	if errors.Is(cause, context.Canceled) || errors.Is(cause, context.DeadlineExceeded) {
		state = "closed"
	}
	gen.setError(fmt.Errorf("Switchyard wasm instance %s: %w", state, cause))
	close(gen.done)
	for run := range gen.runs {
		run.cancel()
	}
	_ = gen.guest.module.Close(context.Background())
	if slot.owner.closed.Load() {
		return
	}
	replacement, err := slot.owner.newGeneration(context.Background(), slot.config)
	if err != nil {
		slot.err = fmt.Errorf("replace trapped Switchyard wasm instance: %w", err)
		return
	}
	slot.gen = replacement
	slot.err = nil
}

func (backend *wasmBackend) route(ctx context.Context, input Input, call HostCall) (Outcome, error) {
	if backend.closed.Load() {
		return Outcome{}, errors.New("Switchyard router is closed")
	}
	data, err := json.Marshal(input)
	if err != nil {
		return Outcome{}, err
	}
	slot := backend.slotFor(input)
	run, err := slot.startRun(ctx, data)
	if err != nil {
		return Outcome{}, err
	}
	defer run.close()
	// Host calls run concurrently, as on the native backend; replies are delivered
	// whenever the guest reports that it needs one to progress.
	replies := make(chan wasmHostReply)
	pending := 0
	for {
		if err := ctx.Err(); err != nil {
			return Outcome{}, err
		}
		stepData, err := run.next(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return Outcome{}, ctx.Err()
			}
			return Outcome{}, err
		}
		if stepData == nil {
			if pending == 0 {
				return Outcome{}, errors.New("Switchyard wasm run is waiting without a host call")
			}
			reply, err := run.awaitReply(replies)
			if err != nil {
				return Outcome{}, err
			}
			pending--
			if err := run.deliver(ctx, reply); err != nil {
				return Outcome{}, err
			}
			continue
		}
		var step struct {
			Type    string          `json:"type"`
			ID      uint64          `json:"id"`
			Models  []string        `json:"models"`
			Model   string          `json:"model"`
			Request json.RawMessage `json:"request"`
		}
		if err := json.Unmarshal(stepData, &step); err != nil {
			return Outcome{}, err
		}
		switch step.Type {
		case "done":
			var outcome Outcome
			if err := json.Unmarshal(stepData, &outcome); err != nil {
				return Outcome{}, err
			}
			return outcome, nil
		case "call_model", "call_decision":
			if call == nil {
				return Outcome{}, errors.New("Switchyard algorithm requires a host model client")
			}
			work := Work{Type: step.Type, ID: step.ID, Models: step.Models, Model: step.Model, Request: step.Request}
			go run.hostCall(call, work, replies)
			pending++
		default:
			return Outcome{}, errors.New("unknown Switchyard step: " + step.Type)
		}
	}
}

func (backend *wasmBackend) slotFor(input Input) *wasmSlot {
	if metadata := input.Metadata; metadata != nil && metadata.SessionID != "" {
		key := "session\x00" + metadata.SessionID
		if metadata.IsSubagent {
			if metadata.AgentID == "" {
				key = ""
			} else {
				key = "subagent\x00" + metadata.SessionID + "\x00" + metadata.AgentID
			}
		}
		if key != "" {
			return backend.slotForKey(key)
		}
	}
	// Mirrors libsy's message-hash fallback in retention_key so requests that libsy keys
	// together share one instance, and with it the state retained under that key.
	if input.Metadata == nil || !input.Metadata.IsSubagent {
		if text, ok := firstUserText(input.Request); ok {
			return backend.slotForKey("message\x00" + text)
		}
	}
	index := backend.next.Add(1) - 1
	return backend.slots[index%uint64(len(backend.slots))]
}

func (backend *wasmBackend) slotForKey(key string) *wasmSlot {
	hash := fnv.New64a()
	_, _ = hash.Write([]byte(key))
	return backend.slots[hash.Sum64()%uint64(len(backend.slots))]
}

// firstUserText matches libsy's Message::text_content("") on the first user message.
func firstUserText(request json.RawMessage) (string, bool) {
	var parsed struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"messages"`
	}
	if json.Unmarshal(request, &parsed) != nil {
		return "", false
	}
	for _, message := range parsed.Messages {
		if message.Role != "user" {
			continue
		}
		var text strings.Builder
		found := false
		for _, block := range message.Content {
			if block.Type == "text" || block.Type == "refusal" {
				text.WriteString(block.Text)
				found = true
			}
		}
		return text.String(), found
	}
	return "", false
}

func (slot *wasmSlot) startRun(ctx context.Context, input []byte) (*wasmRun, error) {
	slot.mu.Lock()
	defer slot.mu.Unlock()
	slot.owner.lifecycle.Lock()
	defer slot.owner.lifecycle.Unlock()
	if slot.owner.closed.Load() {
		return nil, errors.New("Switchyard router is closed")
	}
	if slot.err != nil {
		return nil, slot.err
	}
	gen := slot.gen
	if gen == nil || gen.poisoned {
		return nil, errors.New("Switchyard wasm instance is unavailable")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Guest calls never see the caller's context: wazero closes the whole shared module
	// when a call's context ends. The route loop checks ctx between steps instead.
	ptr, err := gen.newRun(context.Background(), gen.router, input)
	if err != nil {
		if gen.guest.module.IsClosed() || isWasmTrap(err) {
			slot.poisonLocked(gen, err)
		}
		if isWasmTrap(err) {
			return nil, gen.error()
		}
		return nil, err
	}
	runCtx, cancel := context.WithCancel(ctx)
	run := &wasmRun{slot: slot, gen: gen, ptr: ptr, ctx: runCtx, cancel: cancel}
	gen.runs[run] = struct{}{}
	slot.owner.active.Add(1)
	return run, nil
}

func (run *wasmRun) next(ctx context.Context) ([]byte, error) {
	slot := run.slot
	slot.mu.Lock()
	defer slot.mu.Unlock()
	if slot.gen != run.gen || run.gen.poisoned {
		return nil, run.gen.error()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	output, err := run.gen.next(context.Background(), run.ptr)
	if err != nil {
		if run.gen.guest.module.IsClosed() || isWasmTrap(err) {
			slot.poisonLocked(run.gen, err)
			return nil, run.gen.error()
		}
		return nil, err
	}
	return output, nil
}

func (run *wasmRun) respond(ctx context.Context, id uint64, reply []byte) error {
	slot := run.slot
	slot.mu.Lock()
	defer slot.mu.Unlock()
	if slot.gen != run.gen || run.gen.poisoned {
		return run.gen.error()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := run.gen.respond(context.Background(), run.ptr, id, reply); err != nil {
		if run.gen.guest.module.IsClosed() || isWasmTrap(err) {
			slot.poisonLocked(run.gen, err)
			return run.gen.error()
		}
		return err
	}
	return nil
}

type wasmHostReply struct {
	id    uint64
	reply Reply
	err   error
}

func (run *wasmRun) hostCall(call HostCall, work Work, replies chan<- wasmHostReply) {
	reply, err := call(run.ctx, work)
	select {
	case replies <- wasmHostReply{work.ID, reply, err}:
	case <-run.ctx.Done():
	}
}

func (run *wasmRun) awaitReply(replies <-chan wasmHostReply) (wasmHostReply, error) {
	select {
	case reply := <-replies:
		return reply, nil
	case <-run.ctx.Done():
		if err := run.gen.error(); err != nil {
			return wasmHostReply{}, err
		}
		return wasmHostReply{}, run.ctx.Err()
	case <-run.gen.done:
		return wasmHostReply{}, run.gen.error()
	}
}

func (run *wasmRun) deliver(ctx context.Context, reply wasmHostReply) error {
	envelope := struct {
		Response    json.RawMessage `json:"response,omitempty"`
		ServedModel string          `json:"served_model,omitempty"`
		Error       *string         `json:"error,omitempty"`
	}{Response: reply.reply.Response, ServedModel: reply.reply.ServedModel}
	if reply.err != nil {
		envelope.Response = nil
		message := reply.err.Error()
		envelope.Error = &message
	}
	data, err := json.Marshal(envelope)
	if err != nil {
		return err
	}
	if err := run.respond(ctx, reply.id, data); err != nil {
		if reply.err != nil {
			return fmt.Errorf("host call failed: %w", reply.err)
		}
		return err
	}
	return nil
}

func (run *wasmRun) close() {
	run.cancel()
	slot := run.slot
	slot.mu.Lock()
	delete(run.gen.runs, run)
	if slot.gen == run.gen && !run.gen.poisoned {
		_, _ = run.gen.guest.call(context.Background(), "sy_run_cancel", uint64(run.ptr))
		_, _ = run.gen.guest.call(context.Background(), "sy_run_free", uint64(run.ptr))
	}
	slot.mu.Unlock()
	if slot.owner.active.Add(-1) == 0 && slot.owner.closed.Load() {
		slot.owner.finalizeClose()
	}
}
