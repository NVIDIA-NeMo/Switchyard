// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package switchyard runs embedded routing algorithms with host-owned model calls.
package switchyard

/*
#cgo CFLAGS: -I${SRCDIR}/../../crates/switchyard-ffi/include
#cgo LDFLAGS: ${SRCDIR}/../../target/release/libswitchyard_ffi.a
#cgo linux LDFLAGS: -ldl -lpthread -lm
#cgo darwin LDFLAGS: -framework Security -framework CoreFoundation -liconv
#include <stdlib.h>
#include "switchyard.h"
extern void syGoNext(uintptr_t user_data, char *error, char *step_json);
*/
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/cgo"
	"sync"
	"unsafe"
)

// Router shares an algorithm across concurrent requests and value copies. Its zero value is closed.
type Router struct {
	state *routerState
}

type routerState struct {
	mu  sync.RWMutex
	ptr *C.SyRouter
}

// Input supplies a normalized request and the host-approved model groups for this run.
type Input struct {
	Request        json.RawMessage     `json:"request"`
	Models         map[string][]string `json:"models"`
	SubagentModels map[string][]string `json:"subagent_models"`
	Metadata       *RequestMetadata    `json:"metadata,omitempty"`
}

// RequestMetadata carries routing context separately from the model request.
type RequestMetadata struct {
	SessionID       string `json:"session_id,omitempty"`
	AgentID         string `json:"agent_id,omitempty"`
	IsSubagent      bool   `json:"is_subagent,omitempty"`
	IsDelegatedWork bool   `json:"is_delegated_work,omitempty"`
	SessionFinal    *bool  `json:"session_final,omitempty"`
}

// Work requests either a model call (Models) or a typed decision call (Model).
type Work struct {
	Type    string          `json:"type"`
	ID      uint64          `json:"id"`
	Models  []string        `json:"models"`
	Model   string          `json:"model"`
	Request json.RawMessage `json:"request"`
}

// Reply contains a normalized response and, for model calls, the target that actually answered.
type Reply struct {
	Response    json.RawMessage `json:"response"`
	ServedModel string          `json:"served_model,omitempty"`
}

// HostCall executes model or decision work. It must honor ctx and may run concurrently.
// A call that ignores cancellation may outlive Route, but cannot access the native run.
type HostCall func(context.Context, Work) (Reply, error)

// Outcome retains the ordered candidates, prepared request, and any routing-produced answer.
type Outcome struct {
	SelectedModelIDs []string         `json:"selected_model_ids"`
	Request          json.RawMessage  `json:"request"`
	Response         *json.RawMessage `json:"response"`
	Metadata         json.RawMessage  `json:"metadata"`
}

// New constructs an algorithm from {"algorithm": AlgorithmSpec, "targets": {alias: modelID}}.
func New(config json.RawMessage) (*Router, error) {
	if version := C.sy_abi_version(); version != C.SY_ABI_VERSION {
		return nil, fmt.Errorf("Switchyard ABI version mismatch: library=%d, binding=%d", version, C.SY_ABI_VERSION)
	}
	value := C.CString(string(config))
	defer C.free(unsafe.Pointer(value))
	var ptr *C.SyRouter
	if err := nativeError(C.sy_router_new(value, &ptr)); err != nil {
		return nil, err
	}
	return &Router{state: &routerState{ptr: ptr}}, nil
}

// Close prevents new runs and releases the router. Active runs retain their algorithm.
// Repeated calls, including through Router value copies, are safe.
func (router *Router) Close() {
	state := router.state
	if state == nil {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.ptr != nil {
		C.sy_router_free(state.ptr)
		state.ptr = nil
	}
}

// Limit the router borrow to construction; each native run retains its own engine reference.
func (router *Router) newRun(input Input) (*C.SyRun, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	value := C.CString(string(data))
	defer C.free(unsafe.Pointer(value))
	state := router.state
	if state == nil {
		return nil, errors.New("Switchyard router is closed")
	}
	state.mu.RLock()
	defer state.mu.RUnlock()
	if state.ptr == nil {
		return nil, errors.New("Switchyard router is closed")
	}
	var run *C.SyRun
	err = nativeError(C.sy_run_new(state.ptr, value, &run))
	return run, err
}

type nextResult struct {
	data json.RawMessage
	err  error
}

type hostReply struct {
	id    uint64
	reply Reply
	err   error
}

// Route waits for an outcome while the host services calls with its own clients.
// Cancellation stops routing without waiting for host calls that ignore their context.
func (router *Router) Route(ctx context.Context, input Input, call HostCall) (Outcome, error) {
	if err := ctx.Err(); err != nil {
		return Outcome{}, err
	}
	run, err := router.newRun(input)
	if err != nil {
		return Outcome{}, err
	}
	ctx, cancel := context.WithCancel(ctx)
	// At most one read is outstanding; consume its result before requesting another.
	next := make(chan nextResult, 1)
	handle := cgo.NewHandle(next)
	defer func() {
		cancel()
		C.sy_run_cancel(run)
		C.sy_run_free(run)
		handle.Delete()
	}()
	readNext := func() error {
		return nativeError(C.sy_run_next_async(run, C.SyNextCallback(C.syGoNext), C.uintptr_t(handle)))
	}
	if err := readNext(); err != nil {
		return Outcome{}, err
	}
	replies := make(chan hostReply)
	for {
		select {
		case <-ctx.Done():
			return Outcome{}, ctx.Err()
		case result := <-next:
			if err := ctx.Err(); err != nil {
				return Outcome{}, err
			}
			if result.err != nil {
				return Outcome{}, result.err
			}
			var work Work
			if err := json.Unmarshal(result.data, &work); err != nil {
				return Outcome{}, err
			}
			if work.Type == "done" {
				var outcome Outcome
				err := json.Unmarshal(result.data, &outcome)
				if cancelErr := ctx.Err(); cancelErr != nil {
					return Outcome{}, cancelErr
				}
				return outcome, err
			}
			if work.Type != "call_model" && work.Type != "call_decision" {
				return Outcome{}, errors.New("unknown Switchyard step: " + work.Type)
			}
			if call == nil {
				return Outcome{}, errors.New("Switchyard algorithm requires a host model client")
			}
			// Workers own Go data only; late results cannot call into a freed native run.
			go func(work Work) {
				if ctx.Err() != nil {
					return
				}
				reply, err := call(ctx, work)
				select {
				case replies <- hostReply{work.ID, reply, err}:
				case <-ctx.Done():
				}
			}(work)
			if err := readNext(); err != nil {
				return Outcome{}, err
			}
		case reply := <-replies:
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
				return Outcome{}, err
			}
			if err := ctx.Err(); err != nil {
				return Outcome{}, err
			}
			value := C.CString(string(data))
			err = nativeError(C.sy_run_respond(run, C.uint64_t(reply.id), value))
			C.free(unsafe.Pointer(value))
			if err != nil {
				return Outcome{}, err
			}
		}
	}
}

func nativeError(value *C.char) error {
	if value == nil {
		return nil
	}
	defer C.sy_string_free(value)
	return errors.New(C.GoString(value))
}
