// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build cgo

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

type nativeBackend struct {
	mu  sync.RWMutex
	ptr *C.SyRouter
}

func newBackend(config json.RawMessage) (engineBackend, error) {
	if version := C.sy_abi_version(); version != C.SY_ABI_VERSION {
		return nil, fmt.Errorf("Switchyard ABI version mismatch: library=%d, binding=%d", version, C.SY_ABI_VERSION)
	}
	value := C.CString(string(config))
	defer C.free(unsafe.Pointer(value))
	var ptr *C.SyRouter
	if err := nativeError(C.sy_router_new(value, &ptr)); err != nil {
		return nil, err
	}
	return &nativeBackend{ptr: ptr}, nil
}

func (backend *nativeBackend) close() {
	backend.mu.Lock()
	defer backend.mu.Unlock()
	if backend.ptr != nil {
		C.sy_router_free(backend.ptr)
		backend.ptr = nil
	}
}

func (backend *nativeBackend) newRun(input runInput) (*C.SyRun, error) {
	data, err := input.marshal()
	if err != nil {
		return nil, err
	}
	value := C.CString(string(data))
	defer C.free(unsafe.Pointer(value))
	backend.mu.RLock()
	defer backend.mu.RUnlock()
	if backend.ptr == nil {
		return nil, errors.New("Switchyard router is closed")
	}
	var run *C.SyRun
	if input.wire != nil {
		err = nativeError(C.sy_decide_new(backend.ptr, value, &run))
	} else {
		err = nativeError(C.sy_run_new(backend.ptr, value, &run))
	}
	return run, err
}

type nativeNextResult struct {
	data json.RawMessage
	err  error
}

type nativeHostReply struct {
	id    uint64
	reply Reply
	err   error
}

func (backend *nativeBackend) route(ctx context.Context, input runInput, call func(context.Context, runWork) (Reply, error)) (runOutcome, error) {
	if err := ctx.Err(); err != nil {
		return runOutcome{}, err
	}
	run, err := backend.newRun(input)
	if err != nil {
		return runOutcome{}, err
	}
	ctx, cancel := context.WithCancel(ctx)
	next := make(chan nativeNextResult, 1)
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
		return runOutcome{}, err
	}
	replies := make(chan nativeHostReply)
	for {
		select {
		case <-ctx.Done():
			return runOutcome{}, ctx.Err()
		case result := <-next:
			if err := ctx.Err(); err != nil {
				return runOutcome{}, err
			}
			if result.err != nil {
				return runOutcome{}, result.err
			}
			var work runWork
			if err := json.Unmarshal(result.data, &work); err != nil {
				return runOutcome{}, err
			}
			if work.Type == "done" {
				var outcome runOutcome
				if err := json.Unmarshal(result.data, &outcome); err != nil {
					return runOutcome{}, err
				}
				if err := ctx.Err(); err != nil {
					return runOutcome{}, err
				}
				return outcome, nil
			}
			if work.Type != "call_model" && work.Type != "call_decision" {
				return runOutcome{}, errors.New("unknown Switchyard step: " + work.Type)
			}
			if call == nil {
				return runOutcome{}, errors.New("Switchyard algorithm requires a host model client")
			}
			go func(work runWork) {
				if ctx.Err() != nil {
					return
				}
				reply, err := call(ctx, work)
				select {
				case replies <- nativeHostReply{work.ID, reply, err}:
				case <-ctx.Done():
				}
			}(work)
			if err := readNext(); err != nil {
				return runOutcome{}, err
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
				return runOutcome{}, err
			}
			if err := ctx.Err(); err != nil {
				return runOutcome{}, err
			}
			value := C.CString(string(data))
			err = nativeError(C.sy_run_respond(run, C.uint64_t(reply.id), value))
			C.free(unsafe.Pointer(value))
			if err != nil {
				return runOutcome{}, err
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
