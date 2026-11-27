// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"context"
	"encoding/json"
	"errors"
)

func translate(export string, input []byte) (json.RawMessage, error) {
	return wasmTranslate(export, input)
}

// wasmStream lives in one guest. It stops working if a trap replaces that guest.
type wasmStream struct {
	translator *wasmTranslator
	guest      *guestModule
	ptr        uint32
}

var errStreamLost = errors.New("Switchyard stream translation state was lost after a wasm failure")

func newStreamHandle(input []byte) (streamHandle, error) {
	translator, err := sharedWasmTranslator()
	if err != nil {
		return nil, err
	}
	stream := &wasmStream{translator: translator}
	err = translator.use(func(guest *guestModule) error {
		ptr, err := guest.callOut(context.Background(), "sy_stream_new", input)
		stream.guest, stream.ptr = guest, ptr
		return err
	})
	if err != nil {
		return nil, err
	}
	return stream, nil
}

func (stream *wasmStream) translate(event []byte) ([]byte, error) {
	return stream.call("sy_stream_translate", event, uint64(stream.ptr))
}

func (stream *wasmStream) finish(sawDone bool) ([]byte, error) {
	var done uint64
	if sawDone {
		done = 1
	}
	return stream.call("sy_stream_finish", nil, uint64(stream.ptr), done)
}

// call runs export on the stream's guest. args precede input, matching the C signature.
func (stream *wasmStream) call(export string, input []byte, args ...uint64) ([]byte, error) {
	var output []byte
	err := stream.translator.use(func(guest *guestModule) (err error) {
		if guest != stream.guest {
			return errStreamLost
		}
		output, err = guest.callJSON(context.Background(), export, input, args...)
		return err
	})
	return output, err
}

// free locks directly because use would start a guest only to find the stream gone.
func (stream *wasmStream) free() {
	stream.translator.mu.Lock()
	defer stream.translator.mu.Unlock()
	if stream.translator.guest == stream.guest {
		_, _ = stream.guest.call(context.Background(), "sy_stream_free", uint64(stream.ptr))
	}
}
