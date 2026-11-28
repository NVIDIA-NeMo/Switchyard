// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"encoding/json"
	"errors"
)

// DecodeRequest converts a provider wire body to Switchyard's normalized request.
func DecodeRequest(format string, body json.RawMessage) (json.RawMessage, error) {
	return translated("sy_decode_request", struct {
		Format string          `json:"format"`
		Body   json.RawMessage `json:"body"`
	}{format, body})
}

// EncodeRequest stamps the target model and encodes the prepared normalized request.
func EncodeRequest(format string, request json.RawMessage, model string) (json.RawMessage, error) {
	return translated("sy_encode_request", struct {
		Format  string          `json:"format"`
		Request json.RawMessage `json:"request"`
		Model   string          `json:"model"`
	}{format, request, model})
}

// DecodeResponse converts a buffered provider response to Switchyard's normalized aggregate.
func DecodeResponse(format string, body json.RawMessage) (json.RawMessage, error) {
	return translated("sy_decode_response", struct {
		Format string          `json:"format"`
		Body   json.RawMessage `json:"body"`
	}{format, body})
}

// EncodeResponse converts a buffered normalized response to the client's wire format.
// requestExtensions is the decoded request's "extensions" field, or nil.
func EncodeResponse(format string, response, requestExtensions json.RawMessage) (json.RawMessage, error) {
	return translated("sy_encode_response", struct {
		Format            string          `json:"format"`
		Response          json.RawMessage `json:"response"`
		RequestExtensions json.RawMessage `json:"request_extensions,omitempty"`
	}{format, response, requestExtensions})
}

func translated(export string, input any) (json.RawMessage, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return translate(export, data)
}

// StreamTranslator converts one streamed provider response to the client's format.
// Value copies share one stream, and its zero value is closed. Do not call its methods
// concurrently, including through copies. Call Close when the stream ends.
type StreamTranslator struct {
	state *streamState
}

type streamState struct {
	handle  streamHandle
	sawDone bool
}

// streamHandle holds a backend's translation state for one stream.
type streamHandle interface {
	translate(event []byte) ([]byte, error)
	finish(sawDone bool) ([]byte, error)
	free()
}

var errStreamClosed = errors.New("Switchyard stream translator is closed")

var errEmptyEvent = errors.New("Switchyard stream event is empty")

// NewStreamTranslator starts translating a stream from the source format to the target format.
// requestExtensions is the decoded request's "extensions" field, or nil, as for EncodeResponse.
func NewStreamTranslator(source, target string, requestExtensions json.RawMessage) (*StreamTranslator, error) {
	input, err := json.Marshal(struct {
		Source            string          `json:"source"`
		Target            string          `json:"target"`
		RequestExtensions json.RawMessage `json:"request_extensions,omitempty"`
	}{source, target, requestExtensions})
	if err != nil {
		return nil, err
	}
	handle, err := newStreamHandle(input)
	if err != nil {
		return nil, err
	}
	return &StreamTranslator{state: &streamState{handle: handle}}, nil
}

// Translate converts the JSON data of one source event into zero or more target events.
func (stream *StreamTranslator) Translate(event json.RawMessage) ([]json.RawMessage, error) {
	handle := stream.handle()
	if handle == nil {
		return nil, errStreamClosed
	}
	// An empty event would drop the WASM call's event argument and fail the shared guest.
	if len(event) == 0 {
		return nil, errEmptyEvent
	}
	return streamEvents(handle.translate(event))
}

// DoneMarker records that the provider sent the SSE "[DONE]" marker. Call it instead of
// passing the marker to Translate. Finish then accepts the stream as complete if the
// source format treats the marker as final, as OpenAI Chat and Responses do.
func (stream *StreamTranslator) DoneMarker() {
	if stream.handle() != nil {
		stream.state.sawDone = true
	}
}

// Finish returns the target's closing events after the source stream ends. It fails
// when the source stopped before its final event or an accepted "[DONE]" marker.
func (stream *StreamTranslator) Finish() ([]json.RawMessage, error) {
	handle := stream.handle()
	if handle == nil {
		return nil, errStreamClosed
	}
	return streamEvents(handle.finish(stream.state.sawDone))
}

// Close releases the translator. Repeated calls and value copies are safe.
func (stream *StreamTranslator) Close() {
	if handle := stream.handle(); handle != nil {
		stream.state.handle = nil
		handle.free()
	}
}

// handle returns the stream's backend state, or nil once it is closed.
func (stream *StreamTranslator) handle() streamHandle {
	if stream == nil || stream.state == nil {
		return nil
	}
	return stream.state.handle
}

func streamEvents(data []byte, err error) ([]json.RawMessage, error) {
	if err != nil {
		return nil, err
	}
	var events []json.RawMessage
	err = json.Unmarshal(data, &events)
	return events, err
}
