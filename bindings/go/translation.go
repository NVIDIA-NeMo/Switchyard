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
// Do not call its methods concurrently. Call Close when the stream ends.
type StreamTranslator struct {
	handle streamHandle
}

// streamHandle holds a backend's translation state for one stream.
type streamHandle interface {
	translate(event []byte) ([]byte, error)
	finish() ([]byte, error)
	free()
}

var errStreamClosed = errors.New("Switchyard stream translator is closed")

// NewStreamTranslator starts translating a stream from the source format to the target format.
func NewStreamTranslator(source, target string) (*StreamTranslator, error) {
	input, err := json.Marshal(struct {
		Source string `json:"source"`
		Target string `json:"target"`
	}{source, target})
	if err != nil {
		return nil, err
	}
	handle, err := newStreamHandle(input)
	if err != nil {
		return nil, err
	}
	return &StreamTranslator{handle: handle}, nil
}

// Translate converts the JSON data of one source event into zero or more target events.
func (stream *StreamTranslator) Translate(event json.RawMessage) ([]json.RawMessage, error) {
	if stream.handle == nil {
		return nil, errStreamClosed
	}
	return streamEvents(stream.handle.translate(event))
}

// Finish returns the target's closing events after the source stream ends.
func (stream *StreamTranslator) Finish() ([]json.RawMessage, error) {
	if stream.handle == nil {
		return nil, errStreamClosed
	}
	return streamEvents(stream.handle.finish())
}

// Close releases the translator. Repeated calls are safe.
func (stream *StreamTranslator) Close() {
	if stream.handle != nil {
		stream.handle.free()
		stream.handle = nil
	}
}

func streamEvents(data []byte, err error) ([]json.RawMessage, error) {
	if err != nil {
		return nil, err
	}
	var events []json.RawMessage
	err = json.Unmarshal(data, &events)
	return events, err
}
