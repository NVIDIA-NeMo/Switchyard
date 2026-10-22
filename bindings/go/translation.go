// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"encoding/json"
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

func translated(export string, input any) (json.RawMessage, error) {
	data, err := json.Marshal(input)
	if err != nil {
		return nil, err
	}
	return translate(export, data)
}
