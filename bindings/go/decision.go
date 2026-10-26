// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
)

// WireInput supplies provider JSON in Input.Request, plus the usual routing context.
// TargetFormats overrides Format for individual target IDs. Keep its maps unchanged until Decide returns.
type WireInput struct {
	Input
	Format        string            `json:"format"`
	TargetFormats map[string]string `json:"target_formats,omitempty"`
}

// PreparedRequest is a provider-shaped request stamped for one target.
type PreparedRequest struct {
	Model   string          `json:"model"`
	Format  string          `json:"format"`
	Request json.RawMessage `json:"request"`
}

// WireWork carries ordered provider requests for call_model. For call_decision,
// Work.Model and Work.Request retain the typed decision protocol and Requests is empty.
type WireWork struct {
	Work
	Requests []PreparedRequest `json:"requests"`
}

// WireHostCall executes routing work with host-owned auth, transport, retries, and fallbacks.
// Return a buffered wire response for call_model, identifying ServedModel for multiple candidates.
// Typed decision replies use their own protocol. Host errors follow the Route contract;
// invalid provider replies stop routing. Calls may overlap and must honor ctx.
type WireHostCall func(context.Context, WireWork) (Reply, error)

// Decision retains the complete normalized Outcome alongside ordered provider requests.
// An algorithm-produced Response remains normalized; it is not encoded to a provider format.
type Decision struct {
	Outcome
	Requests []PreparedRequest `json:"requests"`
}

// Decide converts provider JSON and completes routing without serving a post-routing answer.
// The host executes routing-time calls through call; each reply must match its target's format.
func (router *Router) Decide(ctx context.Context, input WireInput, call WireHostCall) (Decision, error) {
	var host func(context.Context, runWork) (Reply, error)
	if call != nil {
		host = func(ctx context.Context, work runWork) (Reply, error) {
			return call(ctx, WireWork{Work: work.Work, Requests: work.Requests})
		}
	}
	outcome, err := router.route(ctx, runInput{Input: input.Input, wire: &input}, host)
	return Decision{Outcome: outcome.Outcome, Requests: outcome.Requests}, err
}
