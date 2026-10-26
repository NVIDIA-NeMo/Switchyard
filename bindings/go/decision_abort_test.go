// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"errors"
	"io/fs"
	"strings"
	"testing"
)

// Both judge protocols retain the host's original refusal through the wire facade.
func TestDecideHostAbort(t *testing.T) {
	refused := &fs.PathError{Op: "route", Path: "judge", Err: errors.New("budget exhausted")}
	for _, algorithm := range []string{composite, decision} {
		router := newRouter(t, strings.Replace(algorithm, `"fail_open":false`, `"fail_open":true`, 1))
		calls := 0
		var hostCtx context.Context
		_, err := router.Decide(context.Background(), decisionInput("openai_chat", wireChat), func(ctx context.Context, _ WireWork) (Reply, error) {
			calls++
			hostCtx = ctx
			return Reply{}, Abort(refused)
		})
		if err != refused || calls != 1 {
			t.Fatalf("calls=%d refusal=%v", calls, err)
		}
		if hostCtx.Err() != context.Canceled {
			t.Fatal("refusal did not cancel the host context")
		}
	}
}
