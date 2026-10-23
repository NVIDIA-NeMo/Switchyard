// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package switchyard

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func testConfig(algorithm string) json.RawMessage {
	return json.RawMessage(`{"algorithm":` + algorithm + `,"targets":{"weak":"weak-model","strong":"strong-model","judge":"judge-model"}}`)
}

// Message-key fallback retains a verdict, including children without an agent ID.
func TestMessageRetention(t *testing.T) {
	algorithm := strings.Replace(composite, `"classify_trigger":"user_turn"`, `"classify_trigger":"new_session","message_hash_fallback":true`, 1)
	for _, child := range []bool{false, true} {
		router := newRouter(t, algorithm)
		request := input()
		if child {
			request.Metadata = &RequestMetadata{SessionID: "parent", IsSubagent: true}
		}
		calls := 0
		for range 8 {
			_, err := router.Route(context.Background(), request, func(context.Context, Work) (Reply, error) {
				calls++
				return Reply{Response: json.RawMessage(judgeAnswer)}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
		if calls != 1 {
			t.Errorf("subagent=%t made %d judge calls, want 1", child, calls)
		}
	}
}

// A routing deadline must expire while its host call is still pending.
func TestRoutingDeadline(t *testing.T) {
	router := newRouter(t, `{"type":"vgr","local_target":"weak","cloud_target":"strong","mode":"evaluate","deadline_seconds":0.02,"task_typing":false}`)
	for range 4 {
		t.Run("request", func(t *testing.T) {
			t.Parallel()
			testRoutingDeadline(t, router)
		})
	}
}

func testRoutingDeadline(t *testing.T, router *Router) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	hostFinished := make(chan struct{})
	started := time.Now()
	outcome, err := router.Route(ctx, input(), func(ctx context.Context, _ Work) (Reply, error) {
		defer close(hostFinished)
		delay := time.NewTimer(400 * time.Millisecond)
		defer delay.Stop()
		select {
		case <-delay.C:
			return Reply{Response: json.RawMessage(judgeAnswer)}, nil
		case <-ctx.Done():
			return Reply{}, ctx.Err()
		}
	})
	elapsed := time.Since(started)
	t.Logf("routing deadline=20ms, host delay=400ms, route duration=%v", elapsed)
	select {
	case <-hostFinished:
	case <-time.After(time.Second):
		t.Fatal("host callback did not stop after routing finished")
	}
	if err != nil || len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != "strong-model" {
		t.Fatalf("outcome=%+v err=%v", outcome, err)
	}
	if elapsed >= 200*time.Millisecond {
		t.Fatal("routing waited for the host instead of its configured deadline")
	}
}

// Host calls must overlap without blocking other requests.
func TestConcurrentHostCalls(t *testing.T) {
	router := newRouter(t, composite)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	results := make(chan error, 2)
	for range 2 {
		go func() {
			outcome, err := router.Route(ctx, input(), func(ctx context.Context, _ Work) (Reply, error) {
				started <- struct{}{}
				select {
				case <-release:
					return Reply{Response: json.RawMessage(judgeAnswer)}, nil
				case <-ctx.Done():
					return Reply{}, ctx.Err()
				}
			})
			if err == nil && (len(outcome.SelectedModelIDs) == 0 || outcome.SelectedModelIDs[0] != "weak-model") {
				err = fmt.Errorf("unexpected outcome: %+v", outcome)
			}
			results <- err
		}()
	}
	for range 2 {
		select {
		case <-started:
		case <-ctx.Done():
			t.Fatal("host calls did not overlap")
		}
	}
	close(release)
	for range 2 {
		select {
		case err := <-results:
			if err != nil {
				t.Fatal(err)
			}
		case <-ctx.Done():
			t.Fatal("concurrent routes did not finish")
		}
	}
}
