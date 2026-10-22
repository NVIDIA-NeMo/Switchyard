// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the C bridge.

use super::*;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

type Completion = std::result::Result<CString, CString>;

unsafe extern "C" fn complete(user_data: usize, error: *mut c_char, output: *mut c_char) {
    let sender = unsafe { &*(user_data as *const mpsc::Sender<Completion>) };
    let output = (!output.is_null()).then(|| unsafe { CString::from_raw(output) });
    let result = if error.is_null() {
        Ok(output.unwrap_or_default())
    } else {
        Err(unsafe { CString::from_raw(error) })
    };
    let _ = sender.send(result);
}

unsafe extern "C" fn complete_once(user_data: usize, error: *mut c_char, output: *mut c_char) {
    let sender = unsafe { Box::from_raw(user_data as *mut mpsc::Sender<Completion>) };
    unsafe { complete(&*sender as *const _ as usize, error, output) };
}

fn next_async(run: &SyRun, sender: &mpsc::Sender<Completion>) -> std::result::Result<(), String> {
    let sender = Box::into_raw(Box::new(sender.clone()));
    let error = unsafe { sy_run_next_async(run, Some(complete_once), sender as usize) };
    if error.is_null() {
        Ok(())
    } else {
        drop(unsafe { Box::from_raw(sender) });
        Err(error_message(error))
    }
}

fn received_step(receiver: &mpsc::Receiver<Completion>) -> Value {
    let output = receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    serde_json::from_slice(output.as_bytes()).unwrap()
}

// ABI generations are independent of package releases.
#[test]
fn abi_version_identifies_v1_contract() {
    assert_eq!(sy_abi_version(), 1);
}

fn check(error: *mut c_char) {
    if !error.is_null() {
        panic!("{}", unsafe { CString::from_raw(error) }.to_string_lossy());
    }
}

fn error_message(error: *mut c_char) -> String {
    assert!(!error.is_null(), "expected an error");
    unsafe { CString::from_raw(error) }.into_string().unwrap()
}

fn router(algorithm: Value) -> Box<SyRouter> {
    let config = json!({
        "algorithm": algorithm,
        "targets": {"capable": "capable", "efficient": "efficient", "judge": "judge"},
    });
    let config = CString::new(config.to_string()).unwrap();
    let mut router = std::ptr::null_mut();
    check(unsafe { sy_router_new(config.as_ptr(), &mut router) });
    assert!(!router.is_null());
    unsafe { Box::from_raw(router) }
}

fn run(router: &SyRouter, input: Value) -> Box<SyRun> {
    let input = CString::new(input.to_string()).unwrap();
    let mut run = std::ptr::null_mut();
    check(unsafe { sy_run_new(router, input.as_ptr(), &mut run) });
    assert!(!run.is_null());
    unsafe { Box::from_raw(run) }
}

fn next(run: &SyRun) -> Value {
    let (sender, receiver) = mpsc::channel();
    next_async(run, &sender).unwrap();
    received_step(&receiver)
}

fn translated(
    call: unsafe extern "C" fn(*const c_char, *mut *mut c_char) -> *mut c_char,
    input: Value,
) -> Value {
    let input = CString::new(input.to_string()).unwrap();
    let mut output = std::ptr::null_mut();
    check(unsafe { call(input.as_ptr(), &mut output) });
    assert!(!output.is_null());
    let result = serde_json::from_slice(unsafe { CStr::from_ptr(output) }.to_bytes());
    unsafe { sy_string_free(output) };
    result.unwrap()
}

fn advisor_router() -> Box<SyRouter> {
    router(json!({
        "type": "advisor", "executor_target": "capable", "advisor_target": "judge",
        "gate_trigger": "pattern", "gate_trigger_pattern": "NEVER",
    }))
}

fn advisor_input() -> Value {
    json!({"request": {}, "models": {"efficient": ["capable"], "judge": ["judge"]}})
}

fn pending_advisor() -> (Box<SyRun>, u64) {
    let router = advisor_router();
    let run = run(&router, advisor_input());
    let call = next(&run);
    assert_eq!(call["type"], "call_model");
    (run, call["id"].as_u64().unwrap())
}

#[test]
fn completed_runs_reject_further_reads() {
    let router = router(json!({"type": "passthrough", "target": "capable"}));
    let run = run(
        &router,
        json!({"request": {}, "models": {"any": ["capable"]}}),
    );
    unsafe { sy_router_free(Box::into_raw(router)) };
    assert_eq!(next(&run)["type"], "done");
    let (sender, receiver) = mpsc::channel();
    for _ in 0..2 {
        assert_eq!(
            next_async(&run, &sender).unwrap_err(),
            "routing already completed"
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));
    }
    unsafe { sy_run_free(Box::into_raw(run)) };
}

#[test]
fn cancellation_completes_pending_read_and_rejects_replies() {
    let (run, id) = pending_advisor();
    let (sender, receiver) = mpsc::channel();
    next_async(&run, &sender).unwrap();
    assert_eq!(
        next_async(&run, &sender).unwrap_err(),
        "routing read already in progress"
    );
    unsafe { sy_run_cancel(&*run) };
    let error = receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_str().unwrap(), "routing cancelled");
    let error = unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) };
    assert_eq!(error_message(error), "routing cancelled");
}

// A malformed reply must not consume its call ID.
#[test]
fn invalid_model_replies_preserve_pending_calls() {
    let (run, id) = pending_advisor();
    let error = unsafe {
        sy_run_respond(
            &*run,
            id,
            c"{\"response\":{\"outputs\":\"wrong\"}}".as_ptr(),
        )
    };
    assert!(!error_message(error).is_empty());
    check(unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) });
    let error = unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) };
    assert_eq!(error_message(error), "unknown call ID");
    assert_eq!(next(&run)["type"], "done");
}

// Decision replies have a different schema, but the same retryable validation boundary.
#[test]
fn invalid_decision_replies_preserve_pending_calls() {
    let router = router(json!({
        "type": "llm_classifier", "classifier_target": "judge",
        "strong_target": "capable", "weak_target": "efficient", "fail_open": false,
        "decision": {"cutoff": 0.5, "candidates": {"a": "capable", "b": "efficient"}, "evidence": {}},
    }));
    let run = run(
        &router,
        json!({
            "request": {},
            "models": {
                "any": ["capable", "efficient"], "capable": ["capable"],
                "efficient": ["efficient"], "judge": ["judge"],
            },
        }),
    );
    let call = next(&run);
    assert_eq!(call["type"], "call_decision");
    assert_eq!(call["model"], "judge");
    let id = call["id"].as_u64().unwrap();
    let error = unsafe { sy_run_respond(&*run, id, c"{\"response\":{}}".as_ptr()) };
    assert!(!error_message(error).is_empty());
    let reply = CString::new(json!({"response": {"answers": {"route": {"value": {
        "type": "choice", "data": {"selected": "advantage", "probabilities": {"advantage": 0.8, "no_advantage": 0.2}},
    }}}}}).to_string()).unwrap();
    check(unsafe { sy_run_respond(&*run, id, reply.as_ptr()) });
    let done = next(&run);
    assert_eq!(done["type"], "done");
    assert_eq!(done["selected_model_ids"], json!(["capable", "efficient"]));
}

// Translation exports return owned JSON strings that the caller releases through the ABI.
#[test]
fn translation_exports_return_owned_json() {
    let request = translated(
        sy_decode_request,
        json!({
            "format": "openai_chat",
            "body": {"model": "route", "messages": [{"role": "user", "content": "hello"}]},
        }),
    );
    assert_eq!(request["messages"][0]["content"][0]["text"], "hello");
    let encoded = translated(
        sy_encode_request,
        json!({
            "format": "openai_chat", "request": request, "model": "selected",
        }),
    );
    assert_eq!(encoded["model"], "selected");
    let response = translated(
        sy_decode_response,
        json!({
            "format": "openai_chat",
            "body": {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]},
        }),
    );
    assert_eq!(response["outputs"][0]["content"][0]["text"], "hello");
}

// Pending reads must yield the runtime worker, not consume blocking threads.
#[test]
fn pending_reads_yield_runtime_worker() {
    let thread_starts = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&thread_starts);
    let mut router = advisor_router();
    Arc::get_mut(&mut router.engine).unwrap().runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .on_thread_start(move || {
            observed.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    let (sender, receiver) = mpsc::channel();
    let mut runs = Vec::new();
    for _ in 0..32 {
        let run = run(&router, advisor_input());
        next_async(&run, &sender).unwrap();
        assert_eq!(received_step(&receiver)["type"], "call_model");
        next_async(&run, &sender).unwrap();
        runs.push(run);
    }
    let (scheduled, sentinel) = mpsc::channel();
    router.engine.runtime.spawn(async move {
        let _ = scheduled.send(());
    });
    sentinel.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(thread_starts.load(Ordering::SeqCst), 1);
    drop(runs);
    for _ in 0..32 {
        let error = receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap_err();
        assert_eq!(error.to_str().unwrap(), "routing cancelled");
    }
}

struct BlockedCompletion {
    sender: mpsc::Sender<Completion>,
    resume: Mutex<mpsc::Receiver<()>>,
    finished: mpsc::Sender<()>,
}

unsafe extern "C" fn complete_after_release(
    user_data: usize,
    error: *mut c_char,
    output: *mut c_char,
) {
    let state = unsafe { &*(user_data as *const BlockedCompletion) };
    unsafe { complete(&state.sender as *const _ as usize, error, output) };
    let _ = state.resume.lock().recv();
    let _ = state.finished.send(());
}

// Keep another runtime owner so runtime shutdown cannot mask an early free.
#[test]
fn free_waits_for_cancelled_callback_to_return() {
    let engine = Arc::new(Engine {
        algorithm: Arc::new(switchyard_libsy::Noop {}),
        runtime: tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap(),
    });
    let (sender, receiver) = mpsc::channel();
    let (resume, paused) = mpsc::channel();
    let (finished, returned) = mpsc::channel();
    let state = Box::into_raw(Box::new(BlockedCompletion {
        sender,
        resume: Mutex::new(paused),
        finished,
    }));
    let run = Box::new(SyRun::new(
        Box::pin(futures::stream::pending()),
        Arc::clone(&engine),
    ));
    check(unsafe { sy_run_next_async(&*run, Some(complete_after_release), state as usize) });
    let run = Box::into_raw(run) as usize;
    let (freed, closed) = mpsc::channel();
    let freeing = std::thread::spawn(move || {
        unsafe { sy_run_free(run as *mut SyRun) };
        let _ = freed.send(());
    });
    let cancellation = receiver.recv_timeout(Duration::from_secs(5));
    let premature_free = closed.recv_timeout(Duration::from_millis(50));
    let _ = resume.send(());
    returned.recv_timeout(Duration::from_secs(5)).unwrap();
    if premature_free.is_err() {
        closed.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    freeing.join().unwrap();
    drop(unsafe { Box::from_raw(state) });
    assert!(matches!(
        premature_free,
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(
        cancellation.unwrap().unwrap_err().to_str().unwrap(),
        "routing cancelled"
    );
}

// Runs may share call IDs without sharing replies or cancellation.
#[test]
fn async_runs_complete_independently() {
    let router = advisor_router();
    let (sender, receiver) = mpsc::channel();
    let runs: Vec<_> = (0..3).map(|_| run(&router, advisor_input())).collect();
    let mut ids = Vec::new();
    for run in &runs {
        next_async(run, &sender).unwrap();
        ids.push(received_step(&receiver)["id"].as_u64().unwrap());
        next_async(run, &sender).unwrap();
    }
    for (index, text) in [(1, "second"), (0, "first")] {
        if index == 0 {
            unsafe { sy_run_cancel(&*runs[2]) };
            let error = receiver
                .recv_timeout(Duration::from_secs(5))
                .unwrap()
                .unwrap_err();
            assert_eq!(error.to_str().unwrap(), "routing cancelled");
        }
        let reply = CString::new(
            json!({
                "response": switchyard_protocol::text_response(None, text),
            })
            .to_string(),
        )
        .unwrap();
        check(unsafe { sy_run_respond(&*runs[index], ids[index], reply.as_ptr()) });
        let done = received_step(&receiver);
        assert_eq!(done["type"], "done");
        assert_eq!(done["response"]["outputs"][0]["content"][0]["text"], text);
    }
}

// Host fallback attribution uses target IDs, not provider model names.
#[test]
fn advisor_attributes_host_fallback() {
    let router = advisor_router();
    for served_model in [None, Some("fallback")] {
        let input = json!({
            "request": {}, "models": {"efficient": ["capable", "fallback"], "judge": ["judge"]},
        });
        let run = run(&router, input);
        let call = next(&run);
        assert_eq!(call["models"], json!(["capable", "fallback"]));
        let mut reply = json!({
            "response": switchyard_protocol::text_response(Some("provider-model".into()), "answer"),
        });
        if let Some(model) = served_model {
            reply["served_model"] = json!(model);
        }
        let reply = CString::new(reply.to_string()).unwrap();
        check(unsafe { sy_run_respond(&*run, call["id"].as_u64().unwrap(), reply.as_ptr()) });
        let done = next(&run);
        let expected = served_model.unwrap_or("capable");
        assert_eq!(done["type"], "done");
        assert_eq!(done["selected_model_ids"], json!([expected]));
        assert_eq!(done["request"]["model"], expected);
        assert_eq!(done["response"]["model"], "provider-model");
    }
}

// Child sessions keep separate phases and clear them after the final request.
#[test]
fn plan_execute_isolates_and_finishes_child_sessions() {
    let router = router(json!({
        "type": "plan_execute", "capable_target": "capable", "efficient_target": "efficient",
    }));
    for (agent, is_subagent, is_final, edited, expected) in [
        ("a", true, false, true, "efficient"),
        ("b", true, false, false, "capable"),
        ("root", false, false, false, "capable"),
        ("a", true, true, false, "efficient"),
        ("a", true, false, false, "capable"),
    ] {
        let request = if edited {
            json!({"messages": [{"role": "assistant", "content": [{
                "type": "tool_call", "id": "edit", "name": "apply_patch", "arguments": {},
            }]}]})
        } else {
            json!({})
        };
        let input = json!({
            "request": request, "models": {"capable": ["capable"], "efficient": ["efficient"]},
            "metadata": {"session_id": "s", "agent_id": agent, "is_subagent": is_subagent,
                "session_final": is_final},
        });
        let run = run(&router, input);
        assert_eq!(
            next(&run)["selected_model_ids"],
            json!([expected]),
            "{agent}"
        );
    }
}

// Child identity alone must not activate the delegated-work policy.
#[test]
fn delegated_work_uses_subagent_model_groups() {
    let router = router(json!({
        "type": "passthrough", "target": "capable",
        "subagents": {"type": "passthrough", "target": "efficient"},
    }));
    for (metadata, expected) in [
        (Value::Null, "capable"),
        (json!({"is_subagent": true}), "capable"),
        (
            json!({"is_subagent": true, "is_delegated_work": true}),
            "efficient",
        ),
    ] {
        let input = json!({
            "request": {}, "models": {"any": ["capable"]},
            "subagent_models": {"any": ["efficient"]}, "metadata": metadata,
        });
        let run = run(&router, input);
        assert_eq!(next(&run)["selected_model_ids"], json!([expected]));
    }
}

#[test]
fn invalid_router_input_returns_owned_errors() {
    for input in [std::ptr::null(), c"{".as_ptr()] {
        let mut router = std::ptr::dangling_mut::<SyRouter>();
        let error = unsafe { sy_router_new(input, &mut router) };
        assert!(!error.is_null());
        assert!(router.is_null());
        let message = unsafe { CStr::from_ptr(error) }.to_str().unwrap();
        assert!(!message.is_empty());
        unsafe { sy_string_free(error) };
    }
    let error = unsafe { sy_router_new(c"{".as_ptr(), std::ptr::null_mut()) };
    assert!(!error.is_null());
    unsafe { sy_string_free(error) };
}

// Unwinding panics become owned errors at the C boundary.
#[test]
fn panics_return_owned_errors() {
    let run = SyRun::new(
        Box::pin(futures::stream::poll_fn(|_| panic!("test panic"))),
        Arc::new(Engine {
            algorithm: Arc::new(switchyard_libsy::Noop {}),
            runtime: Runtime::new().unwrap(),
        }),
    );
    let (sender, receiver) = mpsc::channel();
    next_async(&run, &sender).unwrap();
    let error = receiver
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.to_str().unwrap(), "Switchyard binding panicked");
}
