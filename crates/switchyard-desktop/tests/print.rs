// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// An unreadable log must fail instead of reporting that no requests were recorded.
#[test]
fn unreadable_usage_fails_without_an_empty_summary() {
    let home = tempfile::tempdir().expect("fixture");
    let settings = home.path().join("desktop.toml");
    std::fs::write(&settings, format!("routing_log={:?}\n", home.path())).expect("settings");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
        .arg("--print")
        .arg(&settings)
        .env("HOME", home.path())
        .output()
        .expect("print");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Usage unavailable"));
    assert_eq!(std::fs::read_dir(home.path()).expect("files").count(), 1);
}

#[test]
fn print_uses_bundled_prices_and_requires_valid_explicit_overrides()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir().expect("fixture");
    let settings = dir.path().join("desktop.toml");
    let log = dir.path().join("routing.jsonl");
    // Both token counts are positive so the fixture qualifies for a baseline estimate.
    let record = serde_json::json!({
        "ts": chrono::Local::now().to_rfc3339(),
        "model": "gpt-6-sol", "prompt_tokens": 1_000_000, "completion_tokens": 100_000,
        "cost": {"usd": 0.1, "source": "provider_reported"}
    });
    std::fs::write(&log, format!("{record}\n")).expect("log");
    let base = format!("server_url='http://127.0.0.1:1'\nrouting_log={log:?}\n");
    std::fs::write(&settings, &base).expect("settings");
    let output = print(&settings);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Saved $2.90 (97% of $3.00)"));
    assert!(!dir.path().join("custom-prices.toml").exists());
    // A different bundled baseline changes the estimate; the saved $0.10 charge stays the same.
    std::fs::write(
        &settings,
        format!("{base}baseline_model='claude-opus-5-5'\n"),
    )?;
    let output = print(&settings);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("Saved $5.90 (98% of $6.00)"));
    for (name, text) in [
        (
            "custom-prices.json",
            r#"{"gpt-6-sol":{"input":9,"output":0,"cache_read":0,"cache_write":0}}"#,
        ),
        (
            "custom-prices.toml",
            "[\"gpt-6-sol\"]\ninput=9\noutput=0\ncache_read=0\ncache_write=0\n",
        ),
    ] {
        let prices = dir.path().join(name);
        std::fs::write(&settings, format!("{base}pricing_file={prices:?}\n"))
            .expect("override settings");
        for invalid in [None, Some("not json")] {
            if let Some(text) = invalid {
                std::fs::write(&prices, text).expect("invalid file");
            }
            let output = print(&settings);
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            assert!(String::from_utf8_lossy(&output.stderr).contains("pricing file"));
        }
        std::fs::write(&prices, text).expect("override");
        let output = print(&settings);
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("Saved $8.90 (99% of $9.00)"));
        // The omitted model distinguishes table replacement from merging bundled rates.
        std::fs::write(
            &settings,
            format!("{base}pricing_file={prices:?}\nbaseline_model='gpt-6-luna'\n"),
        )
        .expect("missing baseline");
        let output = print(&settings);
        assert!(output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Saved"));
    }
    assert_eq!(
        std::fs::read_to_string(&log).expect("unchanged log"),
        format!("{record}\n")
    );
    Ok(())
}

fn print(settings: &std::path::Path) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
        .arg("--print")
        .arg(settings)
        .output()
        .expect("print")
}

fn record() -> serde_json::Value {
    serde_json::json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "model": "actual",
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "cost": {"usd": 1, "source": "provider_reported"}
    })
}

#[test]
fn recorded_spending_stays_visible_when_savings_are_unavailable() {
    let dir = tempfile::tempdir().expect("fixture");
    let settings = dir.path().join("desktop.toml");
    let log = dir.path().join("routing.jsonl");
    let prices = dir.path().join("prices.json");
    std::fs::write(
        &settings,
        format!("server_url='http://127.0.0.1:0'\nbaseline_model='baseline'\nrouting_log={:?}\npricing_file={:?}\n", log, prices),
    )
    .expect("settings");
    let valid = record();
    let mut unknown = record();
    unknown.as_object_mut().expect("record").remove("cost");
    let mut uncounted = record();
    uncounted["prompt_tokens"] = 0.into();
    for (entries, rates, spending) in [
        (
            vec![valid.clone(), unknown],
            true,
            "Known spending $1.00 · total unknown (1 call has no recorded cost)",
        ),
        (vec![valid], false, "Spent $1.00"),
        (vec![uncounted], true, "Spent $1.00"),
    ] {
        let text = entries
            .iter()
            .map(|entry| format!("{entry}\n"))
            .collect::<String>();
        std::fs::write(&log, text).expect("log");
        std::fs::write(
            &prices,
            if rates {
                r#"{"baseline":{"input":10,"output":10,"cache_read":0,"cache_write":0}}"#
            } else {
                "{}"
            },
        )
        .expect("prices");
        let output = print(&settings);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("UTF-8 output");
        assert!(stdout.contains(spending), "{stdout}");
        assert!(!stdout.contains("Saved "), "{stdout}");
    }
}

#[test]
fn invalid_complete_records_fail_before_health_checks_or_summary_output() {
    let dir = tempfile::tempdir().expect("fixture");
    let settings = dir.path().join("desktop.toml");
    let log = dir.path().join("routing.jsonl");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
    listener.set_nonblocking(true).expect("nonblocking");
    let settings_text = format!(
        "server_url='http://{}'\nrouting_log={:?}\n",
        listener.local_addr().expect("address"),
        log
    );
    std::fs::write(&settings, &settings_text).expect("settings");
    let valid = record();
    let mut cost = valid.clone();
    cost["cost"]["usd"] = "damaged".into();
    let mut timestamp = valid.clone();
    timestamp["ts"] = "bad".into();
    let mut model = valid.clone();
    model["model"] = " ".into();
    let mut tokens = valid.clone();
    tokens["prompt_tokens"] = (-1).into();
    let mut id = valid.clone();
    id["session_id"] = 7.into();
    for text in [
        format!("{valid}\nnot-json\n"),
        format!("{valid}\n{cost}\n"),
        format!("{valid}\n{timestamp}\n"),
        format!("{valid}\n{model}\n"),
        format!("{valid}\n{tokens}\n"),
        format!("{valid}\n{id}\n"),
    ] {
        std::fs::write(&log, &text).expect("log");
        let output = print(&settings);
        assert!(!output.status.success(), "accepted {text}");
        assert!(output.stdout.is_empty(), "printed an incomplete summary");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Usage unavailable"), "{stderr}");
        assert!(
            stderr.contains("invalid routing record at byte"),
            "{stderr}"
        );
        assert_eq!(
            listener.accept().expect_err("no health request").kind(),
            std::io::ErrorKind::WouldBlock
        );
        assert_eq!(std::fs::read_to_string(&log).expect("unchanged log"), text);
        assert_eq!(
            std::fs::read_to_string(&settings).expect("unchanged settings"),
            settings_text
        );
        assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 2);
    }
}

#[test]
fn sy_home_selects_default_settings_and_usage() {
    let home = tempfile::tempdir().expect("fixture");
    let selected = home.path().join("custom-installation");
    std::fs::create_dir(&selected).expect("installation directory");
    std::fs::write(
        selected.join("desktop.toml"),
        "server_url='http://127.0.0.1:0'\n",
    )
    .expect("settings");
    std::fs::write(selected.join("routing.jsonl"), format!("{}\n", record())).expect("log");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
        .arg("--print")
        .env("HOME", home.path())
        .env("SY_HOME", &selected)
        .output()
        .expect("print");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Spent $1.00"));
}

#[cfg(unix)]
#[test]
fn tilde_paths_fail_when_home_resolution_returns_a_relative_directory() {
    let dir = tempfile::tempdir().expect("fixture");
    let settings = dir.path().join("desktop.toml");
    std::fs::write(&settings, "routing_log='~/routing.jsonl'\n").expect("settings");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_switchyard-desktop"))
        .arg("--print")
        .arg(&settings)
        .env("HOME", "relative")
        .output()
        .expect("print");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Could not find your home directory"));
    assert_eq!(std::fs::read_dir(dir.path()).expect("files").count(), 1);
}
