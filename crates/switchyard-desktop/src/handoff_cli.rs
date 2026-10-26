// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `handoff` subcommand continues a session in another harness from the terminal it runs
//! in. It reviews the hand-off the way the apps do, writes it, and then replaces itself with
//! the destination harness in the session's directory.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::Value;

use crate::config::Config;
use crate::controller::{
    Action, Controller, HandoffSource, RouteInfo, canonical, harness_name, other_harness,
};
use crate::harness::{self, Harness};

pub const USAGE: &str = "\
Usage: switchyard-desktop handoff --to <codex|claude> [options]

This continues a session in another harness, here in this terminal. It reads the
session's transcript, writes a bounded hand-off file that only you can read, and
starts the destination harness in the session's directory with an instruction to
read that file first. Nothing is written until you confirm.

  --to HARNESS           The destination: codex or claude.
  --from HARNESS         The source harness. Default: the other one.
  --transcript FILE      The source transcript. Default: the newest transcript of
                         the source harness that was recorded in this directory.
  --session ID           Continue a session that the app started (sy-…) instead.
  --route ROUTE          The route the destination uses, by ID. Default: the first
                         route the destination can use.
  --account NAME         A saved account for the destination. Default: its login.
  --source-account NAME  A saved account whose transcripts hold the source session.
  --show                 Print the whole hand-off text in the review.
  --yes, -y              Skip the confirmation.
  --settings FILE        Read app settings from FILE.

Quit the source harness first. Hand-off files are removed after seven days.";

#[derive(Debug)]
pub struct Options {
    pub to: Harness,
    pub from: Harness,
    pub transcript: Option<PathBuf>,
    pub session: Option<String>,
    pub route: Option<String>,
    pub account: Option<String>,
    pub source_account: Option<String>,
    pub show: bool,
    pub yes: bool,
    pub settings: Option<PathBuf>,
}

/// The program that replaces this one, as the controller returned it.
#[derive(Debug, serde::Deserialize)]
pub struct Launch {
    pub program: PathBuf,
    pub directory: PathBuf,
    pub args: Vec<String>,
    pub environment: Vec<(String, String)>,
    pub remove_environment: Vec<String>,
}

pub fn main(args: &[String]) -> ExitCode {
    let options = match parse(args) {
        Ok(Some(options)) => options,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("switchyard-desktop handoff: {error}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match prepare(&options) {
        Ok(Some(launch)) => exec(launch),
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switchyard-desktop handoff: {error}");
            ExitCode::FAILURE
        }
    }
}

/// This reads the options. `Ok(None)` asks for the usage text.
pub fn parse(args: &[String]) -> Result<Option<Options>, String> {
    let mut to = None;
    let mut from = None;
    let mut transcript = None;
    let mut session = None;
    let mut route = None;
    let mut account = None;
    let mut source_account = None;
    let mut show = false;
    let mut yes = false;
    let mut settings = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--to" => to = Some(parse_harness(&value(&mut rest, arg)?)?),
            "--from" => from = Some(parse_harness(&value(&mut rest, arg)?)?),
            "--transcript" => transcript = Some(PathBuf::from(value(&mut rest, arg)?)),
            "--session" => session = Some(value(&mut rest, arg)?),
            "--route" => route = Some(value(&mut rest, arg)?),
            "--account" => account = Some(value(&mut rest, arg)?),
            "--source-account" => source_account = Some(value(&mut rest, arg)?),
            "--settings" => settings = Some(PathBuf::from(value(&mut rest, arg)?)),
            "--show" => show = true,
            "-y" | "--yes" => yes = true,
            other => return Err(format!("Unknown option {other}.")),
        }
    }
    let to = to.ok_or("Pass --to codex or --to claude.")?;
    if transcript.is_some() && session.is_some() {
        return Err("Pass --transcript or --session, not both.".into());
    }
    Ok(Some(Options {
        to,
        from: from.unwrap_or(other_harness(to)),
        transcript,
        session,
        route,
        account,
        source_account,
        show,
        yes,
        settings,
    }))
}

fn value(rest: &mut std::slice::Iter<'_, String>, flag: &str) -> Result<String, String> {
    rest.next()
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value."))
}

fn parse_harness(text: &str) -> Result<Harness, String> {
    match text {
        "claude" | "claude-code" => Ok(Harness::Claude),
        "codex" | "codex-cli" => Ok(Harness::CodexCli),
        other => Err(format!("Unknown harness {other:?}. Use codex or claude.")),
    }
}

/// This reviews the hand-off, asks for confirmation, writes it, and returns the program to
/// run. `Ok(None)` means the user declined, and nothing was written.
pub fn prepare(options: &Options) -> Result<Option<Launch>, String> {
    let settings = match &options.settings {
        Some(path) => path.clone(),
        None => Config::default_path()?,
    };
    let config = Config::load(&settings)?;
    let mut controller = Controller::new(config, settings);
    let destination = harness_name(options.to);
    if harness::binary(options.to).is_none() {
        return Err(format!(
            "{destination} is not installed, or its program is not on PATH."
        ));
    }
    let snapshot = controller.snapshot()?;
    let (route, id) = choose_route(&snapshot.routes, options.to, options.route.as_deref())?;
    let source = match (&options.session, &options.transcript) {
        (Some(id), _) => HandoffSource::Managed { id: id.clone() },
        (None, Some(path)) => HandoffSource::Transcript {
            path: std::path::absolute(path)
                .map_err(|error| error.to_string())?
                .display()
                .to_string(),
        },
        (None, None) => newest_here(
            &mut controller,
            options.from,
            options.source_account.clone(),
        )?,
    };
    let preview = controller.dispatch(Action::PreviewHandoff {
        source: source.clone(),
        tool: options.to,
        account: options.account.clone(),
        route: route.clone(),
        id: id.clone(),
    })?;
    let data = preview.data;
    let mut out = std::io::stdout().lock();
    let report = |out: &mut std::io::StdoutLock<'_>, text: &str| {
        writeln!(out, "{text}")
            .and_then(|()| out.flush())
            .map_err(|error| error.to_string())
    };
    report(&mut out, &review_text(&data, options.show))?;
    if !options.yes {
        if !std::io::stdin().is_terminal() {
            return Err("Pass --yes to continue without a prompt.".into());
        }
        write!(
            out,
            "Write the hand-off and start {destination} here? [y/N] "
        )
        .and_then(|()| out.flush())
        .map_err(|error| error.to_string())?;
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|error| error.to_string())?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            report(&mut out, "Nothing was written.")?;
            return Ok(None);
        }
    }
    let reply = controller.dispatch(Action::Handoff {
        source,
        tool: options.to,
        account: options.account.clone(),
        route,
        id,
        preview_token: data["preview_token"].as_str().map(str::to_owned),
        attach: true,
    })?;
    report(&mut out, &reply.message)?;
    serde_json::from_value(reply.data["invocation"].clone())
        .map(Some)
        .map_err(|error| format!("The hand-off reply has no command: {error}"))
}

/// The route the destination uses: the one named, or the first one it can use.
fn choose_route(
    routes: &[RouteInfo],
    tool: Harness,
    wanted: Option<&str>,
) -> Result<(String, String), String> {
    let reason = |route: &RouteInfo| route.compatibility.get(&tool).cloned().flatten();
    if let Some(wanted) = wanted {
        let route = routes
            .iter()
            .find(|route| route.id == wanted || route.key == wanted)
            .ok_or_else(|| {
                let ids: Vec<&str> = routes.iter().map(|route| route.id.as_str()).collect();
                format!("No route {wanted:?}. Routes: {}.", ids.join(", "))
            })?;
        if let Some(reason) = reason(route) {
            return Err(format!(
                "{} cannot use route {}: {reason}",
                harness_name(tool),
                route.id
            ));
        }
        return Ok((route.key.clone(), route.id.clone()));
    }
    let route = routes
        .iter()
        .find(|route| reason(route).is_none())
        .ok_or_else(|| {
            let reasons: Vec<String> = routes
                .iter()
                .filter_map(|route| Some(format!("{}: {}", route.id, reason(route)?)))
                .collect();
            format!(
                "No route in the server config is usable by {}. {}",
                harness_name(tool),
                reasons.join(" ")
            )
        })?;
    Ok((route.key.clone(), route.id.clone()))
}

/// The newest transcript of `tool` that was recorded in the current directory.
fn newest_here(
    controller: &mut Controller,
    tool: Harness,
    account: Option<String>,
) -> Result<HandoffSource, String> {
    let here = canonical(&std::env::current_dir().map_err(|error| error.to_string())?);
    let reply = controller.dispatch(Action::RecentSessions {
        tool: Some(tool),
        account,
    })?;
    let sessions = reply.data["sessions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let label = harness_name(tool);
    let recorded_here = |session: &Value| {
        session["cwd"]
            .as_str()
            .is_some_and(|cwd| canonical(Path::new(cwd)) == here)
    };
    if let Some(session) = sessions.iter().find(|session| recorded_here(session)) {
        let path = session["transcript"]
            .as_str()
            .ok_or("The session listing has no transcript path.")?;
        return Ok(HandoffSource::Transcript {
            path: path.to_owned(),
        });
    }
    let mut message = format!("No {label} session was recorded in {}.", here.display());
    if sessions.is_empty() {
        message.push_str(&format!(
            " No transcripts were found under {}.",
            reply.data["roots"][0]["transcripts"]
                .as_str()
                .unwrap_or("?")
        ));
    } else {
        message.push_str(" Newest sessions:");
        for session in sessions.iter().take(5) {
            message.push_str(&format!(
                "\n  {}  {}\n    {}",
                session["last_activity"].as_str().unwrap_or("?"),
                session["cwd"].as_str().unwrap_or("(directory unknown)"),
                session["transcript"].as_str().unwrap_or("?")
            ));
        }
        message.push_str("\nRun this in the session's directory, or pass --transcript FILE.");
    }
    Err(message)
}

fn review_text(data: &Value, show: bool) -> String {
    let text = |value: &Value| value.as_str().unwrap_or("").to_owned();
    let source = &data["source"];
    let destination = &data["destination"];
    let mut lines = vec![
        "Hand-off review".to_owned(),
        format!(
            "From:        {} · {}",
            text(&source["label"]),
            text(&source["transcript"])
        ),
        format!(
            "To:          {} · route {} · {}",
            text(&destination["label"]),
            text(&destination["route"]),
            destination["account_directory"]
                .as_str()
                .unwrap_or("current login")
        ),
        format!("Directory:   {}", text(&destination["directory"])),
        format!("Session ID:  {}", text(&destination["session_id"])),
        format!("Hand-off:    {}", text(&data["handoff_file"])),
        format!(
            "Excerpt:     {} bytes · {} turns · {} files · {} commands{}",
            data["bytes"],
            data["turns"],
            data["files_touched"],
            data["commands"],
            if data["truncated"].as_bool().unwrap_or(false) {
                " · cut to fit"
            } else {
                ""
            }
        ),
    ];
    if let Some(warnings) = data["warnings"].as_array().filter(|list| !list.is_empty()) {
        lines.push("Warnings:".into());
        lines.extend(
            warnings
                .iter()
                .map(|warning| format!("  - {}", text(warning))),
        );
    }
    lines.push("Command:".into());
    lines.extend(
        text(&data["command"])
            .lines()
            .map(|line| format!("  {line}")),
    );
    if show {
        lines.push(String::new());
        lines.push("---- hand-off text, which the new session reads ----".into());
        lines.push(text(&data["handoff"]));
        lines.push("---- end of the hand-off text ----".into());
    } else {
        lines.push("Pass --show to print the hand-off text.".into());
    }
    lines.push(text(&data["notice"]));
    lines.join("\n")
}

fn command(launch: &Launch) -> std::process::Command {
    let mut command = std::process::Command::new(&launch.program);
    command.args(&launch.args).current_dir(&launch.directory);
    for (name, value) in &launch.environment {
        command.env(name, value);
    }
    for name in &launch.remove_environment {
        command.env_remove(name);
    }
    command
}

/// This replaces the current process with the harness, so the terminal belongs to it.
#[cfg(unix)]
fn exec(launch: Launch) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let error = command(&launch).exec();
    eprintln!(
        "switchyard-desktop handoff: could not start {}: {error}",
        launch.program.display()
    );
    ExitCode::FAILURE
}

/// Windows has no exec, so this waits for the harness and passes on its result.
#[cfg(not(unix))]
fn exec(launch: Launch) -> ExitCode {
    match command(&launch).status() {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!(
                "switchyard-desktop handoff: could not start {}: {error}",
                launch.program.display()
            );
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn options_need_a_destination_and_accept_only_cli_harnesses() {
        assert!(parse(&args(&[])).unwrap_err().contains("--to"));
        assert!(
            parse(&args(&["--to", "pi"]))
                .unwrap_err()
                .contains("codex or claude")
        );
        assert!(parse(&args(&["--to", "claude", "--route"])).is_err());
        assert!(parse(&args(&["--to", "claude", "--bogus"])).is_err());
        assert!(
            parse(&args(&[
                "--to",
                "claude",
                "--transcript",
                "a",
                "--session",
                "b"
            ]))
            .is_err()
        );
        assert!(parse(&args(&["--help"])).unwrap().is_none());
        let options = parse(&args(&["--to", "codex", "-y", "--route", "switchyard"]))
            .unwrap()
            .unwrap();
        assert_eq!(
            (options.to, options.from, options.yes),
            (Harness::CodexCli, Harness::Claude, true)
        );
        assert_eq!(options.route.as_deref(), Some("switchyard"));
        let options = parse(&args(&["--to", "claude"])).unwrap().unwrap();
        assert_eq!(options.from, Harness::CodexCli);
    }

    #[test]
    fn the_review_names_the_file_the_command_and_the_warnings() {
        let data = json!({
            "source": {"label": "Codex", "transcript": "/codex/rollout-1.jsonl"},
            "destination": {"label": "Claude Code", "route": "claude", "directory": "/work",
                "session_id": "sy-1", "account_directory": null},
            "handoff_file": "/handoffs/hd-1/HANDOFF.md",
            "bytes": 1200, "turns": 3, "files_touched": 2, "commands": 1, "truncated": true,
            "warnings": ["The transcript changed in the last two minutes."],
            "command": "cd '/work'\n'claude' '--model' 'claude'",
            "handoff": "# Hand-off from Codex\n\nFix the flaky login test",
            "notice": "Nothing is uploaded.",
        });
        let short = review_text(&data, false);
        assert!(short.contains("To:          Claude Code · route claude · current login"));
        assert!(short.contains("cut to fit") && short.contains("  - The transcript changed"));
        assert!(short.contains("  'claude' '--model' 'claude'"));
        assert!(!short.contains("Fix the flaky login test"));
        assert!(review_text(&data, true).contains("Fix the flaky login test"));
    }

    // The child half of this test runs the subcommand against a fixture home and ends inside
    // the stub harness, so the parent sees the stub's exit status and its record.
    #[cfg(unix)]
    #[test]
    fn handoff_command_writes_the_file_and_runs_the_destination_here() {
        use std::os::unix::fs::PermissionsExt;
        use std::{fs, process::Command};
        if let Some(home) = std::env::var_os("SWITCHYARD_HANDOFF_CLI_FIXTURE") {
            let home = PathBuf::from(home);
            let project = home.join("project");
            fs::create_dir(&project).expect("project");
            let settings = home.join("desktop.toml");
            let server = home.join("server.toml");
            fs::write(
                &server,
                "[llm_clients.c]\n\
                 base_url='http://localhost:1234'\n\
                 [targets.m]\n\
                 id='actual'\n\
                 llm_client='c'\n\
                 [routes.r]\n\
                 id='public'\n\
                 type='passthrough'\n\
                 target='m'\n",
            )
            .expect("server");
            fs::write(
                &settings,
                format!(
                    "config_file={server:?}\nrouting_log={:?}\n",
                    home.join("routing.jsonl")
                ),
            )
            .expect("settings");
            let day = PathBuf::from(std::env::var_os("CODEX_HOME").expect("CODEX_HOME"))
                .join("sessions/2026/10/26");
            fs::create_dir_all(&day).expect("rollouts");
            let now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
            let rollout = |name: &str, cwd: &Path, prompt: &str| {
                let thread = uuid::Uuid::now_v7().to_string();
                let lines = [
                    json!({"timestamp": now, "type": "session_meta", "payload": {
                        "id": thread, "cwd": cwd, "timestamp": now}}),
                    json!({"timestamp": now, "type": "event_msg", "payload": {
                        "type": "user_message", "message": prompt}}),
                    json!({"timestamp": now, "type": "event_msg", "payload": {
                        "type": "agent_message",
                        "message": "I changed auth.rs and the test passes now."}}),
                ];
                let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
                fs::write(day.join(format!("rollout-{name}-{thread}.jsonl")), text)
                    .expect("rollout");
            };
            // A session from another directory is not the one to continue here.
            rollout("2026-10-26T11-00-00", &home, "Add retries to the uploader");
            rollout("2026-10-26T10-00-00", &project, "Fix the flaky login test");
            std::env::set_current_dir(&project).expect("cwd");
            let options = parse(&args(&[
                "--to",
                "claude",
                "--yes",
                "--settings",
                settings.to_str().expect("settings path"),
            ]))
            .expect("options")
            .expect("options");
            let launch = prepare(&options).expect("prepare").expect("confirmed");
            assert_eq!(launch.directory, project);
            assert!(launch.program.ends_with("claude"), "{launch:?}");
            exec(launch);
            panic!("exec returned");
        }
        let home = tempfile::tempdir().expect("fixture");
        let bin = home.path().join("bin");
        fs::create_dir(&bin).expect("bin");
        let stub = bin.join("claude");
        fs::write(
            &stub,
            "#!/bin/bash\n\
             printf '%s\\n' \"$PWD\" \"${CLAUDE_CONFIG_DIR:-}\" \"${ANTHROPIC_CUSTOM_HEADERS:-}\" \
             \"$@\" > \"$HOME/claude-arguments\"\n",
        )
        .expect("stub");
        fs::set_permissions(&stub, fs::Permissions::from_mode(0o755)).expect("executable");
        let output = Command::new(std::env::current_exe().expect("test binary"))
            .args([
                "--exact",
                "handoff_cli::tests::handoff_command_writes_the_file_and_runs_the_destination_here",
            ])
            .env("SWITCHYARD_HANDOFF_CLI_FIXTURE", home.path())
            .env("HOME", home.path())
            .env("CODEX_HOME", home.path().join("codex-home"))
            .env("CLAUDE_CONFIG_DIR", home.path().join("claude-config"))
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .output()
            .expect("child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            stdout.contains("Hand-off review")
                && stdout.contains("Continuing the session here in Claude Code"),
            "{stdout}"
        );
        let arguments =
            fs::read_to_string(home.path().join("claude-arguments")).expect("the stub ran");
        let project = home.path().join("project");
        let mut lines = arguments.lines();
        assert_eq!(
            fs::canonicalize(lines.next().expect("PWD")).expect("PWD"),
            fs::canonicalize(&project).expect("project")
        );
        assert_eq!(
            lines.next().expect("CLAUDE_CONFIG_DIR"),
            home.path().join("claude-config").display().to_string()
        );
        assert!(
            arguments.contains("x-switchyard-session-id: sy-"),
            "{arguments}"
        );
        assert!(
            arguments.contains("x-switchyard-origin: claude"),
            "{arguments}"
        );
        for flag in [
            "--settings",
            "--model\npublic",
            "--session-id",
            "--append-system-prompt",
            "--add-dir",
            "Read the hand-off file",
        ] {
            assert!(arguments.contains(flag), "{flag}\n{arguments}");
        }
        // The excerpt stays in the file; the command line only points at it.
        assert!(
            !arguments.contains("Fix the flaky login test"),
            "{arguments}"
        );
        let handoffs = home.path().join(".switchyard/handoffs");
        let entries: Vec<PathBuf> = fs::read_dir(&handoffs)
            .expect("handoffs")
            .map(|entry| entry.expect("entry").path())
            .collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
        let file = entries[0].join("HANDOFF.md");
        let text = fs::read_to_string(&file).expect("hand-off");
        assert!(text.contains("Fix the flaky login test") && text.contains("auth.rs"));
        assert!(!text.contains("Add retries to the uploader"));
        assert_eq!(
            fs::metadata(&file).expect("mode").permissions().mode() & 0o777,
            0o600
        );
        assert!(
            arguments.contains(file.to_str().expect("path")),
            "{arguments}"
        );
        let record: Value =
            serde_json::from_str(&fs::read_to_string(entries[0].join("handoff.json")).unwrap())
                .expect("record");
        assert_eq!(record["opened"], json!(true));
        assert_eq!(record["destination"]["directory"], json!(project));
        assert_eq!(record["source"]["tool"], json!("codex_cli"));
    }
}
