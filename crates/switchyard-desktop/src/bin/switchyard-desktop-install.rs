// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(windows, windows_subsystem = "windows")]

use std::{path::Path, process::ExitCode};
use switchyard_desktop::install::{Settings, credentials, execute};

// A backslash right after the opening quote drops the first newline. Every other line break in
// these literals is part of the text, so the lines start at column 0.
const HELP: &str = "\
Usage: cargo desktop <install | uninstall> [--dry-run]

Build and install the local server and macOS or Windows desktop app, or Linux terminal app.
Uninstall uses the recorded installation and preserves settings, history, and recovery copies.

Defaults: SY_HOME=~/.switchyard, SY_PORT=4123, SY_PROFILE=sy,
SY_MODEL=composite-gpt-6-sol-gpt-6-luna, CODEX_HOME=~/.codex.
Example: SY_PROFILE=team SY_MODEL=my-route cargo desktop install

Preview the installation or removal plan with --dry-run. Installation requires
a Rust toolchain and LaunchAgents (macOS) or a systemd user session (Linux).
Windows requires MSVC build tools, WebView2, and a Task Scheduler user session.
The Cargo command waits for completion. On Windows, running the installed installer
starts a temporary process and returns; read SY_HOME/logs/update.log for progress and errors.
Use uninstall --settings FILE to select another recorded installation.
Preview its removal with: cargo desktop uninstall --settings FILE --dry-run";

const KEY_HELP: &str = "\
Usage: switchyard-desktop-install key <status --config FILE | set NAME | import NAME | remove NAME>

The installed server reads each api_key_env variable from the system credential store, not from your shell.
set reads the key from standard input. import reads the variable once from your login shell.
Restart the server to use a changed key.";

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "validate") {
        let [_, server_flag, server, config_flag, config] = args.as_slice() else {
            return Err("Invalid validate command arguments.".into());
        };
        if server_flag != "--server"
            || config_flag != "--config"
            || [server, config]
                .iter()
                .any(|p| !Path::new(p).is_absolute() || p.chars().any(|c| c.is_ascii_control()))
        {
            return Err("Invalid validate command arguments.".into());
        }
        return credentials::validate(Path::new(server), Path::new(config));
    }
    if args.first().is_some_and(|a| a == "key") {
        return key(&args[1..]);
    }
    #[cfg(unix)]
    if let [action, flag, file] = args.as_slice()
        && action == "serve"
        && flag == "--settings"
    {
        use std::os::unix::process::CommandExt;
        // Exec preserves the service manager's PID while the helper supplies saved keys.
        let s = Settings::load(Path::new(file))?;
        let config = s.sy_home.join("composite.toml");
        let mut server = std::process::Command::new(s.binary("switchyard-server"));
        #[cfg(target_os = "linux")]
        credentials::wait_for_keyring()?;
        credentials::configure(&mut server, &config)?;
        let error = server
            .arg("--config")
            .arg(config)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &s.port.to_string(),
                "--routing-log-file",
            ])
            .arg(s.sy_home.join("routing.jsonl"))
            .exec();
        return Err(format!("Could not start the server: {error}"));
    }
    #[cfg(windows)]
    if let Some(result) = windows_command()? {
        return result;
    }
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"]
        || args == ["-h"]
        || (args.len() == 2
            && matches!(args[0].as_str(), "install" | "uninstall")
            && matches!(args[1].as_str(), "--help" | "-h"))
    {
        println!("{HELP}");
        return Ok(());
    }
    let (action, dry_run, file) = match args.as_slice() {
        [action] if action == "install" || action == "uninstall" => (action.as_str(), false, None),
        [action, flag] if (action == "install" || action == "uninstall") && flag == "--dry-run" => {
            (action.as_str(), true, None)
        }
        [action, flag, file]
            if (action == "update" || action == "uninstall") && flag == "--settings" =>
        {
            (action.as_str(), false, Some(Path::new(file)))
        }
        [action, settings, file, dry]
            if action == "uninstall" && settings == "--settings" && dry == "--dry-run" =>
        {
            (action.as_str(), true, Some(Path::new(file)))
        }
        [action, dry, settings, file]
            if action == "uninstall" && settings == "--settings" && dry == "--dry-run" =>
        {
            (action.as_str(), true, Some(Path::new(file)))
        }
        _ => return Err(HELP.into()),
    };
    let settings = if let Some(file) = file {
        Settings::load(file)?
    } else if action == "uninstall" {
        Settings::for_uninstall()?
    } else {
        Settings::from_env()?
    };
    #[cfg(windows)]
    if !dry_run
        && settings
            .binary("switchyard-desktop-install")
            .try_exists()
            .map_err(|e| e.to_string())?
        && std::fs::canonicalize(std::env::current_exe().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
            == std::fs::canonicalize(settings.binary("switchyard-desktop-install"))
                .map_err(|e| e.to_string())?
    {
        return handoff(&settings, action);
    }
    execute(&settings, action, dry_run)
}

fn key(args: &[String]) -> Result<(), String> {
    match args {
        [action, flag, config] if flag == "--config" && action == "status" => {
            let status = credentials::status(Path::new(config))?;
            println!(
                "{}",
                serde_json::to_string(&status).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        [action, name] if action == "set" => {
            credentials::set(name, &read_key(name)?)?;
            println!("Saved the key for {name}. Restart the server to use it.");
            Ok(())
        }
        [action, name] if action == "import" => {
            credentials::import(name)?;
            println!("Saved {name} from your login shell. Restart the server to use it.");
            Ok(())
        }
        [action, name] if action == "remove" => {
            credentials::remove(name)?;
            println!("Removed the saved key for {name}.");
            Ok(())
        }
        _ => Err(KEY_HELP.into()),
    }
}

/// This function reads a key from standard input. At a terminal it prompts and turns off echo,
/// so the key never appears on screen or in the command line.
fn read_key(name: &str) -> Result<String, String> {
    use std::io::Read;
    let stdin = std::io::stdin();
    #[cfg(unix)]
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 1 {
        use std::io::BufRead;
        eprint!("Key for {name}: ");
        // SAFETY: termios is plain data; tcgetattr fills it before it is read.
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut saved) } != 0 {
            return Err("Could not read the terminal settings.".into());
        }
        let mut quiet = saved;
        quiet.c_lflag &= !libc::ECHO;
        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &quiet) };
        let mut line = String::new();
        let read = stdin.lock().read_line(&mut line);
        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &saved) };
        eprintln!();
        read.map_err(|e| e.to_string())?;
        return Ok(line);
    }
    let _ = name;
    let mut key = String::new();
    stdin
        .lock()
        .read_to_string(&mut key)
        .map_err(|e| e.to_string())?;
    Ok(key)
}

fn main() -> ExitCode {
    #[cfg(windows)]
    if !std::env::args_os().nth(1).is_some_and(|arg| {
        matches!(
            arg.to_str(),
            Some("serve" | "desktop" | "apply" | "update" | "validate")
        )
    }) {
        unsafe {
            windows_sys::Win32::System::Console::AttachConsole(u32::MAX);
        }
    }
    let result = run();
    #[cfg(windows)]
    if std::env::current_exe()
        .ok()
        .and_then(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().starts_with("switchyard-update-"))
        })
        .unwrap_or(false)
        && let Err(error) = self_replace::self_delete()
    {
        eprintln!("switchyard-desktop-install: {error}");
        return ExitCode::FAILURE;
    }
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switchyard-desktop-install: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Handoff<T> {
    settings: T,
}

#[cfg(windows)]
fn windows_command() -> Result<Option<Result<(), String>>, String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [action, flag, file]
            if matches!(action.as_str(), "serve" | "desktop") && flag == "--settings" =>
        {
            let settings = Settings::load(Path::new(file))?;
            Ok(Some(switchyard_desktop::install::windows::launch_managed(
                &settings,
                action == "desktop",
            )))
        }
        [apply, flag, file, action, operation, wait, pid]
            if apply == "apply"
                && flag == "--handoff-record"
                && action == "--action"
                && wait == "--wait"
                && matches!(operation.as_str(), "install" | "update" | "uninstall") =>
        {
            let path = Path::new(file);
            let metadata = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
            let parent = path.parent().ok_or("Invalid handoff record.")?;
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("Invalid handoff record.")?;
            if !metadata.file_type().is_file()
                || !name.starts_with("switchyard-handoff-")
                || !name.ends_with(".toml")
                || std::fs::canonicalize(parent).map_err(|e| e.to_string())?
                    != std::fs::canonicalize(std::env::temp_dir()).map_err(|e| e.to_string())?
            {
                return Err("Invalid handoff record.".into());
            }
            let handoff: Handoff<Settings> =
                toml::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            handoff.settings.validate()?;
            let pid = pid
                .parse::<u32>()
                .map_err(|_| "Invalid parent process ID.")?;
            if pid == 0 {
                return Err("Invalid parent process ID.".into());
            }
            let _record = tempfile::TempPath::try_from_path(path).map_err(|e| e.to_string())?;
            let result = switchyard_desktop::install::windows::wait_parent(pid)
                .and_then(|()| execute(&handoff.settings, operation, false));
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}
#[cfg(windows)]
// The installed executable cannot replace itself while Windows holds it open.
fn handoff(settings: &Settings, action: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    switchyard_desktop::install::windows::record_handoff(settings)?;
    let helper = switchyard_desktop::install::windows::temporary_updater(
        &std::env::current_exe().map_err(|e| e.to_string())?,
    )?;
    let mut record = tempfile::Builder::new()
        .prefix("switchyard-handoff-")
        .suffix(".toml")
        .tempfile()
        .map_err(|e| e.to_string())?;
    record
        .write_all(
            toml::to_string(&Handoff { settings })
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
    let file = record.into_temp_path();
    std::fs::create_dir_all(settings.sy_home.join("logs")).map_err(|e| e.to_string())?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(settings.sy_home.join("logs/update.log"))
        .map_err(|e| e.to_string())?;
    let mut child = Command::new(&helper);
    switchyard_desktop::install::windows::hide_console(&mut child);
    child
        .arg("apply")
        .arg("--handoff-record")
        .arg(&file)
        .arg("--action")
        .arg(action)
        .arg("--wait")
        .arg(std::process::id().to_string())
        .stdin(Stdio::null())
        .stderr(log.try_clone().map_err(|e| e.to_string())?)
        .stdout(log)
        .spawn()
        .map_err(|e| e.to_string())?;
    helper.keep().map_err(|e| e.to_string())?;
    file.keep().map_err(|e| e.to_string())?;
    println!(
        "Started {action}. Progress and errors: {}",
        settings.sy_home.join("logs/update.log").display()
    );
    Ok(())
}
