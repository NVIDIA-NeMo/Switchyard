// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg_attr(windows, windows_subsystem = "windows")]

use std::{path::Path, process::ExitCode};
#[cfg(windows)]
use switchyard_desktop::install::windows;
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
The Cargo command waits for completion. On Windows, run these commands with cargo desktop:
the installed installer cannot replace or remove itself while it runs.
Use uninstall --settings FILE to select another recorded installation.
Preview its removal with: cargo desktop uninstall --settings FILE --dry-run";

const KEY_HELP: &str = "\
Usage: switchyard-desktop-install key <status --config FILE | set NAME | import NAME | remove NAME>

The installed server reads each api_key_env variable from the system credential store, not from your shell.
set reads the key from standard input. import reads the variable once from your login shell.
Restart the server to use a changed key.";

const INVALID_VALIDATE: &str = "Invalid validate command arguments.";

/// This function runs the command that the arguments name. Each accepted shape is one arm, and
/// any other shape is an error.
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["validate", "--server", server, "--config", config] => validate(server, config),
        ["validate", ..] => Err(INVALID_VALIDATE.into()),
        ["key", "status", "--config", config] => key_status(Path::new(config)),
        ["key", "set", name] => key_set(name),
        ["key", "import", name] => key_import(name),
        ["key", "remove", name] => key_remove(name),
        ["key", ..] => Err(KEY_HELP.into()),
        #[cfg(unix)]
        ["serve", "--settings", file] => serve(Path::new(file)),
        #[cfg(windows)]
        [action @ ("serve" | "desktop"), "--settings", file] => {
            windows::launch_managed(&Settings::load(Path::new(file))?, *action == "desktop")
        }
        ["--help" | "-h"] | ["install" | "uninstall", "--help" | "-h"] => {
            println!("{HELP}");
            Ok(())
        }
        [action @ ("install" | "uninstall")] => perform(action, false, None),
        [action @ ("install" | "uninstall"), "--dry-run"] => perform(action, true, None),
        [action @ ("update" | "uninstall"), "--settings", file] => {
            perform(action, false, Some(Path::new(file)))
        }
        ["uninstall", "--settings", file, "--dry-run"]
        | ["uninstall", "--dry-run", "--settings", file] => {
            perform("uninstall", true, Some(Path::new(file)))
        }
        _ => Err(HELP.into()),
    }
}

/// This function runs the server's config check. Both paths must be absolute and free of ASCII
/// control characters.
fn validate(server: &str, config: &str) -> Result<(), String> {
    for path in [server, config] {
        if !Path::new(path).is_absolute() || path.chars().any(|c| c.is_ascii_control()) {
            return Err(INVALID_VALIDATE.into());
        }
    }
    credentials::validate(Path::new(server), Path::new(config))
}

/// This function runs `action` with the settings in `file`. Without a file, uninstall uses the
/// recorded installation and the other actions read the environment.
fn perform(action: &str, dry_run: bool, file: Option<&Path>) -> Result<(), String> {
    let settings = match file {
        Some(file) => Settings::load(file),
        None if action == "uninstall" => Settings::for_uninstall(),
        None => Settings::from_env(),
    }?;
    // Windows cannot replace or delete a running executable, so the installed helper must not
    // change its own installation. The app's update runs a temporary copy instead.
    #[cfg(windows)]
    if !dry_run
        && std::env::current_exe()
            .and_then(std::fs::canonicalize)
            .is_ok_and(|exe| {
                std::fs::canonicalize(settings.binary("switchyard-desktop-install"))
                    .is_ok_and(|installed| installed == exe)
            })
    {
        return Err(format!(
            "The installed installer cannot {action} the installation it belongs to, because \
             Windows keeps a running program from being replaced. Run cargo desktop {} from the \
             Switchyard source checkout instead.",
            if action == "update" {
                "install"
            } else {
                action
            }
        ));
    }
    execute(&settings, action, dry_run)
}

/// This function prints, as JSON, whether each variable that `config` declares has a saved key.
fn key_status(config: &Path) -> Result<(), String> {
    let status = serde_json::to_string(&credentials::status(config)?).map_err(|e| e.to_string())?;
    println!("{status}");
    Ok(())
}

fn key_set(name: &str) -> Result<(), String> {
    credentials::set(name, &read_key(name)?)?;
    println!("Saved the key for {name}. Restart the server to use it.");
    Ok(())
}

fn key_import(name: &str) -> Result<(), String> {
    credentials::import(name)?;
    println!("Saved {name} from your login shell. Restart the server to use it.");
    Ok(())
}

fn key_remove(name: &str) -> Result<(), String> {
    credentials::remove(name)?;
    println!("Removed the saved key for {name}.");
    Ok(())
}

/// This function replaces this process with the server and gives it the saved keys. Exec keeps
/// the service manager's process ID.
#[cfg(unix)]
fn serve(file: &Path) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    let s = Settings::load(file)?;
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
    Err(format!("Could not start the server: {error}"))
}

/// This function reads a key from standard input. At a terminal it prompts and turns off echo,
/// so the key never appears on screen or in the command line.
fn read_key(name: &str) -> Result<String, String> {
    use std::io::Read;
    let mut stdin = std::io::stdin().lock();
    let mut key = String::new();
    #[cfg(unix)]
    if std::io::IsTerminal::is_terminal(&stdin) {
        use std::io::BufRead;
        eprint!("Key for {name}: ");
        // SAFETY: termios is plain data; tcgetattr fills it before it is read.
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut saved) } != 0 {
            return Err("Could not read the terminal settings.".into());
        }
        let quiet = libc::termios {
            c_lflag: saved.c_lflag & !libc::ECHO,
            ..saved
        };
        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &quiet) };
        let read = stdin.read_line(&mut key);
        unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &saved) };
        eprintln!();
        return read.map(|_| key).map_err(|e| e.to_string());
    }
    let _ = name;
    stdin
        .read_to_string(&mut key)
        .map(|_| key)
        .map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    // The Windows subsystem starts without a console. Commands that a person runs attach to the
    // caller's console so their output appears there.
    #[cfg(windows)]
    if !matches!(
        std::env::args_os()
            .nth(1)
            .as_deref()
            .and_then(std::ffi::OsStr::to_str),
        Some("serve" | "desktop" | "update" | "validate")
    ) {
        unsafe { windows_sys::Win32::System::Console::AttachConsole(u32::MAX) };
    }
    let result = run();
    // A temporary updater copy deletes itself when it finishes.
    #[cfg(windows)]
    let result = if std::env::current_exe().is_ok_and(|exe| {
        exe.file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("switchyard-update-"))
    }) {
        self_replace::self_delete()
            .map_err(|e| e.to_string())
            .and(result)
    } else {
        result
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("switchyard-desktop-install: {error}");
            ExitCode::FAILURE
        }
    }
}
