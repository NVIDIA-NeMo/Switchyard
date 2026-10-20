// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs the systemd user service for the server.

use std::{
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};

use super::{Settings, present, remove, run, write};

const UNIT: &str = "switchyard.service";

pub fn preflight() -> Result<(), String> {
    run(systemctl(&["show-environment"]).stdout(Stdio::null()))
}

pub fn stop(s: &Settings) -> Result<(), String> {
    if present(&s.service_dir.join(UNIT))? {
        run(&mut systemctl(&["stop", UNIT]))?;
    }
    Ok(())
}

pub fn install(s: &Settings, keys_saved: bool) -> Result<(), String> {
    let unit = format!(
        "[Unit]\n\
         Description=Switchyard local server\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={} serve --settings {}\n\
         Restart=on-failure\n\
         RestartSec=10\n\
         Environment=RUST_LOG=info\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        systemd_arg(&s.binary("switchyard-desktop-install")),
        systemd_arg(&s.sy_home.join("install.toml")),
    );
    write(&s.service_dir.join(UNIT), unit.as_bytes(), false)?;
    run(&mut systemctl(&["daemon-reload"]))?;
    run(&mut systemctl(&["enable", UNIT]))?;
    run(&mut systemctl(&["restart", UNIT]))?;
    // A server waiting for a key exits and restarts until the key is saved.
    if !keys_saved {
        return Ok(());
    }
    thread::sleep(Duration::from_secs(2));
    run(&mut systemctl(&["is-active", "--quiet", UNIT]))
        .map_err(|e| format!("{e}. See journalctl --user -u switchyard."))
}

/// Removes the service. The binaries stay in SY_HOME.
pub fn uninstall(s: &Settings) -> Result<(), String> {
    let unit = s.service_dir.join(UNIT);
    if present(&unit)? {
        run(&mut systemctl(&["disable", UNIT]))?;
        remove(&unit)?;
    }
    run(&mut systemctl(&["daemon-reload"]))
}

fn systemctl(args: &[&str]) -> Command {
    let mut command = Command::new("systemctl");
    command.arg("--user").args(args);
    command
}

fn systemd_arg(path: &Path) -> String {
    // systemd expands percent specifiers and environment variables inside quoted arguments.
    format!(
        "\"{}\"",
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
            .replace('$', "$$")
    )
}
