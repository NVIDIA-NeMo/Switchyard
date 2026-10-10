// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module installs per-user Windows tasks and shortcuts and replaces stopped app executables.

use super::*;
use std::os::windows::{
    ffi::{OsStrExt, OsStringExt},
    process::CommandExt,
};
use windows_sys::Win32::{
    Foundation::{CloseHandle, ERROR_NOT_FOUND, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0},
    Security::Credentials::{
        CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC, CREDENTIALW, CredFree, CredReadW, CredWriteW,
    },
    System::{
        Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW,
            TH32CS_SNAPPROCESS,
        },
        Threading::{
            CREATE_NO_WINDOW, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
            PROCESS_TERMINATE, QueryFullProcessImageNameW, TerminateProcess, WaitForSingleObject,
        },
    },
};

struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}
fn wide(value: &std::ffi::OsStr) -> Vec<u16> {
    value.encode_wide().chain(Some(0)).collect()
}
fn error() -> String {
    std::io::Error::last_os_error().to_string()
}
pub fn hide_console(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

pub fn task_name(s: &Settings, desktop: bool) -> String {
    // The binary directory keeps task names stable when SY_HOME changes or a Rust toolchain changes.
    let mut directory = PathBuf::new();
    for component in s.bin_dir().components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                directory.pop();
            }
            _ => directory.push(component.as_os_str()),
        }
    }
    let key = directory
        .to_string_lossy()
        .replace('\\', "/")
        .to_lowercase();
    let key = if let Some(unc) = key.strip_prefix("//?/unc/") {
        format!("//{unc}")
    } else {
        key.strip_prefix("//?/").unwrap_or(&key).to_owned()
    };
    let hash = key.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    format!(
        "Switchyard-{}-{hash:016x}",
        if desktop { "Desktop" } else { "Server" }
    )
}
fn scheduler() -> Command {
    let mut c = Command::new("schtasks.exe");
    hide_console(&mut c);
    c
}
fn exists(s: &Settings, desktop: bool) -> Result<bool, String> {
    scheduler()
        .args(["/Query", "/TN", &task_name(s, desktop)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .map_err(|e| e.to_string())
}
fn identity() -> Result<String, String> {
    use windows_sys::Win32::{
        Foundation::LocalFree,
        Security::{
            Authorization::ConvertSidToStringSidW, GetTokenInformation, TOKEN_QUERY, TOKEN_USER,
            TokenUser,
        },
        System::Threading::{GetCurrentProcess, OpenProcessToken},
    };
    unsafe {
        let mut token = std::ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(error());
        }
        let token = Handle(token);
        let mut length = 0;
        GetTokenInformation(token.0, TokenUser, std::ptr::null_mut(), 0, &mut length);
        if length < std::mem::size_of::<TOKEN_USER>() as u32 {
            return Err(error());
        }
        let mut buffer = vec![0usize; (length as usize).div_ceil(std::mem::size_of::<usize>())];
        if GetTokenInformation(
            token.0,
            TokenUser,
            buffer.as_mut_ptr().cast(),
            length,
            &mut length,
        ) == 0
        {
            return Err(error());
        }
        let user = &*buffer.as_ptr().cast::<TOKEN_USER>();
        let mut sid = std::ptr::null_mut();
        if ConvertSidToStringSidW(user.User.Sid, &mut sid) == 0 {
            return Err(error());
        }
        let length = (0..256).find(|&index| *sid.add(index) == 0);
        let result = length
            .ok_or_else(|| "Invalid Windows account SID.".to_owned())
            .and_then(|length| {
                String::from_utf16(std::slice::from_raw_parts(sid, length))
                    .map_err(|e| e.to_string())
            });
        LocalFree(sid.cast());
        result
    }
}
// Separate tasks let the server continue after the desktop app exits.
fn task_xml(s: &Settings, user: &str, desktop: bool) -> String {
    let program = s.binary("switchyard-desktop-install");
    let args = format!(
        "{} --settings {}",
        if desktop { "desktop" } else { "serve" },
        quote_arg(&s.sy_home.join("install.toml").display().to_string())
    );
    let user = xml(user);
    let restart = if desktop {
        ""
    } else {
        "<RestartOnFailure><Interval>PT1M</Interval><Count>999</Count></RestartOnFailure>"
    };
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-16\"?><Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\"><Triggers><LogonTrigger><Enabled>true</Enabled><UserId>{user}</UserId></LogonTrigger></Triggers><Principals><Principal id=\"User\"><UserId>{user}</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals><Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries><StopIfGoingOnBatteries>false</StopIfGoingOnBatteries><AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable><ExecutionTimeLimit>PT0S</ExecutionTimeLimit>{restart}</Settings><Actions Context=\"User\"><Exec><Command>{}</Command><Arguments>{}</Arguments><WorkingDirectory>{}</WorkingDirectory></Exec></Actions></Task>",
        xml(&program.display().to_string()),
        xml(&args),
        xml(&s.bin_dir().display().to_string())
    )
}
// Task Scheduler parses a Windows argument string; trailing slashes must not escape its closing quote.
pub fn quote_arg(value: &str) -> String {
    let mut out = String::from("\"");
    let mut slashes = 0;
    for ch in value.chars() {
        if ch == '\\' {
            slashes += 1;
            continue;
        }
        out.extend(std::iter::repeat_n(
            '\\',
            if ch == '"' { slashes * 2 + 1 } else { slashes },
        ));
        out.push(ch);
        slashes = 0;
    }
    out.extend(std::iter::repeat_n('\\', slashes * 2));
    out.push('"');
    out
}
fn shortcut_paths(s: &Settings) -> [PathBuf; 2] {
    [
        s.service_dir.join("Switchyard.lnk"),
        s.service_dir.join("Startup/Switchyard.lnk"),
    ]
}
pub fn install(s: &Settings) -> Result<(), String> {
    write(
        &s.bin_dir().join("install.toml"),
        toml::to_string(s).map_err(|e| e.to_string())?.as_bytes(),
        false,
    )?;
    let user = identity()?;
    for desktop in [false, true] {
        let mut file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
        let data: Vec<u8> = [0xfeffu16]
            .into_iter()
            .chain(task_xml(s, &user, desktop).encode_utf16())
            .flat_map(u16::to_le_bytes)
            .collect();
        file.write_all(&data).map_err(|e| e.to_string())?;
        run(scheduler()
            .args(["/Create", "/F", "/TN", &task_name(s, desktop), "/XML"])
            .arg(file.path()))?;
    }
    let mut link =
        mslnk::ShellLink::new(s.binary("switchyard-desktop")).map_err(|e| e.to_string())?;
    link.set_arguments(Some(quote_arg(&s.desktop_settings().display().to_string())));
    for path in shortcut_paths(s).into_iter().take(1) {
        fs::create_dir_all(path.parent().ok_or("Shortcut has no directory.")?)
            .map_err(|e| e.to_string())?;
        link.create_lnk(path).map_err(|e| e.to_string())?;
    }
    for desktop in [false, true] {
        run(scheduler().args(["/Run", "/TN", &task_name(s, desktop)]))?;
    }
    Ok(())
}
fn stop_task(s: &Settings, desktop: bool) -> Result<(), String> {
    if exists(s, desktop)? {
        // /End reports an error for a task that has already exited. Path-checked handles below verify shutdown.
        let _ = scheduler()
            .args(["/End", "/TN", &task_name(s, desktop)])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
// Full executable paths distinguish this installation from other copies running under the same user.
fn stop_images(images: &[PathBuf]) -> Result<(), String> {
    let targets = images
        .iter()
        .filter(|p| p.exists())
        .map(fs::canonicalize)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(error());
        }
        let snapshot = Handle(snapshot);
        let mut entry: PROCESSENTRY32W = std::mem::zeroed();
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut found = Process32FirstW(snapshot.0, &mut entry);
        while found != 0 {
            if entry.th32ProcessID != std::process::id() {
                let process = OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | PROCESS_SYNCHRONIZE,
                    0,
                    entry.th32ProcessID,
                );
                if !process.is_null() {
                    let process = Handle(process);
                    let mut buffer = vec![0u16; 32768];
                    let mut length = buffer.len() as u32;
                    if QueryFullProcessImageNameW(process.0, 0, buffer.as_mut_ptr(), &mut length)
                        != 0
                    {
                        let path = PathBuf::from(std::ffi::OsString::from_wide(
                            &buffer[..length as usize],
                        ));
                        if fs::canonicalize(path).is_ok_and(|p| targets.iter().any(|t| t == &p)) {
                            if TerminateProcess(process.0, 0) == 0
                                && WaitForSingleObject(process.0, 0) != WAIT_OBJECT_0
                            {
                                return Err(error());
                            }
                            if WaitForSingleObject(process.0, 10000) != WAIT_OBJECT_0 {
                                return Err(
                                    "A managed Windows process did not stop within ten seconds."
                                        .into(),
                                );
                            }
                        }
                    }
                }
            }
            found = Process32NextW(snapshot.0, &mut entry);
        }
    }
    Ok(())
}
pub fn stop(s: &Settings) -> Result<(), String> {
    for desktop in [false, true] {
        stop_task(s, desktop)?;
    }
    stop_images(&[
        s.binary("switchyard-server"),
        s.binary("switchyard-desktop"),
        s.binary("switchyard-desktop-install"),
    ])
}
pub fn restart(s: &Settings) -> Result<(), String> {
    stop_task(s, false)?;
    stop_images(&[s.binary("switchyard-server")])?;
    run(scheduler().args(["/Run", "/TN", &task_name(s, false)]))
}
pub fn uninstall(s: &Settings) -> Result<(), String> {
    for desktop in [false, true] {
        if exists(s, desktop)? {
            run(scheduler().args(["/Delete", "/F", "/TN", &task_name(s, desktop)]))?;
        }
    }
    for path in shortcut_paths(s) {
        remove(&path)?;
    }
    // Config and history live in SY_HOME. Only the managed application directory is removed.
    if s.bin_dir().exists() {
        fs::remove_dir_all(s.bin_dir()).map_err(|e| e.to_string())?;
    }
    Ok(())
}
/// This function stages all binaries and attempts to restore earlier replacements if one fails.
pub fn replace_binaries(s: &Settings) -> Result<(), String> {
    let names = [
        "switchyard-server",
        "switchyard-desktop",
        "switchyard-desktop-install",
    ];
    fs::create_dir_all(s.bin_dir()).map_err(|e| e.to_string())?;
    let stage = tempfile::tempdir_in(s.bin_dir()).map_err(|e| e.to_string())?;
    for name in names {
        fs::copy(s.release(name), stage.path().join(executable_name(name)))
            .map_err(|e| e.to_string())?;
    }
    let mut changed: Vec<(&str, bool)> = Vec::new();
    for name in names {
        let dest = s.binary(name);
        let previous = stage.path().join(format!("{name}.previous"));
        let existed = dest.exists();
        let result = (|| {
            if existed {
                fs::rename(&dest, &previous).map_err(|e| e.to_string())?;
            }
            changed.push((name, existed));
            fs::rename(stage.path().join(executable_name(name)), &dest).map_err(|e| e.to_string())
        })();
        if let Err(error) = result {
            let mut failures = Vec::new();
            for (name, existed) in changed.into_iter().rev() {
                let restored = (|| {
                    remove(&s.binary(name))?;
                    if existed {
                        fs::rename(
                            stage.path().join(format!("{name}.previous")),
                            s.binary(name),
                        )
                        .map_err(|e| e.to_string())?;
                    }
                    Ok::<(), String>(())
                })();
                if let Err(restore) = restored {
                    failures.push(format!("{name}: {restore}"));
                }
            }
            if !failures.is_empty() {
                // Failed restoration must retain the previous files for manual recovery.
                let recovery = stage.keep();
                return Err(format!(
                    "Could not replace app files: {error}. Restore failures: {}. Previous files remain at {}.",
                    failures.join("; "),
                    recovery.display()
                ));
            }
            return Err(format!(
                "Could not replace app files; restored the previous binaries: {error}"
            ));
        }
    }
    Ok(())
}
/// The updater runs from a temporary copy because Windows locks running executables against replacement.
pub fn temporary_updater(source: &Path) -> Result<tempfile::TempPath, String> {
    let file = tempfile::Builder::new()
        .prefix("switchyard-update-")
        .suffix(".exe")
        .tempfile()
        .map_err(|e| e.to_string())?;
    fs::copy(source, file.path()).map_err(|e| e.to_string())?;
    Ok(file.into_temp_path())
}
pub fn wait_parent(pid: u32) -> Result<(), String> {
    use windows_sys::Win32::Foundation::{ERROR_INVALID_PARAMETER, GetLastError};
    if pid == 0 {
        return Err("Invalid parent process ID.".into());
    }
    unsafe {
        let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if handle.is_null() {
            return if GetLastError() == ERROR_INVALID_PARAMETER {
                Ok(())
            } else {
                Err(error())
            };
        }
        let handle = Handle(handle);
        if WaitForSingleObject(handle.0, 30000) != WAIT_OBJECT_0 {
            return Err("The installer parent did not exit within thirty seconds.".into());
        }
    }
    Ok(())
}
/// This function waits for its child so Task Scheduler tracks the app or server's lifetime.
pub fn launch_managed(s: &Settings, desktop: bool) -> Result<(), String> {
    fs::create_dir_all(s.sy_home.join("logs")).map_err(|e| e.to_string())?;
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(s.sy_home.join(if desktop {
            "logs/desktop.log"
        } else {
            "logs/server.log"
        }))
        .map_err(|e| e.to_string())?;
    let mut c = Command::new(s.binary(if desktop {
        "switchyard-desktop"
    } else {
        "switchyard-server"
    }));
    hide_console(&mut c);
    c.env("HOME", &s.home)
        .env("USERPROFILE", &s.home)
        .env("CODEX_HOME", &s.codex_home)
        .env("SY_HOME", &s.sy_home)
        .current_dir(s.bin_dir());
    if desktop {
        c.arg(s.desktop_settings());
    } else {
        c.arg("--config")
            .arg(s.sy_home.join("composite.toml"))
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &s.port.to_string(),
                "--routing-log-file",
            ])
            .arg(s.sy_home.join("routing.jsonl"));
    }
    let status = c
        .stdin(Stdio::null())
        .stderr(log.try_clone().map_err(|e| e.to_string())?)
        .stdout(log)
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Managed application exited with {status}."))
    }
}
pub fn open(value: &str) -> Result<(), String> {
    use windows_sys::Win32::UI::{Shell::ShellExecuteW, WindowsAndMessaging::SW_SHOWNORMAL};
    let value = wide(std::ffi::OsStr::new(value));
    let verb = wide(std::ffi::OsStr::new("open"));
    let status = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            value.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    } as isize;
    if status > 32 {
        Ok(())
    } else {
        Err(format!(
            "Windows could not open the file or URL (error {status})."
        ))
    }
}
pub fn saved_key(base_url: &str) -> Result<Option<String>, String> {
    let target = wide(std::ffi::OsStr::new(&format!(
        "Switchyard:model-list:{base_url}"
    )));
    unsafe {
        let mut credential: *mut CREDENTIALW = std::ptr::null_mut();
        if CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) == 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(ERROR_NOT_FOUND as i32) {
                return Ok(None);
            }
            return Err(error());
        }
        let credential_ref = &*credential;
        let bytes = if credential_ref.CredentialBlobSize == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(
                credential_ref.CredentialBlob,
                credential_ref.CredentialBlobSize as usize,
            )
            .to_vec()
        };
        CredFree(credential.cast());
        String::from_utf8(bytes)
            .map(|s| Some(s).filter(|s| !s.trim().is_empty()))
            .map_err(|e| e.to_string())
    }
}
pub fn save_key(base_url: &str, key: &str) -> Result<(), String> {
    let mut target = wide(std::ffi::OsStr::new(&format!(
        "Switchyard:model-list:{base_url}"
    )));
    let mut bytes = key.trim().as_bytes().to_vec();
    let mut credential: CREDENTIALW = unsafe { std::mem::zeroed() };
    credential.Type = CRED_TYPE_GENERIC;
    credential.TargetName = target.as_mut_ptr();
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE;
    credential.CredentialBlobSize = bytes.len().try_into().map_err(|_| "The key is too long.")?;
    credential.CredentialBlob = bytes.as_mut_ptr();
    if unsafe { CredWriteW(&credential, 0) } == 0 {
        Err(error())
    } else {
        Ok(())
    }
}

pub fn record_handoff(s: &Settings) -> Result<(), String> {
    let _lock = update_lock(s)?;
    set_update_state(
        s,
        "starting",
        "start",
        "Starting the Windows source installer.",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tasks_follow_the_installed_app_directory() {
        let mut settings = Settings::from_env().expect("settings");
        settings.app_dir = Some(PathBuf::from(r"C:\Users\Team\Apps\Switchyard"));
        let names = [task_name(&settings, false), task_name(&settings, true)];
        assert_ne!(names[0], names[1]);
        settings.sy_home = PathBuf::from(r"C:\Users\Team\New settings");
        for path in [
            r"c:/users/team/apps/switchyard/",
            r"C:\Users\Team\Apps\.\Other\..\Switchyard",
            r"\\?\C:\Users\Team\Apps\Switchyard",
        ] {
            settings.app_dir = Some(PathBuf::from(path));
            assert_eq!(
                [task_name(&settings, false), task_name(&settings, true)],
                names
            );
        }
        settings.app_dir = Some(PathBuf::from(r"C:\Users\Team\Apps\Another"));
        assert_ne!(task_name(&settings, false), names[0]);
        settings.app_dir = Some(PathBuf::from(r"\\server\share\Apps\Switchyard"));
        let unc = task_name(&settings, false);
        settings.app_dir = Some(PathBuf::from(r"\\?\UNC\SERVER\SHARE\Apps\Switchyard"));
        assert_eq!(task_name(&settings, false), unc);
    }
    #[test]
    fn windows_arguments_escape_quotes_and_trailing_slashes() {
        assert_eq!(quote_arg("C:\\équipe space\\"), "\"C:\\équipe space\\\\\"");
        assert_eq!(quote_arg("a\"b"), "\"a\\\"b\"");
    }
}
