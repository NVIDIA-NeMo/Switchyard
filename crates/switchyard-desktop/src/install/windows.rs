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
        CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC, CREDENTIALW, CredDeleteW, CredFree,
        CredReadW, CredWriteW,
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

/// The managed executables, in the order they are replaced and stopped.
const BINARIES: [&str; 3] = [
    "switchyard-server",
    "switchyard-desktop",
    "switchyard-desktop-install",
];

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
/// Credential Manager reports a missing target as ERROR_NOT_FOUND; this returns `absent` for it.
fn absent_or_error<T>(absent: T) -> Result<T, String> {
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_NOT_FOUND as i32) {
        Ok(absent)
    } else {
        Err(error.to_string())
    }
}
pub fn hide_console(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

pub fn task_name(s: &Settings, desktop: bool) -> String {
    // The binary directory keeps task names stable when SY_HOME changes or a Rust toolchain changes.
    let mut directory = PathBuf::new();
    for component in s.bin_dir().components() {
        if component == std::path::Component::ParentDir {
            directory.pop();
        } else if component != std::path::Component::CurDir {
            directory.push(component);
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
/// This function runs `schtasks <action> /TN <task>` without output and reports whether it succeeded.
fn task_succeeds(s: &Settings, desktop: bool, action: &str) -> Result<bool, String> {
    scheduler()
        .args([action, "/TN", &task_name(s, desktop)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
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
        let result = match (0..256).find(|&index| *sid.add(index) == 0) {
            Some(length) => String::from_utf16(std::slice::from_raw_parts(sid, length))
                .map_err(|e| e.to_string()),
            None => Err("Invalid Windows account SID.".to_owned()),
        };
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
        "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\
         <Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\
         <Triggers><LogonTrigger><Enabled>true</Enabled>\
         <UserId>{user}</UserId></LogonTrigger></Triggers>\
         <Principals><Principal id=\"User\"><UserId>{user}</UserId>\
         <LogonType>InteractiveToken</LogonType>\
         <RunLevel>LeastPrivilege</RunLevel></Principal></Principals>\
         <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\
         <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\
         <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\
         <AllowHardTerminate>true</AllowHardTerminate><StartWhenAvailable>true</StartWhenAvailable>\
         <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>{restart}</Settings>\
         <Actions Context=\"User\"><Exec><Command>{}</Command><Arguments>{}</Arguments>\
         <WorkingDirectory>{}</WorkingDirectory></Exec></Actions>\
         </Task>",
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
        create_task(s, &user, desktop)?;
    }
    let mut link =
        mslnk::ShellLink::new(s.binary("switchyard-desktop")).map_err(|e| e.to_string())?;
    link.set_arguments(Some(quote_arg(&s.desktop_settings().display().to_string())));
    // The desktop logon task starts the app, so only the Start menu shortcut is created.
    let [start_menu, _startup] = shortcut_paths(s);
    fs::create_dir_all(&s.service_dir).map_err(|e| e.to_string())?;
    link.create_lnk(start_menu).map_err(|e| e.to_string())?;
    for desktop in [false, true] {
        run(scheduler().args(["/Run", "/TN", &task_name(s, desktop)]))?;
    }
    Ok(())
}
/// This function registers one task from its XML, written as UTF-16 to match the XML declaration.
fn create_task(s: &Settings, user: &str, desktop: bool) -> Result<(), String> {
    let mut file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    let data: Vec<u8> = [0xfeffu16]
        .into_iter()
        .chain(task_xml(s, user, desktop).encode_utf16())
        .flat_map(u16::to_le_bytes)
        .collect();
    file.write_all(&data).map_err(|e| e.to_string())?;
    // Closing the writer lets schtasks read the XML; TempPath keeps cleanup active during import.
    let path = file.into_temp_path();
    run(scheduler()
        .args(["/Create", "/F", "/TN", &task_name(s, desktop), "/XML"])
        .arg(&path))
}
fn stop_task(s: &Settings, desktop: bool) -> Result<(), String> {
    if task_succeeds(s, desktop, "/Query")? {
        // /End reports an error for a task that has already exited. Path-checked handles below verify shutdown.
        task_succeeds(s, desktop, "/End")?;
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
                stop_process(entry.th32ProcessID, &targets)?;
            }
            found = Process32NextW(snapshot.0, &mut entry);
        }
    }
    Ok(())
}
/// This function terminates process `pid` and waits for it to exit when its executable is one of
/// `targets`. Processes this user cannot open or query are skipped.
fn stop_process(pid: u32, targets: &[PathBuf]) -> Result<(), String> {
    let access = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | PROCESS_SYNCHRONIZE;
    let process = unsafe { OpenProcess(access, 0, pid) };
    if process.is_null() {
        return Ok(());
    }
    let process = Handle(process);
    let mut buffer = vec![0u16; 32768];
    let mut length = buffer.len() as u32;
    if unsafe { QueryFullProcessImageNameW(process.0, 0, buffer.as_mut_ptr(), &mut length) } == 0 {
        return Ok(());
    }
    let image = PathBuf::from(std::ffi::OsString::from_wide(&buffer[..length as usize]));
    if !fs::canonicalize(image).is_ok_and(|path| targets.contains(&path)) {
        return Ok(());
    }
    unsafe {
        if TerminateProcess(process.0, 0) == 0 && WaitForSingleObject(process.0, 0) != WAIT_OBJECT_0
        {
            return Err(error());
        }
        if WaitForSingleObject(process.0, 10000) != WAIT_OBJECT_0 {
            return Err("A managed Windows process did not stop within ten seconds.".into());
        }
    }
    Ok(())
}
pub fn stop(s: &Settings) -> Result<(), String> {
    for desktop in [false, true] {
        stop_task(s, desktop)?;
    }
    stop_images(&BINARIES.map(|name| s.binary(name)))
}
pub fn restart(s: &Settings) -> Result<(), String> {
    stop_task(s, false)?;
    stop_images(&[s.binary("switchyard-server")])?;
    run(scheduler().args(["/Run", "/TN", &task_name(s, false)]))
}
pub fn uninstall(s: &Settings) -> Result<(), String> {
    for desktop in [false, true] {
        if task_succeeds(s, desktop, "/Query")? {
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
    fs::create_dir_all(s.bin_dir()).map_err(|e| e.to_string())?;
    let stage = tempfile::tempdir_in(s.bin_dir()).map_err(|e| e.to_string())?;
    for name in BINARIES {
        fs::copy(s.release(name), stage.path().join(executable_name(name)))
            .map_err(|e| e.to_string())?;
    }
    let mut changed = Vec::new();
    let Err(error) = BINARIES
        .into_iter()
        .try_for_each(|name| swap(s, stage.path(), name, &mut changed))
    else {
        return Ok(());
    };
    let mut failures = Vec::new();
    for (name, saved) in changed.iter().rev() {
        if let Err(e) = restore(&s.binary(name), saved.as_deref()) {
            failures.push(format!("{name}: {e}"));
        }
    }
    if failures.is_empty() {
        return Err(format!(
            "Could not replace app files; restored the previous binaries: {error}"
        ));
    }
    // Failed restoration must retain the previous files for manual recovery.
    let recovery = stage.keep();
    Err(format!(
        "Could not replace app files: {error}. Restore failures: {}. \
         Previous files remain at {}.",
        failures.join("; "),
        recovery.display()
    ))
}
/// This function saves any installed `name` in `stage`, then moves the staged copy into place.
/// It records `name` and the saved copy as soon as the installed file has moved, so a failure
/// in the second move still restores it.
fn swap(
    s: &Settings,
    stage: &Path,
    name: &'static str,
    changed: &mut Vec<(&'static str, Option<PathBuf>)>,
) -> std::io::Result<()> {
    let dest = s.binary(name);
    let saved = dest
        .exists()
        .then(|| stage.join(format!("{name}.previous")));
    if let Some(saved) = &saved {
        fs::rename(&dest, saved)?;
    }
    changed.push((name, saved));
    fs::rename(stage.join(executable_name(name)), dest)
}
fn restore(dest: &Path, saved: Option<&Path>) -> Result<(), String> {
    remove(dest)?;
    match saved {
        Some(saved) => fs::rename(saved, dest).map_err(|e| e.to_string()),
        None => Ok(()),
    }
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
    let (program, log) = if desktop {
        ("switchyard-desktop", "logs/desktop.log")
    } else {
        ("switchyard-server", "logs/server.log")
    };
    fs::create_dir_all(s.sy_home.join("logs")).map_err(|e| e.to_string())?;
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(s.sy_home.join(log))
        .map_err(|e| e.to_string())?;
    let mut c = Command::new(s.binary(program));
    hide_console(&mut c);
    c.env("HOME", &s.home)
        .env("USERPROFILE", &s.home)
        .env("CODEX_HOME", &s.codex_home)
        .env("SY_HOME", &s.sy_home)
        .current_dir(s.bin_dir());
    if desktop {
        c.arg(s.desktop_settings());
    } else {
        let config = s.sy_home.join("composite.toml");
        c.arg("--config")
            .arg(&config)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &s.port.to_string(),
                "--routing-log-file",
            ])
            .arg(s.sy_home.join("routing.jsonl"));
        // The helper has no console, so a missing key is reported in the server log.
        credentials::configure(&mut c, &config).inspect_err(|error| {
            let _ = writeln!(&log, "switchyard-desktop-install: {error}");
        })?;
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
    read_credential(&format!("Switchyard:model-list:{base_url}"))
}

pub(crate) fn read_credential(target: &str) -> Result<Option<String>, String> {
    let target = wide(std::ffi::OsStr::new(target));
    let mut credential: *mut CREDENTIALW = std::ptr::null_mut();
    if unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) } == 0 {
        return absent_or_error(None);
    }
    let bytes = unsafe {
        let size = (*credential).CredentialBlobSize as usize;
        // An empty secret may have a null blob, which `from_raw_parts` does not accept.
        let bytes = if size == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts((*credential).CredentialBlob, size).to_vec()
        };
        CredFree(credential.cast());
        bytes
    };
    String::from_utf8(bytes)
        .map(|s| Some(s).filter(|s| !s.trim().is_empty()))
        .map_err(|e| e.to_string())
}
pub fn save_key(base_url: &str, key: &str) -> Result<(), String> {
    write_credential(&format!("Switchyard:model-list:{base_url}"), key.trim())
}

pub(crate) fn write_credential(target: &str, key: &str) -> Result<(), String> {
    let mut target = wide(std::ffi::OsStr::new(target));
    let credential = CREDENTIALW {
        Type: CRED_TYPE_GENERIC,
        TargetName: target.as_mut_ptr(),
        Persist: CRED_PERSIST_LOCAL_MACHINE,
        CredentialBlobSize: key.len().try_into().map_err(|_| "The key is too long.")?,
        // CredWriteW only reads the blob, so the key needs no mutable copy.
        CredentialBlob: key.as_ptr().cast_mut(),
        ..unsafe { std::mem::zeroed() }
    };
    if unsafe { CredWriteW(&credential, 0) } == 0 {
        return Err(error());
    }
    Ok(())
}

pub(crate) fn delete_credential(target: &str) -> Result<(), String> {
    let target = wide(std::ffi::OsStr::new(target));
    if unsafe { CredDeleteW(target.as_ptr(), CRED_TYPE_GENERIC, 0) } == 0 {
        return absent_or_error(());
    }
    Ok(())
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
