// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Platform controls that make checker process and filesystem evidence fail closed.

use std::ffi::OsStr;
use std::fs::{File, OpenOptions};
use std::path::Path;

use tokio::process::{Child, Command};

#[cfg(unix)]
mod imp {
    use std::os::unix::ffi::OsStrExt as _;
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};

    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct MutationStamp {
        inode: u64,
        size: u64,
        ctime: i64,
        ctime_nsec: i64,
        mode: u32,
    }

    pub(crate) fn is_directory(metadata: &std::fs::Metadata) -> bool {
        metadata.file_type().is_dir()
    }

    pub(crate) fn is_regular_file(metadata: &std::fs::Metadata) -> bool {
        metadata.file_type().is_file()
    }

    pub(crate) fn mutation_stamp_path(
        _path: &Path,
        metadata: &std::fs::Metadata,
    ) -> std::io::Result<MutationStamp> {
        Ok(mutation_stamp(metadata))
    }

    pub(crate) fn mutation_stamp_file(file: &File) -> std::io::Result<MutationStamp> {
        Ok(mutation_stamp(&file.metadata()?))
    }

    fn mutation_stamp(metadata: &std::fs::Metadata) -> MutationStamp {
        MutationStamp {
            inode: metadata.ino(),
            size: metadata.size(),
            ctime: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
            mode: metadata.mode(),
        }
    }

    pub(crate) fn open_regular_file(path: &Path) -> std::io::Result<File> {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(path)
    }

    pub(crate) fn os_str_bytes(value: &OsStr) -> Vec<u8> {
        value.as_bytes().to_vec()
    }

    pub(crate) fn configure_process_tree(command: &mut Command) {
        command.process_group(0);
    }

    /// Signals a checker run's whole process group when the run goes out of scope.
    pub(crate) struct ProcessTreeReaper(Option<u32>);

    impl ProcessTreeReaper {
        pub(crate) fn attach(child: &Child) -> std::io::Result<Self> {
            Ok(Self(child.id()))
        }
    }

    impl Drop for ProcessTreeReaper {
        fn drop(&mut self) {
            let Some(group) = self.0 else {
                return;
            };
            if let Err(error) = kill_process_group(group) {
                tracing::warn!(
                    target: "libsy",
                    kind = ?error.kind(),
                    raw_os_error = error.raw_os_error(),
                    "vgr checker could not signal its process tree"
                );
            }
        }
    }

    /// Sends SIGKILL directly to a positive child process-group id.
    fn kill_process_group(group: u32) -> std::io::Result<()> {
        let group = i32::try_from(group)
            .ok()
            .filter(|group| *group > 0)
            .ok_or_else(|| std::io::Error::other("invalid checker process group"))?;
        // SAFETY: `libc::kill` dereferences no pointers. `group` is checked positive,
        // so negating it is representable and POSIX treats it as a process-group id.
        let result = unsafe { libc::kill(-group, libc::SIGKILL) };
        if result == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            return Ok(());
        }
        Err(error)
    }
}

#[cfg(windows)]
mod imp {
    use std::mem::size_of;
    use std::os::windows::ffi::OsStrExt as _;
    use std::os::windows::fs::{MetadataExt as _, OpenOptionsExt as _};
    use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};

    use windows_sys::Win32::Foundation::{ERROR_NO_MORE_FILES, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_REPARSE_POINT,
        FILE_BASIC_INFO, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
        FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, FileBasicInfo,
        GetFileInformationByHandle, GetFileInformationByHandleEx,
    };
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::{
        CREATE_SUSPENDED, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME,
    };

    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct MutationStamp {
        volume_serial: u32,
        file_index: u64,
        size: u64,
        creation_time: i64,
        change_time: i64,
        last_write_time: i64,
        attributes: u32,
        links: u32,
    }

    pub(crate) fn is_directory(metadata: &std::fs::Metadata) -> bool {
        metadata.file_type().is_dir()
            && metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0
    }

    pub(crate) fn is_regular_file(metadata: &std::fs::Metadata) -> bool {
        metadata.file_type().is_file()
            && metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT == 0
    }

    pub(crate) fn mutation_stamp_path(
        path: &Path,
        _metadata: &std::fs::Metadata,
    ) -> std::io::Result<MutationStamp> {
        let file = OpenOptions::new()
            .access_mode(0)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)?;
        mutation_stamp_file(&file)
    }

    pub(crate) fn mutation_stamp_file(file: &File) -> std::io::Result<MutationStamp> {
        let handle = file.as_raw_handle().cast();
        let mut identity = BY_HANDLE_FILE_INFORMATION::default();
        let mut basic = FILE_BASIC_INFO::default();
        // SAFETY: both output pointers are valid for their declared sizes and the
        // borrowed file handle remains open for the duration of both calls.
        if unsafe { GetFileInformationByHandle(handle, &mut identity) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: `basic` is writable for exactly the size passed, and `handle`
        // is the same live borrowed handle validated above.
        if unsafe {
            GetFileInformationByHandleEx(
                handle,
                FileBasicInfo,
                (&raw mut basic).cast(),
                u32::try_from(size_of::<FILE_BASIC_INFO>())
                    .map_err(|_| std::io::Error::other("Windows file metadata size overflow"))?,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        if identity.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "checker snapshot entry is a Windows reparse point",
            ));
        }
        Ok(MutationStamp {
            volume_serial: identity.dwVolumeSerialNumber,
            file_index: (u64::from(identity.nFileIndexHigh) << 32)
                | u64::from(identity.nFileIndexLow),
            size: (u64::from(identity.nFileSizeHigh) << 32) | u64::from(identity.nFileSizeLow),
            creation_time: basic.CreationTime,
            change_time: basic.ChangeTime,
            last_write_time: basic.LastWriteTime,
            attributes: basic.FileAttributes,
            links: identity.nNumberOfLinks,
        })
    }

    pub(crate) fn open_regular_file(path: &Path) -> std::io::Result<File> {
        let file = OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)?;
        let stamp = mutation_stamp_file(&file)?;
        if stamp.attributes & FILE_ATTRIBUTE_DIRECTORY != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "checker snapshot entry is not a regular file",
            ));
        }
        Ok(file)
    }

    pub(crate) fn os_str_bytes(value: &OsStr) -> Vec<u8> {
        value
            .encode_wide()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    }

    pub(crate) fn configure_process_tree(command: &mut Command) {
        // The child cannot execute or spawn descendants before it is attached
        // to the kill-on-close Job Object and its primary thread is resumed.
        command.creation_flags(CREATE_SUSPENDED);
    }

    /// Owns the kill-on-close Job Object containing one checker process tree.
    pub(crate) struct ProcessTreeReaper {
        job: OwnedHandle,
    }

    impl ProcessTreeReaper {
        pub(crate) fn attach(child: &Child) -> std::io::Result<Self> {
            let job = create_kill_on_close_job()?;
            let process = child
                .raw_handle()
                .ok_or_else(|| std::io::Error::other("checker process handle is unavailable"))?
                .cast();
            // SAFETY: `job` and `process` are live kernel handles. The child is
            // still suspended, so it cannot create an unassigned descendant.
            if unsafe { AssignProcessToJobObject(raw_handle(&job), process) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
            resume_primary_thread(
                child
                    .id()
                    .ok_or_else(|| std::io::Error::other("checker process id is unavailable"))?,
            )?;
            Ok(Self { job })
        }
    }

    impl Drop for ProcessTreeReaper {
        fn drop(&mut self) {
            // KILL_ON_JOB_CLOSE is the fallback if this explicit termination
            // fails; closing the owned handle immediately follows this method.
            if unsafe { TerminateJobObject(raw_handle(&self.job), 1) } == 0 {
                let error = std::io::Error::last_os_error();
                tracing::warn!(
                    target: "libsy",
                    kind = ?error.kind(),
                    raw_os_error = error.raw_os_error(),
                    "vgr checker could not signal its process tree"
                );
            }
        }
    }

    fn create_kill_on_close_job() -> std::io::Result<OwnedHandle> {
        // SAFETY: null security attributes and name request an unnamed Job
        // Object with default security and return a newly owned handle.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        let job = owned_handle(handle)?;
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `limits` has the layout and exact byte size required by the
        // selected information class, and `job` remains open for the call.
        if unsafe {
            SetInformationJobObject(
                raw_handle(&job),
                JobObjectExtendedLimitInformation,
                (&raw const limits).cast(),
                u32::try_from(size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>())
                    .map_err(|_| std::io::Error::other("Windows Job Object size overflow"))?,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    fn resume_primary_thread(process_id: u32) -> std::io::Result<()> {
        // SAFETY: the call returns a newly owned snapshot handle.
        let snapshot = owned_handle(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) })?;
        let entry_size = u32::try_from(size_of::<THREADENTRY32>())
            .map_err(|_| std::io::Error::other("Windows thread entry size overflow"))?;
        let mut entry = THREADENTRY32 {
            dwSize: entry_size,
            ..THREADENTRY32::default()
        };
        // SAFETY: `entry` is initialized with the required size and remains
        // writable while the live snapshot is enumerated.
        if unsafe { Thread32First(raw_handle(&snapshot), &mut entry) } == 0 {
            return toolhelp_end_or_error("checker primary thread was not found");
        }
        loop {
            if entry.th32OwnerProcessID == process_id {
                // SAFETY: the enumerated thread id belongs to the suspended
                // child; the returned handle is newly owned.
                let thread = owned_handle(unsafe {
                    OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID)
                })?;
                // SAFETY: `thread` grants THREAD_SUSPEND_RESUME and is live.
                let previous_count = unsafe { ResumeThread(raw_handle(&thread)) };
                if previous_count == u32::MAX {
                    return Err(std::io::Error::last_os_error());
                }
                if previous_count != 1 {
                    return Err(std::io::Error::other(format!(
                        "checker primary thread had unexpected suspend count {previous_count}"
                    )));
                }
                return Ok(());
            }
            // ToolHelp structures carry their size on every call; reset it in
            // case the previous API invocation changed the output structure.
            entry.dwSize = entry_size;
            // SAFETY: same initialized entry and live snapshot as above.
            if unsafe { Thread32Next(raw_handle(&snapshot), &mut entry) } == 0 {
                return toolhelp_end_or_error("checker primary thread was not found");
            }
        }
    }

    fn toolhelp_end_or_error(not_found: &'static str) -> std::io::Result<()> {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(ERROR_NO_MORE_FILES as i32) {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, not_found))
        } else {
            Err(error)
        }
    }

    fn owned_handle(handle: HANDLE) -> std::io::Result<OwnedHandle> {
        if handle.is_null() || handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: callers pass only newly owned handles after rejecting both
        // invalid sentinel values, transferring exactly one close obligation.
        Ok(unsafe { OwnedHandle::from_raw_handle(handle.cast()) })
    }

    fn raw_handle(handle: &OwnedHandle) -> HANDLE {
        handle.as_raw_handle().cast()
    }
}

pub(super) use imp::{
    MutationStamp, ProcessTreeReaper, configure_process_tree, is_directory, is_regular_file,
    mutation_stamp_file, mutation_stamp_path, open_regular_file, os_str_bytes,
};
