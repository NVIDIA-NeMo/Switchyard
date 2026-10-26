// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn attach_console() {
    unsafe {
        windows_console(false);
    }
}
pub fn allocate_console() {
    unsafe {
        windows_console(true);
    }
}
unsafe fn windows_console(allocate: bool) {
    unsafe extern "system" {
        fn AllocConsole() -> i32;
        fn FreeConsole() -> i32;
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        if allocate {
            // The app detaches first because AllocConsole fails while a console is attached.
            // A separate console keeps the parent shell from reading the terminal app's input.
            FreeConsole();
            AllocConsole();
        } else {
            AttachConsole(u32::MAX);
        }
    }
}
