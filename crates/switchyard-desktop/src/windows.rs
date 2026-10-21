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
    // The GUI has no console; terminal mode and harness jobs allocate one only when needed.
    unsafe extern "system" {
        fn AllocConsole() -> i32;
        fn FreeConsole() -> i32;
        fn AttachConsole(pid: u32) -> i32;
    }
    unsafe {
        if allocate {
            // Detach from a parent console first; AllocConsole fails while one is attached,
            // and a shared console lets the parent shell read the terminal app's input.
            FreeConsole();
            AllocConsole();
        } else {
            AttachConsole(u32::MAX);
        }
    }
}
