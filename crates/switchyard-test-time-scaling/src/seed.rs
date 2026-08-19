// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Repeatable seed helpers with no runtime state.

pub(crate) fn derive(root: u64, parts: &[u64]) -> u64 {
    parts.iter().fold(mix(root), |seed, part| mix(seed ^ part))
}

pub(crate) fn shuffle<T>(values: &mut [T], seed: u64) {
    let mut state = seed;
    for end in (1..values.len()).rev() {
        state = mix(state);
        let index = (state as usize) % (end + 1);
        values.swap(index, end);
    }
}

fn mix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
