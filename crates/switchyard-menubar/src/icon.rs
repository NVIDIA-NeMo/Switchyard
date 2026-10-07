// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The menu bar glyph: a track that splits in two, drawn in code.
//!
//! macOS scales the image to 18 points tall, so it is drawn at 2x for retina
//! displays. It is a template image, so only the alpha channel matters and the
//! system recolors it for light and dark menu bars.

/// Side length of the generated image, in pixels.
pub const SIZE: u32 = 36;

/// Half the stroke width, in pixels.
const HALF_STROKE: f64 = 1.7;

const SEGMENTS: [((f64, f64), (f64, f64)); 3] = [
    ((5.0, 18.0), (17.0, 18.0)),
    ((17.0, 18.0), (30.0, 7.0)),
    ((17.0, 18.0), (30.0, 29.0)),
];

/// How much of its opacity the glyph keeps while the server does not answer.
/// The fainter glyph shows at a glance that the server is down.
const STOPPED_OPACITY: f64 = 0.35;

/// Renders the glyph as RGBA pixels, row-major from the top left. It is faint
/// when `running` is false.
pub fn glyph(running: bool) -> Vec<u8> {
    let opacity = if running { 1.0 } else { STOPPED_OPACITY };
    let mut pixels = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let point = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let distance = SEGMENTS
                .iter()
                .map(|(start, end)| distance_to_segment(point, *start, *end))
                .fold(f64::INFINITY, f64::min);
            // One pixel of falloff past the stroke edge softens the diagonals.
            let coverage = (HALF_STROKE + 0.5 - distance).clamp(0.0, 1.0);
            pixels.extend_from_slice(&[0, 0, 0, (coverage * opacity * 255.0).round() as u8]);
        }
    }
    pixels
}

fn distance_to_segment(point: (f64, f64), start: (f64, f64), end: (f64, f64)) -> f64 {
    let (dx, dy) = (end.0 - start.0, end.1 - start.1);
    let length_squared = dx * dx + dy * dy;
    let along = ((point.0 - start.0) * dx + (point.1 - start.1) * dy) / length_squared;
    let along = along.clamp(0.0, 1.0);
    let (cx, cy) = (start.0 + dx * along, start.1 + dy * along);
    ((point.0 - cx).powi(2) + (point.1 - cy).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha(pixels: &[u8], x: u32, y: u32) -> u8 {
        pixels[((y * SIZE + x) * 4 + 3) as usize]
    }

    #[test]
    fn draws_a_trunk_that_splits_in_two() {
        let pixels = glyph(true);

        assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
        assert_eq!(alpha(&pixels, 10, 18), 255, "the trunk is solid");
        assert!(alpha(&pixels, 29, 7) > 0, "upper branch reaches its end");
        assert!(alpha(&pixels, 29, 29) > 0, "lower branch reaches its end");
        assert_eq!(alpha(&pixels, 29, 18), 0, "nothing runs straight through");
        assert_eq!(alpha(&pixels, 0, 0), 0, "corners stay transparent");
    }

    #[test]
    fn a_server_that_does_not_answer_gets_a_fainter_glyph() {
        let (running, stopped) = (glyph(true), glyph(false));

        assert_eq!(running.len(), stopped.len());
        assert!(alpha(&stopped, 10, 18) > 0, "the shape stays");
        assert!(alpha(&stopped, 10, 18) < alpha(&running, 10, 18) / 2);
        assert_eq!(alpha(&stopped, 0, 0), 0, "corners stay transparent");
    }
}
