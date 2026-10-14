// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded image preparation and Dynamo video frame decoding.

use std::io::Cursor;
use std::sync::Arc;

use dynamo_multimodal::image::{
    decode::{DecodeLimits, decode_rgb_exif_transposed},
    resize::resize_lanczos_rgb,
};
use dynamo_multimodal::{
    MmError,
    frames::{VideoFrames, VideoTiming, encode_jpeg_frames},
    video_decode::{VideoDecodeLimits, decode_sampled, probe},
};
use image::{
    DynamicImage, ImageDecoder, ImageFormat, ImageReader, RgbImage, metadata::Orientation,
};
use tokio::sync::Semaphore;

use super::{MediaError, Result};

const MAX_DIMENSION: u32 = 16_384;
const MAX_ALLOC: u64 = 128 * 1024 * 1024;

type EncodedFrames = Vec<(f64, Vec<u8>)>;

pub(crate) async fn resize(
    bytes: Vec<u8>,
    edge: u32,
    slots: Arc<Semaphore>,
) -> Result<(&'static str, Vec<u8>)> {
    let permit = slots
        .acquire_owned()
        .await
        .map_err(|_| MediaError::Invalid("media worker closed"))?;
    tokio::task::spawn_blocking(move || {
        // Keep the permit in the worker even if the caller cancels its wait.
        let _permit = permit;
        // Adapted from Dynamo's ImageReaderBackend; see README.md for provenance.
        let mut reader = ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(image::ImageError::IoError)?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(MAX_DIMENSION);
        limits.max_image_height = Some(MAX_DIMENSION);
        limits.max_alloc = Some(MAX_ALLOC);
        reader.limits(limits);
        let format = reader
            .format()
            .ok_or(MediaError::Invalid("unsupported image format"))?;
        let mut decoder = reader.into_decoder()?;
        if decoder.total_bytes() > MAX_ALLOC {
            return Err(MediaError::Invalid("image exceeds decode allocation limit"));
        }
        let orientation = decoder.orientation()?;
        let decoded = if format == ImageFormat::Jpeg {
            drop(decoder);
            let (rgb, height, width) = decode_rgb_exif_transposed(
                &bytes,
                &DecodeLimits {
                    max_width: Some(MAX_DIMENSION as usize),
                    max_height: Some(MAX_DIMENSION as usize),
                    max_pixels: Some(MAX_ALLOC as usize / 3),
                },
            )
            .map_err(|_| MediaError::Invalid("image decoding failed"))?;
            DynamicImage::ImageRgb8(
                RgbImage::from_raw(width as u32, height as u32, rgb)
                    .ok_or(MediaError::Invalid("invalid decoded image dimensions"))?,
            )
        } else {
            // Retain transparency and higher bit depths supported by the image crate.
            let mut decoded = DynamicImage::from_decoder(decoder)?;
            decoded.apply_orientation(orientation);
            decoded
        };
        if decoded.width() <= edge
            && decoded.height() <= edge
            && orientation == Orientation::NoTransforms
        {
            return Ok((format.to_mime_type(), bytes));
        }
        let resized = if decoded.width() > edge || decoded.height() > edge {
            if decoded.color().has_alpha() {
                // Dynamo's RGB resampler cannot preserve transparency.
                decoded.resize(edge, edge, image::imageops::FilterType::Triangle)
            } else {
                let rgb = decoded.into_rgb8();
                let (width, height) = rgb.dimensions();
                let scale = f64::from(edge) / f64::from(width.max(height));
                let out_width = (f64::from(width) * scale).round().max(1.0) as u32;
                let out_height = (f64::from(height) * scale).round().max(1.0) as u32;
                let pixels = resize_lanczos_rgb(
                    rgb.as_raw(),
                    height as usize,
                    width as usize,
                    out_height as usize,
                    out_width as usize,
                );
                DynamicImage::ImageRgb8(
                    RgbImage::from_raw(out_width, out_height, pixels)
                        .ok_or(MediaError::Invalid("invalid resized image dimensions"))?,
                )
            }
        } else {
            decoded
        };
        let mut output = Cursor::new(Vec::new());
        let mime = if resized.color().has_alpha() {
            resized.write_to(&mut output, ImageFormat::Png)?;
            "image/png"
        } else {
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85)
                .encode_image(&resized.to_rgb8())?;
            "image/jpeg"
        };
        Ok((mime, output.into_inner()))
    })
    .await
    .map_err(|_| MediaError::Invalid("image worker failed"))?
}

pub(crate) async fn frames(
    bytes: Vec<u8>,
    count: usize,
    edge: u32,
    slots: Arc<Semaphore>,
    remaining_bytes: usize,
) -> Result<EncodedFrames> {
    let permit = slots
        .acquire_owned()
        .await
        .map_err(|_| MediaError::Invalid("media worker closed"))?;
    tokio::task::spawn_blocking(move || {
        // Retain the worker permit even if the request deadline cancels the wait.
        let _permit = permit;
        dynamo_frames(&bytes, count, edge, remaining_bytes)
    })
    .await
    .map_err(|_| MediaError::Invalid("video worker failed"))?
}

fn dynamo_frames(
    bytes: &[u8],
    count: usize,
    edge: u32,
    mut remaining_bytes: usize,
) -> Result<EncodedFrames> {
    let mut limits = VideoDecodeLimits::default();
    limits.max_width = MAX_DIMENSION;
    limits.max_height = MAX_DIMENSION;
    limits.max_output_bytes = MAX_ALLOC as usize;
    let decoded = match probe(bytes)
        .and_then(|info| decode_sampled(bytes, (count as u64).min(info.frame_count), &limits))
    {
        Ok(frames) => frames,
        Err(MmError::Unsupported { .. }) => {
            return Err(MediaError::Invalid(
                "video codec, format, or decoder is unsupported",
            ));
        }
        Err(MmError::LimitExceeded { .. }) => {
            return Err(MediaError::Invalid("video exceeds decode allocation limit"));
        }
        Err(_) => return Err(MediaError::Invalid("video decoding failed")),
    };
    let (width, height) = (decoded.width(), decoded.height());
    let prepared = if width.max(height) > edge as usize {
        let scale = f64::from(edge) / width.max(height) as f64;
        let out_width = (width as f64 * scale).round().max(1.0) as usize;
        let out_height = (height as f64 * scale).round().max(1.0) as usize;
        let rgb = decoded
            .rgb()
            .chunks_exact(width * height * 3)
            .flat_map(|pixels| resize_lanczos_rgb(pixels, height, width, out_height, out_width))
            .collect();
        VideoFrames::new(
            out_width,
            out_height,
            rgb,
            VideoTiming {
                timestamps: decoded.timestamps().to_vec(),
                source_fps: decoded.source_fps(),
                source_start: decoded.source_start(),
                source_duration: decoded.source_duration(),
            },
        )
        .map_err(|_| MediaError::Invalid("video frame resizing failed"))?
    } else {
        decoded
    };
    let frames = encode_jpeg_frames(&prepared, 85)
        .map_err(|_| MediaError::Invalid("video frame encoding failed"))?;
    frames
        .into_iter()
        .map(|frame| {
            remaining_bytes =
                remaining_bytes
                    .checked_sub(frame.bytes.len())
                    .ok_or(MediaError::Invalid(
                        "prepared media exceeds max_output_bytes",
                    ))?;
            Ok((frame.timestamp_secs, frame.bytes))
        })
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rgb_resize_keeps_thin_images_nonempty_and_does_not_upscale() {
        let slots = Arc::new(Semaphore::new(1));
        for (format, width, height, expected) in [ImageFormat::Png, ImageFormat::Jpeg]
            .into_iter()
            .flat_map(|format| {
                [(80, 1, (20, 1)), (1, 80, (1, 20)), (8, 4, (8, 4))]
                    .into_iter()
                    .map(move |(w, h, expected)| (format, w, h, expected))
            })
        {
            let image = RgbImage::from_pixel(width, height, image::Rgb([210, 40, 20]));
            let mut input = Cursor::new(Vec::new());
            image.write_to(&mut input, format).unwrap();
            let original = input.into_inner();
            let (_, output) = resize(original.clone(), 20, slots.clone()).await.unwrap();
            let resized = image::load_from_memory(&output).unwrap();
            assert_eq!((resized.width(), resized.height()), expected);
            let pixel = resized.to_rgb8().get_pixel(0, 0).0;
            assert!(pixel[0] > 190 && pixel[1] < 60 && pixel[2] < 40);
            if width <= 20 && height <= 20 {
                assert_eq!(output, original);
            }
        }
    }

    #[test]
    fn malformed_videos_are_rejected() {
        for bytes in [
            b"not a video".as_slice(),
            b"\0\0\0\x18ftypisom\0\0\0\0isom".as_slice(),
        ] {
            let result = dynamo_frames(bytes, 1, 32, 1024);
            assert!(
                matches!(&result, Err(MediaError::Invalid("video decoding failed"))),
                "{result:?}"
            );
        }
    }
}
