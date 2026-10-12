// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module exports app and tray icons from their PNG sources on any platform.

use image::{ImageEncoder, RgbaImage, codecs::png::PngEncoder, imageops::FilterType};
use std::{fs, path::Path};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read_source(path: &Path, size: u32) -> Result<RgbaImage> {
    let image = image::open(path)?.into_rgba8();
    if image.dimensions() != (size, size) {
        return Err(format!("{} must be {size} × {size} pixels", path.display()).into());
    }
    Ok(image)
}

fn png(image: &RgbaImage) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    PngEncoder::new(&mut bytes).write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgba8,
    )?;
    Ok(bytes)
}

fn representations(source: &RgbaImage) -> Result<Vec<(u32, Vec<u8>)>> {
    [16, 32, 64, 128, 256, 512, 1024]
        .into_iter()
        .map(|size| {
            let resized = image::imageops::resize(source, size, size, FilterType::Lanczos3);
            Ok((size, png(&resized)?))
        })
        .collect()
}

fn icns(images: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut bytes = b"icns\0\0\0\0".to_vec();
    // Retina entries name the logical size separately from their pixel size.
    for (index, tag) in [
        (0, b"icp4"),
        (1, b"icp5"),
        (1, b"ic11"),
        (2, b"ic12"),
        (3, b"ic07"),
        (4, b"ic08"),
        (4, b"ic13"),
        (5, b"ic09"),
        (5, b"ic14"),
        (6, b"ic10"),
    ] {
        let png = &images[index].1;
        bytes.extend(tag);
        bytes.extend(((png.len() + 8) as u32).to_be_bytes());
        bytes.extend(png);
    }
    let length = (bytes.len() as u32).to_be_bytes();
    bytes[4..8].copy_from_slice(&length);
    bytes
}

fn ico(images: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let images = &images[..5];
    let mut bytes = vec![0, 0, 1, 0, images.len() as u8, 0];
    let mut offset = 6 + 16 * images.len();
    for (size, png) in images {
        bytes.extend([*size as u8, *size as u8, 0, 0]);
        bytes.extend(1_u16.to_le_bytes());
        bytes.extend(32_u16.to_le_bytes());
        bytes.extend((png.len() as u32).to_le_bytes());
        bytes.extend((offset as u32).to_le_bytes());
        offset += png.len();
    }
    for (_, png) in images {
        bytes.extend(png);
    }
    bytes
}

/// This function checks or writes the macOS and Windows icon files and the tray's decoded RGBA pixels.
/// A check preserves all files and reports the command needed to rebuild a stale export.
pub fn export(directory: &Path, check: bool) -> Result<()> {
    let source = read_source(&directory.join("icon.png"), 1024)?;
    let tray_source = read_source(&directory.join("tray-source.png"), 1024)?;
    let tray = image::imageops::resize(&tray_source, 22, 22, FilterType::Lanczos3);
    let images = representations(&source)?;
    for (name, expected) in [
        ("icon.icns", icns(&images)),
        ("icon.ico", ico(&images)),
        ("tray.png", png(&tray)?),
        ("tray.rgba", tray.into_raw()),
    ] {
        let path = directory.join(name);
        if check {
            if fs::read(&path).ok().as_deref() != Some(expected.as_slice()) {
                return Err(format!("{} is stale; run cargo desktop-icons", path.display()).into());
            }
        } else {
            fs::write(path, expected)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_tray_sources_reject_before_writing_exports() -> Result<()> {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("icons/icon.png");
        for bytes in [
            None,
            Some(b"unreadable".to_vec()),
            Some(png(&RgbaImage::new(22, 22))?),
        ] {
            let directory = tempfile::tempdir()?;
            fs::copy(&source, directory.path().join("icon.png"))?;
            if let Some(bytes) = bytes {
                fs::write(directory.path().join("tray-source.png"), bytes)?;
            }
            assert!(export(directory.path(), false).is_err());
            for name in ["icon.icns", "icon.ico", "tray.png", "tray.rgba"] {
                assert!(!directory.path().join(name).exists());
            }
        }
        Ok(())
    }

    #[test]
    fn exports_all_sizes_and_rejects_a_stale_windows_icon_without_writing() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("icons");
        fs::copy(source.join("icon.png"), directory.path().join("icon.png"))?;
        fs::copy(
            source.join("tray-source.png"),
            directory.path().join("tray-source.png"),
        )?;
        export(directory.path(), false)?;
        export(directory.path(), true)?;
        let expected_tray = image::imageops::resize(
            &read_source(&source.join("tray-source.png"), 1024)?,
            22,
            22,
            FilterType::Lanczos3,
        );
        assert_eq!(
            image::open(directory.path().join("tray.png"))?.into_rgba8(),
            expected_tray
        );
        assert_eq!(
            fs::read(directory.path().join("tray.rgba"))?,
            expected_tray.into_raw()
        );
        let ico = fs::read(directory.path().join("icon.ico"))?;
        assert_eq!(u16::from_le_bytes([ico[4], ico[5]]), 5);
        for (index, size) in [16, 32, 64, 128, 256].into_iter().enumerate() {
            let entry = 6 + 16 * index;
            let offset = u32::from_le_bytes(ico[entry + 12..entry + 16].try_into()?) as usize;
            let length = u32::from_le_bytes(ico[entry + 8..entry + 12].try_into()?) as usize;
            assert_eq!(
                image::load_from_memory(&ico[offset..offset + length])?.width(),
                size
            );
        }
        let icns = fs::read(directory.path().join("icon.icns"))?;
        assert_eq!(
            u32::from_be_bytes(icns[4..8].try_into()?) as usize,
            icns.len()
        );
        let mut offset = 8;
        for size in [16, 32, 32, 64, 128, 256, 256, 512, 512, 1024] {
            let length = u32::from_be_bytes(icns[offset + 4..offset + 8].try_into()?) as usize;
            assert_eq!(
                image::load_from_memory(&icns[offset + 8..offset + length])?.width(),
                size
            );
            offset += length;
        }
        assert_eq!(offset, icns.len());
        fs::write(directory.path().join("icon.ico"), b"stale")?;
        assert!(
            export(directory.path(), true)
                .unwrap_err()
                .to_string()
                .contains("icon.ico is stale")
        );
        assert_eq!(fs::read(directory.path().join("icon.ico"))?, b"stale");
        Ok(())
    }
}
