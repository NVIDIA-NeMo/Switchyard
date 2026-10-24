# Desktop app icon

The app icon shows a raised titanium railway turnout with pale-blue supports on
a navy beveled tile. It represents Switchyard routing requests between models.

`icon.png` is the 1024 × 1024 RGBA source with a transparent tile exterior.
`icon.icns` contains macOS sizes from 16 to 1024 pixels. `icon.ico` contains
Windows sizes of 16, 32, 64, 128, and 256 pixels.

`tray-source.png` contains the railway turnout on a transparent background,
enlarged to fill the tray icon without the navy tile or its frame. The desktop
tray uses a 22 × 22 pixel color copy of this separate source on macOS and
Windows. `tray.png` is the generated preview, and `tray.rgba` contains the same
decoded pixels. macOS displays the icon's original colors.

## Generation

An image-generation edit extracted `tray-source.png` from the app artwork. The
edit removed the navy tile and frame, kept the titanium turnout and pale-blue
supports, and enlarged the mark on a transparent canvas. The source was then
resized to 1024 × 1024 pixels before export.

Gemini 3 Pro Image generated the selected artwork through Inference Hub with
`gcp/google/gemini-3-pro-image`, `n: 1`, and `size: "1024x1024"`.
The selected image used this exact prompt:

```text
Create a refined Switchyard macOS app icon based on this visual description of the selected reference: raised silver titanium paired railway rails, ONE right-hand turnout with one straight continuing main route and one gradually curved branch, dark navy beveled rounded-square tile, restrained cool-blue accents. Generate one new polished 1024x1024 icon refinement. The reference has finely brushed silver rail top faces, charcoal metal rail webs, broad pale-blue base plates around the switch, and a small cool-blue inset near the central frog, all mounted on a textured midnight tile with a thick rounded beveled frame. Improve its clipping and mixed perspective. Reduce and recenter the ENTIRE rail motif so EVERY part, including end faces, foundations and shadows, is safely inside the tile. Leave at least 140 pixels of empty navy tile between EVERY rail end and the nearest tile edge; no rail reaches a tile rim or image boundary. Rail motif bounding box approximately x220..804 and y210..804, tile occupies x70..954 and y70..954. ALL rail endpoints clearly visible and finished, not cropped. Use ONE coherent near-front orthographic camera for the whole object: tile surface, rail top faces, metal side faces, gauge spacing and shadows all follow the same projection. Straight main rail pair stays straight and parallel. Branch rail pair separates gradually beginning tangent, with matching constant gauge along the curve and continuous plausible switch/frog junction. No extra branches, crossings or disconnected rail fragments. Maintain realistic physical extrusion, titanium bevels, softly grounded contact shadows, and large readable shapes. NO text, logos, arrows, trains, scenery, decorative stars or circuit patterns. Simple professional app icon, plain white exterior, no checkerboard. Change only the subtle material, bevel or lighting treatment specified below while keeping the reference turnout concept and corrected generous margins. Variant: bright polished titanium upper bevels with satin silver top faces and graphite side faces; cool pale blue support inlay, frontal orthographic camera, restrained upper-left soft key light. Moderate visible metal thickness and very clean machining. Rail motif small enough for generous interior margins.
```

## Export

1. Preserve the generated image before export.
2. Isolate the navy tile from its white exterior. Fill enclosed mask regions so
   the rails and supports stay opaque, then soften the outer mask by 0.35 pixels.
3. Save the result as a 1024 × 1024 RGBA PNG, keeping the artwork inside the tile
   unchanged.
4. Keep `tray-source.png` as a separate 1024 × 1024 RGBA image. Its transparent
   background includes the gaps between rails and supports. Check the 22 × 22
   pixel preview to make sure the turnout endpoints remain visible.
5. Run `cargo desktop-icons` from the repository. The Rust command exports
   `icon.icns` and `icon.ico` from `icon.png`, and the tray PNG and pixel file from
   `tray-source.png`. It uses Lanczos filtering when resizing both sources.
6. Inspect both PNGs on light, dark, and vivid backgrounds. Compare the app icon
   at 16, 32, 64, and 128 pixels and the tray icon at 22 × 22 pixels.

Run `cargo desktop-icons --check` to verify exports without changing files.
`cargo test` runs the same check on every platform, so CI fails when an export
is stale. After changing either PNG source, regenerate its exports before you
commit. Keep both sources as
1024 × 1024 RGBA images.
