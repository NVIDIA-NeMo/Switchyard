// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::Cursor;

use image::{ImageFormat, Rgba, RgbaImage};

use super::*;

fn still_image() -> String {
    let image = RgbaImage::from_pixel(80, 40, Rgba([255, 0, 0, 128]));
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, ImageFormat::Png).unwrap();
    data_uri("image/png", png.get_ref())
}

#[tokio::test]
async fn resizing_retains_alpha_aspect_ratio_and_provider_fields() {
    let processor = MediaProcessor::new().unwrap();
    for format in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let image = image_part(format, &still_image());
        let mut body = json!({"messages":[{"role":"user","content":[image]}],"tools":[{"input_schema":{"example":{"type":"image_url","image_url":{"url":"https://private.invalid"}}}}],"temperature":0.3,"custom_provider_control":true});
        body["messages"][0]["content"][0]["cache_control"] = json!({"type":"ephemeral"});
        let original = body.clone();
        processor
            .prepare(
                &mut body,
                format,
                &MediaConfig {
                    image_max_edge: Some(20),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let item = &body["messages"][0]["content"][0];
        let url = image_url(item).unwrap();
        let bytes = fetch::load(&processor.client, &url, 1024 * 1024)
            .await
            .unwrap();
        let resized = image::load_from_memory(&bytes).unwrap();
        assert_eq!((resized.width(), resized.height()), (20, 10));
        assert_eq!(resized.to_rgba8().get_pixel(0, 0)[3], 128);
        assert_eq!(body["tools"], original["tools"]);
        assert_eq!(item["cache_control"], json!({"type":"ephemeral"}));
        assert_eq!(body["custom_provider_control"], true);
        assert_eq!(body["temperature"], 0.3);
    }
}

#[tokio::test]
async fn image_limit_is_shared_across_messages_and_nested_tool_results() {
    let processor = MediaProcessor::new().unwrap();
    let mut body = json!({"messages":[
        {"role":"user","content":[{"type":"image_url","image_url":{"url":"https://invalid/old"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image_url","image_url":{"url":"https://invalid/new"}}]}]}
    ]});
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                max_images: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "[image omitted]");
    assert_eq!(
        body["messages"][1]["content"][0]["content"][0]["image_url"]["url"],
        "https://invalid/new"
    );
}

#[tokio::test]
async fn text_only_judge_never_fetches_images_or_videos() {
    let processor = MediaProcessor::new().unwrap();
    let mut body = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"question"},
        {"type":"image_url","image_url":{"url":"https://127.0.0.1/private"}},
        {"type":"video_url","video_url":{"url":"https://127.0.0.1/private"}}
    ]}]});
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                max_images: Some(0),
                video: VideoMode::Omit,
                image_max_edge: Some(384),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "question");
    assert_eq!(body["messages"][0]["content"][1]["text"], "[image omitted]");
    assert_eq!(body["messages"][0]["content"][2]["text"], "[video omitted]");
}

#[tokio::test]
async fn native_video_formats_need_no_download_or_decoder() {
    let processor = MediaProcessor::new().unwrap();
    for url in ["https://example.com/clip.mp4", "data:video/mp4;base64,AQID"] {
        let mut body = json!({"messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":url}}]}]});
        processor
            .prepare(
                &mut body,
                WireFormat::OpenAiChat,
                &MediaConfig {
                    video: VideoMode::File,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let file = &body["messages"][0]["content"][0]["file"];
        assert_eq!(file["format"], "video/mp4");
        assert_eq!(
            file[if url.starts_with("data:") {
                "file_data"
            } else {
                "file_id"
            }],
            url
        );
        processor
            .prepare(
                &mut body,
                WireFormat::OpenAiChat,
                &MediaConfig {
                    video: VideoMode::VideoUrl,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(body["messages"][0]["content"][0]["video_url"]["url"], url);
    }
}

#[tokio::test]
async fn disabled_processing_is_exact_passthrough_and_limits_fail_closed() {
    let processor = MediaProcessor::new().unwrap();
    let original = json!({"input":[{"role":"user","content":[image_part(WireFormat::OpenAiResponses, &still_image())]}]});
    let mut body = original.clone();
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiResponses,
            &MediaConfig::default(),
        )
        .await
        .unwrap();
    assert_eq!(body, original);
    let result = processor
        .prepare(
            &mut body,
            WireFormat::OpenAiResponses,
            &MediaConfig {
                image_max_edge: Some(20),
                max_output_bytes: 1,
                ..Default::default()
            },
        )
        .await;
    assert!(result.unwrap_err().to_string().contains("max_output_bytes"));
    assert!(
        MediaConfig {
            video: VideoMode::File,
            ..Default::default()
        }
        .validate(WireFormat::OpenAiResponses)
        .is_err()
    );
    assert!(
        MediaConfig {
            video_max_frames: 0,
            ..Default::default()
        }
        .validate(WireFormat::OpenAiChat)
        .is_err()
    );
}

#[tokio::test]
#[ignore = "requires ffmpeg and ffprobe on PATH"]
async fn video_frames_work_in_all_formats_and_preserve_original_request() {
    let processor = MediaProcessor::new().unwrap();
    let bytes = include_bytes!("../tests/fixtures/colors.mp4");
    let source = data_uri("video/mp4", bytes);
    for format in [
        WireFormat::OpenAiChat,
        WireFormat::OpenAiResponses,
        WireFormat::AnthropicMessages,
    ] {
        let original = json!({"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"video_url","video_url":{"url":source}}]}]});
        let mut body = original.clone();
        processor
            .prepare(
                &mut body,
                format,
                &MediaConfig {
                    video: VideoMode::Frames,
                    video_max_frames: 3,
                    frame_max_edge: 32,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let content = body["messages"][0]["content"].as_array().unwrap();
        let frames: Vec<_> = content.iter().filter_map(image_url).collect();
        assert_eq!(frames.len(), 3);
        for url in frames {
            let image = image::load_from_memory(
                &fetch::load(&processor.client, &url, 1_000_000)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(image.width() <= 32 && image.height() <= 32);
        }
        assert_eq!(
            original["messages"][0]["content"][1]["video_url"]["url"],
            source
        );
        assert!(content.iter().any(|item| {
            item["text"]
                .as_str()
                .is_some_and(|text| text.contains("0.000s"))
        }));
    }
}

#[tokio::test]
#[ignore = "requires ffmpeg and ffprobe on PATH"]
async fn image_budget_skips_fetches_and_retains_latest_sample_without_resampling() {
    let processor = MediaProcessor::new().unwrap();
    let mut body = json!({"messages":[{"role":"user","content":[
        {"type":"image_url","image_url":{"url":"https://127.0.0.1/never-fetch"}},
        {"type":"video_url","video_url":{"url":data_uri("video/mp4", include_bytes!("../tests/fixtures/colors.mp4"))}}
    ]}]});
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                image_max_edge: Some(32),
                max_images: Some(1),
                video: VideoMode::Frames,
                video_max_frames: 3,
                frame_max_edge: 32,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content.iter().filter_map(image_url).count(), 1);
    assert_eq!(content[0]["text"], "[image omitted]");
    assert!(content.iter().any(|item| {
        item["text"]
            .as_str()
            .is_some_and(|text| text.contains("1.749s"))
    }));
}

#[tokio::test]
async fn image_resize_applies_exif_and_avoids_reencoding_small_images() {
    let slots = Arc::new(Semaphore::new(1));
    let (_, bytes) = decode::resize(
        include_bytes!("../tests/fixtures/oriented.jpg").to_vec(),
        40,
        slots.clone(),
    )
    .await
    .unwrap();
    let rotated = image::load_from_memory(&bytes).unwrap();
    assert_eq!((rotated.width(), rotated.height()), (20, 40));
    let processor = MediaProcessor::new().unwrap();
    let original = fetch::load(&processor.client, &still_image(), 100_000)
        .await
        .unwrap();
    let (_, retained) = decode::resize(original.clone(), 200, slots).await.unwrap();
    assert_eq!(retained, original);
}

#[tokio::test]
async fn preparation_deadline_includes_worker_queueing() {
    let processor = MediaProcessor::new().unwrap();
    let _permits = processor.slots.clone().acquire_many_owned(2).await.unwrap();
    let mut body = json!({"input":[{"role":"user","content":[image_part(WireFormat::OpenAiResponses, &still_image())]}]});
    let result = processor
        .prepare(
            &mut body,
            WireFormat::OpenAiResponses,
            &MediaConfig {
                image_max_edge: Some(20),
                timeout_ms: 5,
                ..Default::default()
            },
        )
        .await;
    assert!(matches!(result, Err(MediaError::Timeout)));
}

#[tokio::test]
async fn native_video_conversion_keeps_mime_and_same_format_options() {
    let processor = MediaProcessor::new().unwrap();
    let mut body = json!({"messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"https://example.com/a.webm","fps":2}}
    ]}]});
    let original = body.clone();
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                video: VideoMode::VideoUrl,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(body, original);
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                video: VideoMode::File,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["file"]["format"],
        "video/webm"
    );
}

#[tokio::test]
async fn zero_image_budget_skips_video_sources_before_validation() {
    let processor = MediaProcessor::new().unwrap();
    let mut body = json!({"messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"unsupported_id":"unused"}}
    ]}]});
    processor
        .prepare(
            &mut body,
            WireFormat::OpenAiChat,
            &MediaConfig {
                max_images: Some(0),
                video: VideoMode::Frames,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["text"],
        "[video frames omitted]"
    );
}

#[test]
fn declared_video_mime_takes_precedence_over_the_url_suffix() {
    let source = video_source(&json!({
        "type":"input_video", "video_url":"https://example.com/opaque.mov",
        "media_type":"video/webm"
    }))
    .unwrap();
    assert_eq!(source.1, "video/webm");
}
