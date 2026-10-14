// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Initialize and reuse one Dynamo download pool per target.

use std::sync::Arc;

use dynamo_multimodal::fetch::{FetchPolicy, MediaFetcher};
use tokio::sync::Semaphore;

use super::{MediaConfig, MediaError, MediaProcessor, Result};

impl MediaProcessor {
    pub(crate) fn new(config: MediaConfig, slots: Arc<Semaphore>) -> Result<Self> {
        let mut policy = FetchPolicy::default();
        policy.allow_direct_ip = true;
        policy.allow_direct_port = true;
        policy.max_bytes = config.max_input_bytes as u64;
        // Dynamo caps the encoded payload; our setting counts decoded bytes.
        policy.max_data_url_bytes = config.max_input_bytes.div_ceil(3) * 4;
        // Preparation bounds the whole call, including queueing and decoding.
        policy.timeout = None;
        let fetcher = MediaFetcher::new(policy)
            .map_err(|_| MediaError::Invalid("cannot create media fetcher"))?;
        Ok(Self {
            config,
            fetcher,
            slots,
        })
    }

    pub(super) async fn load(&self, source: String) -> Result<Vec<u8>> {
        // Dynamo errors may contain URLs or hostnames. Keep request data out of errors.
        let bytes =
            self.fetcher.fetch(source).await.map_err(|_| {
                MediaError::Invalid("media fetch failed or exceeds max_input_bytes")
            })?;
        if bytes.len() > self.config.max_input_bytes {
            return Err(MediaError::Invalid("media exceeds max_input_bytes"));
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enforces_target_limits_and_redacts_errors() {
        let slots = Arc::new(Semaphore::new(2));
        let target = |max_input_bytes| {
            MediaProcessor::new(
                MediaConfig {
                    max_input_bytes,
                    ..Default::default()
                },
                slots.clone(),
            )
            .unwrap()
        };
        let fetcher = target(3);
        let source = "data:image/png;base64,AQID";
        assert_eq!(fetcher.load(source.into()).await.unwrap(), [1, 2, 3]);
        for limit in [1, 2] {
            assert!(target(limit).load(source.into()).await.is_err());
        }
        assert_eq!(fetcher.load(source.into()).await.unwrap(), [1, 2, 3]);
        let fetcher = target(100);
        for source in [
            "file:///tmp/secret",
            "https://127.0.0.1/secret",
            "https://[::1]/secret",
            "https://169.254.169.254/secret",
            "https://user:secret@example.com/a",
            "https://metadata.google.internal./secret",
            "data:image/png,secret",
        ] {
            let error = fetcher.load(source.into()).await.unwrap_err();
            assert_eq!(
                error.to_string(),
                "media fetch failed or exceeds max_input_bytes"
            );
        }
    }
}
