// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reuse Dynamo download pools with each target's input size limit.

use std::collections::{HashMap, hash_map::Entry};

use dynamo_multimodal::fetch::{FetchPolicy, MediaFetcher};
use parking_lot::Mutex;

use crate::{MediaError, Result};

#[derive(Default)]
pub(crate) struct Fetchers(Mutex<HashMap<usize, MediaFetcher>>);

impl Fetchers {
    pub(crate) async fn load(&self, source: String, max_bytes: usize) -> Result<Vec<u8>> {
        // Policies are immutable; targets with the same limit share a connection pool.
        let fetcher = {
            let mut fetchers = self.0.lock();
            if let Entry::Vacant(entry) = fetchers.entry(max_bytes) {
                let mut policy = FetchPolicy::default();
                policy.allow_direct_ip = true;
                policy.allow_direct_port = true;
                policy.max_bytes = max_bytes as u64;
                // Dynamo caps the encoded payload; our setting counts decoded bytes.
                policy.max_data_url_bytes = max_bytes.div_ceil(3) * 4;
                // MediaProcessor bounds the whole call, including queueing and decoding.
                policy.timeout = None;
                let fetcher = MediaFetcher::new(policy)
                    .map_err(|_| MediaError::Invalid("cannot create media fetcher"))?;
                entry.insert(fetcher);
            }
            fetchers[&max_bytes].clone()
        };
        // Dynamo errors may contain URLs or hostnames. Keep request data out of errors.
        let bytes = fetcher
            .fetch(source)
            .await
            .map_err(|_| MediaError::Invalid("media fetch failed or exceeds max_input_bytes"))?;
        if bytes.len() > max_bytes {
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
        let fetchers = Fetchers::default();
        let source = "data:image/png;base64,AQID";
        assert_eq!(fetchers.load(source.into(), 3).await.unwrap(), [1, 2, 3]);
        for limit in [1, 2] {
            assert!(fetchers.load(source.into(), limit).await.is_err());
        }
        assert_eq!(fetchers.load(source.into(), 3).await.unwrap(), [1, 2, 3]);
        for source in [
            "file:///tmp/secret",
            "https://127.0.0.1/secret",
            "https://[::1]/secret",
            "https://169.254.169.254/secret",
            "https://user:secret@example.com/a",
            "https://metadata.google.internal./secret",
            "data:image/png,secret",
        ] {
            let error = fetchers.load(source.into(), 100).await.unwrap_err();
            assert_eq!(
                error.to_string(),
                "media fetch failed or exceeds max_input_bytes"
            );
        }
    }
}
