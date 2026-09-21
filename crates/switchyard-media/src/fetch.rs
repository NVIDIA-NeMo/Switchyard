// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded media loading without forwarding inference credentials.

use std::net::IpAddr;
use std::sync::{Arc, LazyLock};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures_util::StreamExt;
use ipnet::IpNet;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::{Client, Url};

use crate::{MediaError, Result};

// Adapted from Dynamo's media/loader.rs; see README.md for source and revision.
static BLOCKED_NETWORKS: LazyLock<Vec<IpNet>> = LazyLock::new(|| {
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    ]
    .iter()
    .map(|cidr| cidr.parse().expect("constant CIDR"))
    .collect()
});

fn blocked(ip: IpAddr) -> bool {
    BLOCKED_NETWORKS.iter().any(|network| network.contains(&ip))
}

fn validate_url(url: &Url) -> Result<()> {
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return Err(MediaError::Invalid(
            "media downloads require HTTPS without credentials",
        ));
    }
    let host = url
        .host_str()
        .ok_or(MediaError::Invalid("missing media host"))?;
    if matches!(
        host.trim_end_matches('.'),
        "localhost"
            | "localhost.localdomain"
            | "metadata"
            | "metadata.google.internal"
            | "metadata.goog"
            | "kubernetes.default"
            | "kubernetes.default.svc"
    ) || host
        .trim_matches(['[', ']'])
        .parse::<IpAddr>()
        .is_ok_and(blocked)
    {
        return Err(MediaError::Invalid("media URL must use a public address"));
    }
    Ok(())
}

struct PublicResolver;

impl Resolve for PublicResolver {
    fn resolve(&self, name: Name) -> Resolving {
        Box::pin(async move {
            let addresses: Vec<_> = tokio::net::lookup_host((name.as_str(), 0)).await?.collect();
            if addresses.is_empty() || addresses.iter().any(|address| blocked(address.ip())) {
                return Err(
                    std::io::Error::other("media DNS must resolve to public addresses").into(),
                );
            }
            Ok(Box::new(addresses.into_iter()) as Addrs)
        })
    }
}

pub(crate) fn client() -> Result<Client> {
    Client::builder()
        // An environment proxy could bypass destination validation.
        .no_proxy()
        .dns_resolver(Arc::new(PublicResolver))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= 5 || validate_url(attempt.url()).is_err() {
                attempt.error("media redirect rejected")
            } else {
                attempt.follow()
            }
        }))
        .build()
        .map_err(|_| MediaError::Invalid("cannot create media HTTP client"))
}

pub(crate) async fn load(client: &Client, source: &str, max_bytes: usize) -> Result<Vec<u8>> {
    if let Some(data) = source.strip_prefix("data:") {
        let (header, encoded) = data
            .split_once(',')
            .ok_or(MediaError::Invalid("invalid media data URI"))?;
        if !header.ends_with(";base64") {
            return Err(MediaError::Invalid("media data URI must be base64 encoded"));
        }
        if encoded.len() > max_bytes.saturating_add(2) / 3 * 4 {
            return Err(MediaError::Invalid("media exceeds max_input_bytes"));
        }
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| MediaError::Invalid("invalid media base64"))?;
        if bytes.len() > max_bytes {
            return Err(MediaError::Invalid("media exceeds max_input_bytes"));
        }
        return Ok(bytes);
    }
    let url = Url::parse(source).map_err(|_| MediaError::Invalid("invalid media URL"))?;
    validate_url(&url)?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| MediaError::Invalid("media download failed"))?
        .error_for_status()
        .map_err(|_| MediaError::Invalid("media download returned an error status"))?;
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(MediaError::Invalid("media exceeds max_input_bytes"));
    }
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| MediaError::Invalid("media download interrupted"))?;
        if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
            return Err(MediaError::Invalid("media exceeds max_input_bytes"));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_local_and_credential_urls() {
        for url in [
            "http://example.com/a",
            "file:///tmp/a",
            "https://127.0.0.1/a",
            "https://[::1]/a",
            "https://[::ffff:127.0.0.1]/a",
            "https://169.254.169.254/a",
            "https://user:secret@example.com/a",
            "https://metadata.google.internal./a",
        ] {
            assert!(validate_url(&Url::parse(url).unwrap()).is_err(), "{url}");
        }
        assert!(validate_url(&Url::parse("https://huggingface.co/example").unwrap()).is_ok());
    }

    #[tokio::test]
    async fn bounds_inline_media_without_leaking_source() {
        let client = client().unwrap();
        assert_eq!(
            load(&client, "data:image/png;base64,AQID", 3)
                .await
                .unwrap(),
            [1, 2, 3]
        );
        assert!(
            load(&client, "data:image/png;base64,AQID", 2)
                .await
                .is_err()
        );
        assert!(load(&client, "data:image/png,secret", 100).await.is_err());
    }
}
