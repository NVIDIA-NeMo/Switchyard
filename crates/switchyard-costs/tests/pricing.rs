// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use switchyard_costs::{CostSource, Pricing};
use switchyard_protocol::{InputCacheUsage, Usage};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const GROUPED_TOML: &str = r#"
[[prices]]
models = ["direct", "provider/model"]
input = 1
output = 2
cache_read = 0.1
cache_write = 1.25
cache_write_1h = 2
"#;

#[test]
fn grouped_and_keyed_files_price_the_same_usage() -> TestResult {
    let dir = tempfile::tempdir()?;
    let usage = Usage {
        input_tokens: Some(1_000_000),
        output_tokens: Some(2_000_000),
        cache: Some(Box::new(InputCacheUsage {
            cached_input_tokens: Some(3_000_000),
            cache_creation_input_tokens: Some(5_000_000),
            cache_creation_1h_input_tokens: Some(2_000_000),
        })),
        ..Usage::default()
    };
    for (name, text, models) in [
        (
            "grouped.ToMl",
            GROUPED_TOML,
            vec!["direct", "provider/model"],
        ),
        (
            "grouped.json",
            r#"{"prices":[{"models":["direct","provider/model"],"input":1,"output":2,"cache_read":0.1,"cache_write":1.25,"cache_write_1h":2}]}"#,
            vec!["direct", "provider/model"],
        ),
        (
            "keyed.toml",
            "[\"provider/model\"]\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1.25\ncache_write_1h=2",
            vec!["provider/model"],
        ),
        (
            "keyed.json",
            r#"{"provider/model":{"input":1,"output":2,"cache_read":0.1,"cache_write":1.25,"cache_write_1h":2}}"#,
            vec!["provider/model"],
        ),
        (
            "keyed-reserved.toml",
            "[prices]\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1.25\ncache_write_1h=2",
            vec!["prices"],
        ),
        (
            "keyed-reserved.json",
            r#"{"prices":{"input":1,"output":2,"cache_read":0.1,"cache_write":1.25,"cache_write_1h":2}}"#,
            vec!["prices"],
        ),
        ("empty-grouped.toml", "prices=[]", vec![]),
        ("empty-grouped.json", r#"{"prices":[]}"#, vec![]),
        ("empty-keyed.toml", "", vec![]),
        ("empty-keyed.json", "{}", vec![]),
    ] {
        let path = dir.path().join(name);
        std::fs::write(&path, text)?;
        let prices = Pricing::load(&path)?;
        for model in models {
            let cost = prices.estimate(model, &usage).ok_or("missing estimate")?;
            assert!((cost.usd - 13.05).abs() < 1e-9, "{name}: {model}");
            assert_eq!(cost.source, CostSource::Estimated);
        }
        assert!(!prices.contains("provider/typo"));
        assert_eq!(prices.estimate("provider/typo", &usage), None);
    }
    Ok(())
}

// These fixtures keep repeated fields in JSON text so decoding cannot erase them first.
#[test]
fn rejects_invalid_groups_and_preserved_raw_json_fields() {
    for text in [
        r#"{"prices":[{"models":[],"input":1,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[{"models":["a","a"],"input":1,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[{"models":["a"],"input":1,"output":2,"cache_read":0.1,"cache_write":1},{"models":["a"],"input":3,"output":4,"cache_read":0.2,"cache_write":3}]}"#,
        r#"{"prices":[{"models":[""],"input":1,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[{"models":[" a"],"input":1,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[{"models":["a"],"input":1,"output":2,"cache_read":0.1}]}"#,
        r#"{"prices":[{"models":["a"],"input":1,"output":2,"cache_read":0.1,"cache_write":1,"cache_wirte":1}]}"#,
        r#"{"prices":[{"models":["a"],"models":["b"],"input":1,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[{"models":["a"],"input":1,"input":3,"output":2,"cache_read":0.1,"cache_write":1}]}"#,
        r#"{"prices":[],"prices":[]}"#,
        r#"{"prices":[],"unexpected":true}"#,
        r#"{"prices":[{"models":["a"],"input":1,"output":2,"cache_read":0.1,"cache_write":1}],"b":{"input":1,"output":2,"cache_read":0.1,"cache_write":1}}"#,
        r#"{"a":{"input":1,"output":2,"cache_read":0.1,"cache_write":1},"a":{"input":3,"output":4,"cache_read":0.2,"cache_write":3}}"#,
        r#"{"a":{"input":1,"input":3,"output":2,"cache_read":0.1,"cache_write":1}}"#,
        r#"{"a":{"input":1,"output":2,"cache_read":0.1,"cache_write":1,"cache_wirte":1}}"#,
    ] {
        assert!(Pricing::parse(text.as_bytes()).is_err(), "accepted {text}");
    }
}

#[test]
fn rejects_invalid_toml_before_returning_a_price_table() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("prices.toml");
    for text in [
        "[[prices]]\nmodels=[]\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "[[prices]]\nmodels=['a','a']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "[[prices]]\nmodels=['a']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1\n[[prices]]\nmodels=['a']\ninput=3\noutput=4\ncache_read=0.2\ncache_write=3",
        "[[prices]]\nmodels=['a ']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "[[prices]]\nmodels=['a']\nmodels=['b']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "[[prices]]\nmodels=['a']\ninput=1\ninput=3\noutput=2\ncache_read=0.1\ncache_write=1",
        "unexpected=true\n[[prices]]\nmodels=['a']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "['b']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1\n[[prices]]\nmodels=['a']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1",
        "[[prices]]\nmodels=['a']\ninput=1\noutput=2\ncache_read=0.1\ncache_write=1\ncache_wirte=1",
    ] {
        std::fs::write(&path, text)?;
        let error = Pricing::load(&path).err().ok_or("invalid pricing loaded")?;
        assert!(error.starts_with("invalid pricing file "), "{error}");
        assert!(error.contains(&path.display().to_string()), "{error}");
    }
    Ok(())
}

#[test]
fn every_grouped_rate_must_be_finite_and_in_range() -> TestResult {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("prices.toml");
    for field in [
        "input",
        "output",
        "cache_read",
        "cache_write",
        "cache_write_1h",
    ] {
        for value in ["-1", "10001", "nan", "inf"] {
            let mut text = String::from("[[prices]]\nmodels=['a']\n");
            for name in [
                "input",
                "output",
                "cache_read",
                "cache_write",
                "cache_write_1h",
            ] {
                text.push_str(&format!(
                    "{name}={}\n",
                    if name == field { value } else { "1" }
                ));
            }
            std::fs::write(&path, &text)?;
            assert!(Pricing::load(&path).is_err(), "accepted {field}={value}");
        }
    }
    Ok(())
}

#[test]
fn bundled_aliases_keep_all_cache_rates() -> TestResult {
    let prices = Pricing::bundled()?;
    let usage = Usage {
        input_tokens: Some(1_000_000),
        output_tokens: Some(1_000_000),
        cache: Some(Box::new(InputCacheUsage {
            cached_input_tokens: Some(1_000_000),
            cache_creation_input_tokens: Some(2_000_000),
            cache_creation_1h_input_tokens: Some(1_000_000),
        })),
        ..Usage::default()
    };
    for model in [
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "anthropic/claude-haiku-4.5",
        "azure/anthropic/claude-haiku-4-5",
    ] {
        let cost = prices.estimate(model, &usage).ok_or("missing alias")?;
        assert!((cost.usd - 9.35).abs() < 1e-9, "{model}");
    }
    Ok(())
}

#[test]
fn bundled_provider_prices_and_optional_rates_stay_distinct() -> TestResult {
    let prices = Pricing::bundled()?;
    let mut usage = Usage {
        input_tokens: Some(1_000_000),
        output_tokens: Some(0),
        ..Usage::default()
    };
    for (model, expected) in [
        ("gemini-3.8-flash", 0.75),
        ("google/gemini-3.8-flash", 0.825),
    ] {
        assert_eq!(
            prices.estimate(model, &usage).ok_or("missing model")?.usd,
            expected
        );
    }
    // A missing one-hour rate must keep the estimate unknown, even if other rates match.
    usage.cache = Some(Box::new(InputCacheUsage {
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(1_000_000),
        cache_creation_1h_input_tokens: Some(1_000_000),
    }));
    assert_eq!(
        prices
            .estimate("gemini-3.7-flash", &usage)
            .ok_or("missing hourly rate")?
            .usd,
        1.5
    );
    assert_eq!(prices.estimate("google/gemini-3.7-flash", &usage), None);
    usage.cache = None;
    for (model, expected) in [
        ("nvidia/nemotron-3-ultra-550b-a55b", 0.6),
        ("nvidia/nemotron-3-ultra-550b-a55b:free", 0.0),
    ] {
        assert_eq!(
            prices
                .estimate(model, &usage)
                .ok_or("missing free or paid rate")?
                .usd,
            expected
        );
    }
    assert!(!prices.contains("google/gemini-3.8-flash:free"));
    Ok(())
}
