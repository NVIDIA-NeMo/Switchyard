<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# switchyard-typesafe-client

An HTTP client for [TypeSafe](https://typesafe.ai)'s "System One Model" (Jev),
implementing `switchyard-libsy`'s [`TypeSafeProvider`] port.

`switchyard-libsy` stays I/O-free: it depends only on the `TypeSafeProvider`
trait, never on an HTTP client. This crate is the concrete implementation that
actually performs the network call — the "runner-owned classification
provider" described by
[Switchyard issue #723](https://github.com/NVIDIA-NeMo/Switchyard/issues/723).
`switchyard-runner` constructs one [`TypeSafeHttpClient`] per deployment (from
the optional `[type_safe_client]` table in the deployment TOML) and injects it
into every route configured with `type = "type_safe_classifier"`.

## Usage

```rust,no_run
use std::sync::Arc;
use switchyard_libsy::{TypeSafeClassifierInput, TypeSafeOption, TypeSafeProvider};
use switchyard_typesafe_client::TypeSafeHttpClient;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = TypeSafeHttpClient::from_env("TYPESAFE_API_KEY")?;

let verdict = client
    .classify(
        TypeSafeClassifierInput {
            question: "Which tier does this need?".to_string(),
            context: "[user] how do I list files in a directory?".to_string(),
        },
        &[
            TypeSafeOption::new("capable", "complex, multi-step work"),
            TypeSafeOption::new("efficient", "short, simple requests"),
        ],
    )
    .await?;

println!("{} ({:.2})", verdict.label, verdict.confidence);
println!("{:?}", verdict.probabilities);
# Ok(())
# }
```

`switchyard-runner` always reads the API key from the environment through
[`TypeSafeHttpClient::from_env`]. Direct library users may instead pass a key
to [`TypeSafeHttpClient::new`]. [`TypeSafeHttpClient`]'s `Debug` implementation
redacts the key in either case.

## Wire format

`POST {base_url}/v1/systemone`, `Authorization: Bearer <api_key>`:

```json
{
  "state": "<flattened conversation text>",
  "model": "jev-latest",
  "questions": {
    "route_0": {
      "type": "choice",
      "instructions": "<the classifier's question>",
      "criteria": { "capable": "...", "efficient": "..." }
    },
    "route_1": {
      "type": "choice",
      "instructions": "<the classifier's question>",
      "criteria": { "efficient": "...", "capable": "..." }
    }
  }
}
```

```json
{
  "usage": { "input_tokens": 42 },
  "answers": {
    "route_0": {
      "choice": "efficient",
      "confidence": 0.8,
      "probabilities": { "capable": 0.1, "efficient": 0.9 }
    },
    "route_1": {
      "choice": "efficient",
      "confidence": 0.6,
      "probabilities": { "capable": 0.2, "efficient": 0.8 }
    }
  }
}
```

The client sends up to three deterministic option orders in one request. It
averages and normalizes their probability distributions, selects the largest
average probability, and recomputes TypeSafe's Choice confidence. The verdict
also contains the full averaged distribution and wall-clock decision latency.
