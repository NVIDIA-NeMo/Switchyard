# LiteLLM OpenRouter Refactor Design

## Goal

Refactor the experimental Switchyard and LiteLLM package so its Dockerized
LiteLLM gateway sends Chat Completions inference through OpenRouter instead of
directly to OpenAI.

The example will route between these OpenRouter model IDs:

- Large: `openai/gpt-5.6-sol`
- Fast: `moonshotai/kimi-k3`

## Architecture

The integration retains its existing boundaries:

```text
application → Switchyard random router → LiteLLMSyClient
            → local LiteLLM gateway → OpenRouter Chat Completions
            → selected model
```

`LiteLLMSyClient` continues addressing the local gateway through LiteLLM's
OpenAI-compatible Chat Completions interface. Its existing `openai/strong` and
`openai/fast` LiteLLM client model values describe that local protocol hop; they
do not select OpenAI as the upstream inference provider.

The gateway configuration owns the upstream provider choice:

- `strong` maps to `openrouter/openai/gpt-5.6-sol`.
- `fast` maps to `openrouter/moonshotai/kimi-k3`.
- Both aliases read `OPENROUTER_API_KEY`.

The public `LiteLLMSyClient` constructor, normalized request and response
contracts, and `strong`/`fast` alias names remain unchanged. The requested
"Large" model is represented by the existing `strong` alias.

## Configuration

`compose.yaml` passes `OPENROUTER_API_KEY` into the LiteLLM container and fails
early when it is absent. `litellm-config.yaml` uses LiteLLM's native
`openrouter/...` provider prefix rather than representing OpenRouter as a
generic OpenAI-compatible endpoint.

`.env.example` contains `OPENROUTER_API_KEY`. The example remains local-only
and keeps its existing unauthenticated loopback gateway behavior.

The pinned LiteLLM release does not yet recognize Kimi K3's current
`reasoning_effort` support. `LiteLLMSyClient` forwards LiteLLM's
`allowed_openai_params` hint to the gateway so the supported parameter reaches
OpenRouter instead of being rejected by LiteLLM's stale model metadata.

No package dependency changes are required.

## Documentation

The package README will:

- describe OpenRouter as the inference provider;
- list the exact strong and fast model mappings;
- require `OPENROUTER_API_KEY`;
- explain that both models are called through OpenRouter Chat Completions;
- update the E2E, Harbor benchmark, troubleshooting, and request-flow wording;
- replace direct OpenAI model references with OpenRouter model pages; and
- retain the experimental notice and existing Switchyard random-routing link.

The runnable Python example needs no behavioral change because it uses stable
gateway aliases rather than provider model IDs.

## Testing

Offline coverage will verify:

- the gateway aliases use the exact `openrouter/...` model values;
- both aliases reference `OPENROUTER_API_KEY`;
- the client forwards the `reasoning_effort` compatibility hint;
- Compose passes `OPENROUTER_API_KEY`;
- the E2E fixture requires the explicit spend opt-in before checking Docker;
- the E2E fixture requires `OPENROUTER_API_KEY`; and
- normalized client responses use provider-neutral or OpenRouter model
  fixtures instead of the removed GPT-5.6 Luna model.

The authorized live E2E test will load `OPENROUTER_API_KEY` from the repository
root `.env`, start an isolated Docker Compose project, force Switchyard's random
router to select each alias once, and assert that both responses contain text.
The fixture will always tear down its containers, network, and volumes.

Validation will include focused package tests, package lint and type checks,
Compose configuration validation, root repository lint/type/test gates
appropriate to the diff, documentation link/content checks, and one live paid
call to each configured OpenRouter model.

## Scope

This refactor does not add streaming, tools, media, structured output, new
routing algorithms, new public APIs, or direct OpenRouter calls from
`LiteLLMSyClient`.
