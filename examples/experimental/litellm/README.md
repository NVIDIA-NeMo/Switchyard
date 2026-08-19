# Switchyard routing plugin for LiteLLM

> **Experimental integration:** This example and its Python API may change without notice.

> **Checkout-only dependency:** The current published `nemo-switchyard` release does not contain
> the decision-only Python API used here. Run this example from this repository checkout, where
> `[tool.uv.sources]` binds the example to the adjacent Switchyard source. Do not install or publish
> `switchyard-litellm` as a standalone package until a compatible Switchyard release is available
> and its dependency floor can be updated.

This package integrates Switchyard at LiteLLM's native routing-plugin boundary. Applications keep
using LiteLLM's `Router` or OpenAI-compatible proxy. The same configured object has two LiteLLM
roles: before deployment selection it narrows the candidate set to the model selected by
Switchyard; after selection it applies any supported Switchyard request rewrite before LiteLLM
translates and sends the provider request.

The integration pins LiteLLM 1.97.0. Model inventory and routing policy are both owned by the
deployer:

- LiteLLM YAML defines the public model group, candidate models, credentials, and provider options.
- Switchyard TOML selects the algorithm and all of its supported routing parameters.

The checked-in profiles use two OpenRouter models as demonstration values. Those model IDs are not
built into `switchyard_litellm` and can be replaced without changing package source.

## Layout

```text
litellm/
├── deployment/
│   ├── .env.example
│   ├── Dockerfile
│   ├── compose.yaml
│   └── profiles/
│       ├── stage/{litellm.yaml,switchyard.toml}
│       └── random/{litellm.yaml,switchyard.toml}
├── examples/python_router.py
├── src/switchyard_litellm/
│   ├── configuration/
│   └── plugins/
└── tests/{unit,integration}/
```

## Request flow

```text
application → LiteLLM Router → Switchyard routing plugin
                              → Algorithm.run_stream()
                              → one selected candidate + optional request delta
            → LiteLLM deployment selection
            → same object as deployment callback → provider inference
```

The candidate-bound Stage and Random plugins construct Switchyard algorithms from the live
candidate list supplied by LiteLLM. The low-level `SwitchyardRoutingPlugin` then:

1. converts LiteLLM's `structured_messages` to a normalized Switchyard request;
2. runs the supplied algorithm until `Step.Done`;
3. validates and converts any supported request delta;
4. replaces `context.candidate_models` with the selected exact deployment; and
5. records the decision and private delta under `context.signals["switchyard"]`.

LiteLLM copies those signals into request metadata. After LiteLLM selects a deployment, the same
object's `async_pre_call_deployment_hook` applies the delta and removes it from the metadata passed
downstream. The proxy profile therefore registers the dotted object as both a routing plugin and a
callback.

LiteLLM remains responsible for credentials, retries, provider translation, inference, and the
OpenAI-compatible API.

## Supported algorithm semantics

Compatibility is determined by behavior rather than an algorithm-name allowlist.
`StageRoutingPlugin` and `RandomRoutingPlugin` provide the deployer-oriented bindings supported by
the checked-in configuration loader.

| Behavior | Result |
|---|---|
| Random selects a current candidate | Supported |
| Stage uses only request-history signals | Supported |
| Algorithm requests `Step.CallModel` | Rejected |
| Algorithm returns an existing response | Rejected |
| Algorithm rewrites text instructions/messages | Applied after deployment selection |
| Algorithm rewrites tools/tool choice | Applied after deployment selection |
| Algorithm rewrites sampling, output, reasoning effort, or stream | Applied after deployment selection |
| Algorithm rewrites supported provider-extension fields | Applied after deployment selection |
| Algorithm rewrites raw reasoning, preservation, or unsupported content/extensions | Rejected |
| Algorithm selects outside the current candidate pool | Rejected |
| Algorithm stream ends without `Step.Done` | Rejected |

Escalation and classifier-backed routers are not compatible because a LiteLLM routing plugin cannot
service intermediate model calls. Stage's system prompts and handoff notes do not require an
intermediate call, so the dual-role Stage plugin supports them. Unsupported behavior fails closed
before LiteLLM sends inference.

LiteLLM 1.97.0's routing context exposes structured messages but does not expose the caller's tools,
sampling controls, output controls, or provider-specific arguments. The adapter therefore uses
delta semantics: a field explicitly changed by Switchyard overrides the corresponding LiteLLM
argument, while a field Switchyard leaves at the adapter default does not clear or replace the
caller's original value. A Switchyard algorithm can produce an outbound override for tools,
`tool_choice`, `temperature`, `top_p`, `top_k`, `max_completion_tokens`, `response_format`,
`reasoning_effort`, `stream`, and the safe OpenAI Chat extension fields supported by the adapter.
It cannot inspect the original value of those fields through the current routing-plugin API, or
intentionally clear a caller value back to an adapter default such as `None` or an empty tool list.

The message adapter supports `system`, `developer`, and `user` text; assistant text and OpenAI
function tool calls; and text tool results. It rejects malformed function arguments, media or audio
content, legacy `function` messages, and unknown roles or content blocks.

## Quick start with the local proxy

Prerequisites are Docker Compose and an OpenRouter key with access to the example models. From this
directory:

```bash
cp deployment/.env.example deployment/.env
# Set OPENROUTER_API_KEY in deployment/.env.
docker compose -f deployment/compose.yaml up -d --build --wait
curl -fsS http://127.0.0.1:4000/health/liveliness
```

The default profile is `stage`. Send a request to its public `switchyard` model group:

```bash
curl -i http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "switchyard",
    "messages": [{"role": "user", "content": "Reply with the word hello."}],
    "max_tokens": 64
  }'
```

LiteLLM keeps the response body's `model` field equal to the public group. The
`x-litellm-model-name` response header identifies the concrete model selected by Switchyard.

Start the Random profile instead with:

```bash
SWITCHYARD_LITELLM_PROFILE=random \
  docker compose -f deployment/compose.yaml up -d --build --wait
```

Stop the proxy without deleting repository data:

```bash
docker compose -f deployment/compose.yaml down
```

## Configure a deployment profile

Each directory under `deployment/profiles/` is a complete selectable profile:

```text
profiles/my-profile/
├── litellm.yaml       # model inventory and LiteLLM settings
└── switchyard.toml    # Switchyard algorithm and parameters
```

The LiteLLM YAML registers the environment-configured plugin object:

```yaml
model_list:
  - model_name: switchyard
    litellm_params:
      model: provider/capable-model
      api_key: os.environ/PROVIDER_API_KEY
  - model_name: switchyard
    litellm_params:
      model: provider/efficient-model
      api_key: os.environ/PROVIDER_API_KEY

router_settings:
  plugins:
    - switchyard_litellm.configuration.configured_plugin.ROUTING_PLUGIN

litellm_settings:
  callbacks:
    - switchyard_litellm.configuration.configured_plugin.ROUTING_PLUGIN
```

Compose mounts the selected profile at `/app/deployment` and sets
`SWITCHYARD_LITELLM_CONFIG=/app/deployment/switchyard.toml`. LiteLLM imports the pre-created object
from the dotted path above when the proxy starts.

### Stage fields

```toml
algorithm = "stage"
picker = "efficient_first"
confidence_threshold = 0.5
recent_window = 3
only_on_wrong_signal_escalation = true
escalation_note = "The efficient tier failed; continue from its work."
deescalation_note = "The capable tier completed the recovery."
capable_system_prompt = "Handle this request as the capable tier."
efficient_system_prompt = "Handle this request as the efficient tier."
```

| Field | Required | Accepted value |
|---|---:|---|
| `algorithm` | yes | `"stage"` |
| `picker` | yes | `"capable_first"` or `"efficient_first"` |
| `confidence_threshold` | yes | finite number from `0` through `1` |
| `recent_window` | no | nonnegative integer; omitted means no fixed recent window |
| `only_on_wrong_signal_escalation` | no | Boolean; defaults to `true` |
| `escalation_note` | no | nonempty text appended when Stage hands work to the capable tier |
| `deescalation_note` | no | nonempty text appended when Stage hands work back; requires `escalation_note` |
| `capable_system_prompt` | no | nonempty system instruction used for the capable tier |
| `efficient_system_prompt` | no | nonempty system instruction used for the efficient tier |

Stage requires exactly two unique candidate model IDs. Their first occurrence in LiteLLM's
candidate order defines the capable model first and the efficient model second. `picker` controls
which tier is chosen in the absence of an escalation signal; it does not change those roles.

### Random fields

```toml
algorithm = "random"
seed = 6
weights = [0.25, 0.75]
```

| Field | Required | Accepted value |
|---|---:|---|
| `algorithm` | yes | `"random"` |
| `seed` | no | integer from `0` through `2^64 - 1` |
| `weights` | no | nonempty array of finite, nonnegative numbers with at least one positive value |

Random selects across every unique candidate ID in first-seen LiteLLM order. `weights` maps to that
same order, so its length must equal the number of unique candidates in the request. Omitting
`weights` gives each candidate equal weight; omitting `seed` uses Switchyard's unseeded behavior.

Static errors—missing files, malformed TOML, unknown keys, wrong types, invalid ranges, and invalid
weight values—fail while LiteLLM imports the plugin at proxy startup. Constraints that depend on
LiteLLM's live candidates fail at request time: Stage's two-candidate requirement and Random's
candidate-to-weight count.

To add a profile, copy an existing directory, edit both files, and select its directory name:

```bash
SWITCHYARD_LITELLM_PROFILE=my-profile \
  docker compose -f deployment/compose.yaml up -d --build --wait
```

On a successful decision without a rewrite, the plugin preserves signals from earlier plugins and
adds:

```python
context.signals["switchyard"] = {
    "selected_model_id": "provider/efficient-model",
    "fallback_models": ["provider/capable-model"],
}
```

`fallback_models` is diagnostic metadata only. The plugin narrows the candidate set; it does not add
a LiteLLM fallback policy. When Switchyard rewrites a supported request field, the signal also
temporarily contains a private `request_patch`. The callback consumes that patch after deployment
selection and excludes it from the metadata it returns downstream.

## Use the plugin with LiteLLM's Python Router

For programmatic use, construct the plugin directly; TOML and the environment-backed import module
are proxy deployment conveniences.

```python
import litellm
from litellm import Router
from switchyard_litellm import StageRoutingPlugin

model_list = [
    {
        "model_name": "switchyard",
        "litellm_params": {"model": "provider/capable-model"},
    },
    {
        "model_name": "switchyard",
        "litellm_params": {"model": "provider/efficient-model"},
    },
]
plugin = StageRoutingPlugin(
    picker="efficient_first",
    confidence_threshold=0.5,
    recent_window=3,
)
router = Router(model_list=model_list, plugins=[plugin])
litellm.callbacks.append(plugin)
```

The explicit callback registration is required for programmatic `Router` use because LiteLLM's
constructor accepts routing plugins but not deployment callbacks. In a long-running application,
register the object once during startup. The proxy YAML shown above performs both registrations
declaratively.

Install the locked example environment and run the complete example from this directory:

```bash
uv sync --locked --python 3.12
uv run --locked --env-file deployment/.env python examples/python_router.py
```

## Develop against a LiteLLM source checkout

The pinned container is the reproducible default. To work on LiteLLM itself, follow LiteLLM's
[local development setup](https://docs.litellm.ai/docs/extras/contributing_code#1-setting-up-your-local-dev-environment),
including its proxy dependencies, then install this Switchyard checkout into that environment.
Point the plugin loader at a routing TOML and LiteLLM at the matching model YAML:

```bash
export SWITCHYARD_LITELLM_CONFIG=/absolute/path/to/switchyard-new/examples/experimental/litellm/deployment/profiles/stage/switchyard.toml
PYTHONPATH=/absolute/path/to/switchyard-new/examples/experimental/litellm/src \
  uv run litellm --config \
  /absolute/path/to/switchyard-new/examples/experimental/litellm/deployment/profiles/stage/litellm.yaml
```

Keep LiteLLM at v1.97.0 when reproducing this example's verified behavior.

## Tests

From the repository root, run the offline suite without an API key or provider calls:

```bash
PYTHONPATH=examples/experimental/litellm/src \
  uv run --project examples/experimental/litellm --locked \
  pytest examples/experimental/litellm/tests -m "not e2e" -v
```

The paid suite has an explicit opt-in. It builds and starts the local proxy once for each profile
and verifies the concrete target through LiteLLM's `x-litellm-model-name` response header:

```bash
SWITCHYARD_LITELLM_E2E=1 \
  uv run --env-file examples/experimental/litellm/deployment/.env \
  --project examples/experimental/litellm --locked \
  pytest examples/experimental/litellm/tests/integration/test_e2e.py -m e2e -v
```

## Troubleshooting

- If Compose says the key is missing, place `OPENROUTER_API_KEY` in `deployment/.env`, or pass
  another file with `docker compose --env-file /path/to/.env -f deployment/compose.yaml ...`.
- If startup reports `SWITCHYARD_LITELLM_CONFIG`, verify that the selected profile contains a
  readable `switchyard.toml` and that its `algorithm` and fields match the tables above.
- If port 4000 is occupied, set `LITELLM_PORT` before starting Compose and use that port in requests.
- If the service is unhealthy, run `docker compose -f deployment/compose.yaml logs litellm`.
- If routing fails only when a request arrives, verify candidate ordering/count and Random weight
  count before checking the request's supported message shapes.

## Security

The bundled proxy is unauthenticated and binds only to loopback. It is intended for local
development. Local `.env` files are excluded from the Docker build context; keep credentials out of
YAML, TOML, source control, and command output. Configure LiteLLM authentication and follow its
production deployment guidance before exposing the service.

## References

- [LiteLLM routing plugins](https://docs.litellm.ai/docs/routing_plugins)
- [Switchyard decision-only API PR #459](https://github.com/NVIDIA-NeMo/Switchyard/pull/459)
- [Switchyard Python binding updates PR #479](https://github.com/NVIDIA-NeMo/Switchyard/pull/479)
- [Switchyard Stage routing](../../../docs/routing_algorithms/stage_router_routing.md)
- [Switchyard Random routing](../../../docs/routing_algorithms/random_routing.md)
