# Oracle Cloud Infrastructure Generative AI

Route between [OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm)
models through the service's OpenAI-compatible endpoint. The released `v0.3.0` server and
Python bindings work unchanged: OCI speaks the `openai_chat` and `openai_responses`
formats, and a Generative AI API key is sent as a bearer token.

## Endpoint and credential

| Setting | Value |
|---|---|
| `base_url` | `https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1` (for example `us-chicago-1`) |
| `api_key_env` | the variable holding an OCI Generative AI API key, `sk-...` |
| Model IDs | OCI's `vendor.model` form, for example `meta.llama-4-scout-17b-16e-instruct`, `xai.grok-4.7`, `openai.gpt-5.5`, `google.gemini-2.5-flash` |

The key is scoped to the compartment it was created in, so requests carry no
compartment header. Create the IAM policy **before** the key: a key minted before
its policy can stay unauthorized long after the policy lands, and OCI returns the
same 401 for an unknown key and an unauthorized one.

```bash
# 1. Policy, in the tenancy's home region. Least privilege: one compartment.
oci iam policy create --compartment-id <compartment-ocid> --name genai-api-keys \
  --description "Generative AI API keys may call inference" \
  --statements '["allow any-user to use generative-ai-family in compartment id <compartment-ocid> where ALL {request.principal.type='"'"'generativeaiapikey'"'"'}"]'

# 2. Key, in the inference region, with an expiry. The sk-... secret is returned once.
oci generative-ai api-key create --compartment-id <compartment-ocid> --region us-chicago-1 \
  --display-name switchyard --key-details '[{"keyName":"primary","timeExpiry":"2027-01-01T00:00:00Z"}]'
export OCI_GENAI_API_KEY=sk-...
```

`GET /openai/v1/models` is not served; list the region's catalog with
`oci generative-ai model-collection list-models --compartment-id <ocid> --region <region>`.
The catalog can list models that are not served on demand; those return
404 `Entity with key <model> not found`.

## Task routing between an efficient and a capable model

Save as `oci-routing.toml`. Meta, xAI and Google models answer on Chat
Completions. OpenAI `gpt-5.x` and `gpt-6.x` models reject `max_tokens` there, so
address them through the Responses API, where the limit is `max_output_tokens`.

```toml
schema_version = 1

[llm_clients.oci_chat]
format = "openai_chat"
base_url = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
api_key_env = "OCI_GENAI_API_KEY"
timeout_ms = 60000

[llm_clients.oci_responses]
format = "openai_responses"
base_url = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
api_key_env = "OCI_GENAI_API_KEY"
timeout_ms = 120000

[targets.judge]
id = "openai.gpt-5.4-mini"
llm_client = "oci_responses"

[targets.efficient]
id = "meta.llama-4-scout-17b-16e-instruct"
llm_client = "oci_chat"

[targets.capable]
id = "openai.gpt-5.5"
llm_client = "oci_responses"

[routes.task]
id = "oci-task"
type = "llm_classifier"
mode = "capability"
classifier_target = "judge"
strong_target = "capable"
weak_target = "efficient"
base_threshold = 0.5
threshold_step = 0.1
```

Start the server and send a request with `model = "oci-task"`:

```bash
switchyard-server --config oci-routing.toml --dry-run
switchyard-server --config oci-routing.toml --host 127.0.0.1 --port 4123
```

For agents, add a [stage router](../routing_algorithms/stage_router_routing.md)
over the same targets; it reads tool results and needs no judge call:

```toml
[routes.execution]
id = "oci-execution"
type = "stage_router"
capable_target = "capable"
efficient_target = "efficient"
picker = "efficient_first"
confidence_threshold = 0.5
```

## From Python

The released bindings run the same algorithms. Your code serves the judge call and
the answer with any OpenAI client pointed at the endpoint; see
[Host Integration](../reference/host_integration.md).

```python
from openai import AsyncOpenAI
from switchyard.libsy import LlmClassifierConfig, LlmResponse, Step, TaskClassifierConfig, algorithms

client = AsyncOpenAI(
    base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1",
    api_key=os.environ["OCI_GENAI_API_KEY"],
)
models = {"judge": ["openai.gpt-5.4-mini"], "efficient": ["meta.llama-4-scout-17b-16e-instruct"],
          "capable": ["openai.gpt-5.5"], "any": ["meta.llama-4-scout-17b-16e-instruct", "openai.gpt-5.5"]}
algorithm = algorithms.llm_classifier(LlmClassifierConfig.capability(
    config=TaskClassifierConfig(0.5, threshold_step=0.1, max_output_tokens=512, response_format_type="json_schema")))

async for step in algorithm.run_stream(request, models):
    if isinstance(step, Step.CallModel):
        verdict = await client.chat.completions.create(**judge_body(step.call.request))  # your translation
        step.call.respond(LlmResponse.Agg({"model": step.call.models[0], "outputs": [
            {"role": "assistant", "content": [{"type": "text", "text": verdict.choices[0].message.content}],
             "stop_reason": "end_turn"}]}))
    elif isinstance(step, Step.Done):
        selected = step.outcome.selected_model_ids[0]   # then call it through the same client
```

## Notes

- Streaming, tool calling, vision input, the Responses API and `openai.text-embedding-3-small`
  embeddings work on this endpoint; Cohere embeddings do not (`400 Unsupported OpenAI operation`).
- Set `timeout_ms` on the judge's client so an unresponsive judge returns `504` instead of waiting.
- Cost estimates use OCI's own model IDs, which are not in the packaged price table;
  supply an override with OCI's list prices if you want cost figures.
- Dedicated AI Cluster endpoints and signed OCI principals use the native OCI API, which
  needs request signing. Put a gateway that signs for OCI (for example LiteLLM with its
  OCI provider) in front and point Switchyard's clients at the gateway instead.
- Other realms use their realm domain in place of `oraclecloud.com`.
