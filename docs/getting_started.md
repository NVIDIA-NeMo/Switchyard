# Getting Started with Switchyard

Add routing to your existing gateway or agent harness with **libsy**, Switchyard's Rust library. Your application keeps its model clients, credentials, and service.

This guide uses the Python bindings. Start with a local routing decision, then send the selected request through a model client. For Rust, see the [complete Rust example](reference/host_integration.md#route-in-rust).

## Install

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/). These examples use the released `nemo-switchyard==0.3.0` package. Supported wheels need no Rust toolchain; see [platform requirements](../INSTALLATION.md#requirements).

```bash
uv init --bare switchyard-demo
cd switchyard-demo
uv add "nemo-switchyard==0.3.0"
```

## Route a request

Save this complete example as `route.py`:

```python
import asyncio
from switchyard.libsy import Step, algorithms

async def main():
    router = algorithms.stage_router(
        picker="efficient_first", confidence_threshold=0.5)
    models = {"efficient": ["fast"], "capable": ["strong"]}
    models["any"] = ["fast", "strong"]
    text = {"type": "text", "text": "Hello"}
    request = {"messages": [{"role": "user", "content": [text]}]}
    async for step in router.run_stream(request, models):
        if isinstance(step, Step.Done):
            print("Selected:", step.outcome.selected_model_ids[0])

asyncio.run(main())
```

```bash
uv run route.py
```

Expected output:

```text
Selected: fast
```

`fast` and `strong` are host-supplied model IDs. `efficient` and `capable` assign their routing roles; `any` lists the full model pool. This example makes no network calls and needs no API key.

The Stage router reads conversation and tool-result signals. With `efficient_first` and no tool history, it selects the efficient model. The `0.5` value is a routing confidence threshold, not a measured probability of task success. This is the same preset the TOML runner calls **Auto**.

## Call your model

In an existing gateway, insert libsy before the model call:

```text
request -> libsy.run_stream -> selected model + returned request -> your model client
```

The following complete example uses the OpenAI Python client against OpenRouter. You can keep the client you already use in your application. Install the example's client dependency:

```bash
uv add openai
```

Set `OPENROUTER_API_KEY` in your environment. Your key needs access to `z-ai/glm-5.2` and `anthropic/claude-opus-4.8`, or replace both IDs with models you can use. These models also appear in our [Terminal-Bench 2.1 results](benchmarks.md#terminal-bench-21).

```bash
export OPENROUTER_API_KEY="your-openrouter-key"  # pragma: allowlist secret
```

Save as `gateway.py`:

```python
import asyncio
import os

from openai import AsyncOpenAI
from switchyard.libsy import Step, algorithms

async def main():
    router = algorithms.stage_router(
        picker="efficient_first", confidence_threshold=0.5)
    models = {
        "efficient": ["z-ai/glm-5.2"],
        "capable": ["anthropic/claude-opus-4.8"],
    }
    models["any"] = models["efficient"] + models["capable"]
    text = {"type": "text", "text": "Say hello in one sentence."}
    request = {"messages": [{"role": "user", "content": [text]}]}
    async with AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ) as client:
        async for step in router.run_stream(request, models):
            if isinstance(step, Step.Done):
                outcome = step.outcome
                print("Selected:", outcome.selected_model_ids[0])
                response = await client.chat.completions.create(
                    model=outcome.selected_model_ids[0],
                    messages=outcome.request["messages"],
                )
                print("Answer:", response.choices[0].message.content)

asyncio.run(main())
```

This command sends a billable provider request:

```bash
uv run gateway.py
```

The first line is `Selected: z-ai/glm-5.2`. The second starts with `Answer:` and contains the model's reply. A successful response verifies provider access as well as routing.

For this text-only example, the host sends the **selected model and returned messages**. User text blocks also work with Chat Completions. Switchyard's full request format is provider-neutral: instructions, tools, media, and other controls need a host adapter. Use the [LiteLLM integration](../examples/litellm/README.md), [NeMo Relay plugin](integrations/nemo_relay.md), or Rust [translation crate](../crates/switchyard-translation/README.md) for fuller integrations.

The Stage configuration above produces only `Step.Done`. Algorithms that ask a judge to choose a model also yield calls for the host to serve. See [Drive the algorithm](#drive-the-algorithm) before changing to one of those algorithms.

## See routing change

Routing can change between agent turns. Save as `tool_error.py` and run `uv run tool_error.py`. This check stays local and needs no credentials:

```python
import asyncio
from switchyard.libsy import Step, algorithms

async def main():
    router = algorithms.stage_router(picker="efficient_first", confidence_threshold=0.5)
    models = {"efficient": ["fast"], "capable": ["strong"], "any": ["fast", "strong"]}
    messages = [{"role": "user", "content": [{"type": "text", "text": "Run the tests."}]}]
    async for step in router.run_stream({"messages": messages}, models):
        if isinstance(step, Step.Done):
            print("Fresh task:", step.outcome.selected_model_ids[0])

    messages += [
        {"role": "assistant", "content": [{"type": "tool_call", "id": "c1",
            "name": "bash", "arguments": {"command": "pytest -q"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_call_id": "c1",
            "is_error": True, "content": [{"type": "text",
                "text": "MemoryError: cannot allocate memory\nKilled"}]}]},
    ]
    async for step in router.run_stream({"messages": messages}, models):
        if isinstance(step, Step.Done):
            print("Critical tool error:", step.outcome.selected_model_ids[0])

asyncio.run(main())
```

Expected output:

```text
Fresh task: fast
Critical tool error: strong
```

The critical tool error forces the capable model. Ordinary requests without usable tool signals stay on the efficient model in this preset. This checks model selection; it does not measure answer quality or cost savings. Evaluate your agent and model pool against a single-model baseline before relying on savings.

## Drive the algorithm

`run_stream` is the integration boundary. It yields routing steps, ending with `Step.Done` and a `RoutingOutcome`:

| Value | Host action |
| --- | --- |
| `selected_model_ids` | Use the first model; later entries are ordered fallbacks. |
| `request` | Use this request after routing-time rewrites. |
| `response` | Reuse this answer when the algorithm already produced one. Otherwise make the answer call. |

The examples above use Stage without a judge, so `response` is absent. For classifier calls, streaming, concurrent work, cancellation, and metadata, see the [host integration guide](reference/host_integration.md). That guide follows the development API on `main`; the examples on this page use the released package.

## Next steps

- **Embed in Rust:** [complete example](reference/host_integration.md#route-in-rust) and the [Rust API reference](reference/rust_api.md).
- **Use a gateway integration:** [LiteLLM](../examples/litellm/README.md) or [NeMo Relay](integrations/nemo_relay.md). Check each guide's supported algorithms and versions.
- **Choose another algorithm:** [routing overview](routing_algorithms/overview.md).
- **Try the standalone proxy:** [local evaluation guide](operations/local_evaluation.md), with model configuration, requests, and routing checks. The server is for demos, evaluation, and personal use, not a supported production path.
- **Use an agent directly:** [Claude Code example](../examples/claude/README.md), [pi](integrations/pi.md), or [Oh My Pi](integrations/oh_my_pi.md).
