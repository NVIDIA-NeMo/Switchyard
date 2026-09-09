# Ensemble Routing

Ensemble routing calls two to four candidate targets concurrently, buffers the
successful responses, and asks a synthesizer target to produce one final answer.
It is response-level fusion: Switchyard does not merge model weights or logits.

Use it when answer quality can justify several candidate calls plus synthesis.
Compared with ordinary routing, every request costs at least three model calls
and waits for every candidate call to finish before synthesis begins.

## Configure an ensemble

Declare every candidate and the synthesizer as normal targets, then reference
their target names from the route:

```toml
schema_version = 1

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.fast]
id = "provider/fast-model"
llm_client = "openrouter"

[targets.reasoning]
id = "provider/reasoning-model"
llm_client = "openrouter"

[targets.synthesizer]
id = "provider/synthesis-model"
llm_client = "openrouter"

[routes.fused]
id = "fused"
type = "ensemble"
candidates = ["fast", "reasoning"]
synthesizer_target = "synthesizer"
minimum_successful_candidates = 2
candidate_max_output_tokens = 2048
```

Clients send `fused` as the model ID. Switchyard sends the original request to
both candidates. The synthesizer then receives the original conversation plus
the normalized outputs of every successful candidate. Internal reasoning blocks
are removed before synthesis; final text, refusals, and tool calls are retained.

Candidate calls are always buffered because their partial tokens are not sent to
the caller. `candidate_max_output_tokens` gives each internal candidate an
independent budget, which may be higher or lower than the final response budget.
The final synthesis keeps the caller's original streaming and output settings.

The packaged synthesis prompt asks for a concise, complete answer that follows
the caller's length and format constraints. Set `synthesizer_system_prompt` when
the application needs a domain-specific rubric or output style.

## Failure behavior

A failed or empty candidate is omitted from synthesis. By default the route
continues when at least one candidate produces usable output. Set
`minimum_successful_candidates` to require more contributors; synthesis fails
when fewer usable candidates remain. A synthesizer failure fails the request
normally.

Candidate responses must finish before synthesis, so callers do not receive
candidate tokens as they arrive. When the original request asks for streaming,
the synthesizer's final response can still stream to the caller.

## Limits

- Configure between two and four candidates.
- Candidate ordering does not express priority; calls run concurrently.
- Tool definitions from the original request remain available to the synthesizer.
