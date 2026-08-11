# Routing to a self-hosted model

Every deployment example in this repository points at a hosted provider. This one
puts the **weak tier on a local OpenAI-compatible server** — vLLM, NIM, Ollama or
llama.cpp — and keeps a hosted model for the hard turns, so routine steps never
leave the machine.

## Files

| File | Purpose |
|---|---|
| `routes.toml` | Hybrid deployment: local weak + local classifier, hosted strong. |

## Run it

Start the local server, giving it a stable name to route to:

```bash
vllm serve <checkpoint> --served-model-name my-local-model --port 8000
```

Point the deployment at it and check the wiring without sending traffic:

```bash
export OPENROUTER_API_KEY=...
switchyard-server --config routes.toml --dry-run
switchyard-server --config routes.toml --host 127.0.0.1 --port 4000
```

Then send the route id as the model name:

```bash
curl -s http://127.0.0.1:4000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local-first","messages":[{"role":"user","content":"hello"}]}'
```

## Notes

- **`id` must match `/v1/models` on the local server**, not the checkpoint path.
  With vLLM that is whatever `--served-model-name` was set to; unset, it defaults
  to the full path you passed to `vllm serve`, which is rarely what you want in a
  config file.
- **`api_key_env` is omitted for the local client.** Omitting it sends no
  authentication, which is what a server started without `--api-key` expects. Add
  it back if yours requires a key — the value never belongs in the TOML.
- **The classifier runs locally** so the per-turn routing decision costs nothing
  on the metered path. Point `classifier_target` at the hosted client if you would
  rather trade that cost for a stronger classifier.
