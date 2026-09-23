# GLiNER router sidecar

This example uses GLiNER as a fast, local classifier for Switchyard's custom
multi-target router. Switchyard still owns schema validation, fallback order,
session affinity, and final dispatch. The sidecar only returns a typed route
decision and confidence.

The configured GLiNER labels are deliberately different from the public route
names. This reduces the effect of prompts that try to select a route by spelling
its name. The optional confidence threshold can send low-confidence decisions
to the configured human-review route, but the included benchmark shows why it
must be calibrated on deployment-specific traffic rather than assumed safe.

## Run

Create an isolated Python environment and install GLiNER2 with its local-model
dependencies. Then start the classifier:

```console
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "gliner2[local] @ git+https://github.com/fastino-ai/GLiNER2@3c913c7369301133d3b7699252074c4303ada50e" \
  "protobuf>=5,<7"
.venv/bin/python examples/gliner-router/server.py \
  --routes examples/gliner-router/routes.json
```

Start Switchyard with `examples/gliner-router/switchyard.toml` after replacing
the example completion backend and target model IDs with the deployment's real
values. Send requests to model `switchyard/gliner` as usual.

`routes.json` is the policy boundary. Its aliases and descriptions control the
model's classification task, while `target` values must match the custom groups
in `switchyard.toml`. Keep consequential actions behind deterministic checks or
human approval even when classifier confidence is high.

The example target names describe routing intent; they do not create tool or
human workflows by themselves. In a deployment, `deterministic_tool` should
point at an API-backed deterministic handler and `human_review` at a service
that queues or gates work for a person.

## Test

The contract tests use a fake classifier and do not download a model:

```console
python -m unittest examples/gliner-router/test_server.py
```

`benchmark.py` runs the labeled ordinary, adversarial, and ambiguous routing
suite against a deployed GLiNER sidecar. `benchmark_typesafe.py` runs the same
cases against TypeSafe Jev when its API key and URL are supplied through the
environment. Benchmark output contains decisions and measurements, never
credentials.

The sidecar accepts only `POST /v1/chat/completions`. It validates the target
enum supplied by Switchyard before returning a decision and does not log prompt
content.
