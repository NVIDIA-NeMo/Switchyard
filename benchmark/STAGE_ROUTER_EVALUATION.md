# Stage-router evaluation plan

This plan compares the current stage router with two fixed-model baselines on
DeepSWE v1.1. It measures task success, cost, and latency after the clean-production
[scoring change](https://github.com/NVIDIA-NeMo/Switchyard/pull/984).
It has no live results yet.

The earlier offline replay used fixed trajectories. Live routing changes later
model outputs and tool calls, so that replay cannot establish task accuracy or cost.

## Freeze the inputs

Use the same dataset revision, task selection, model IDs, agent version, network
policy, and timeouts for every arm. Start with the versions and budgets in
[the qualification settings](DEEPSWE_V11_QUALIFICATION.md), then record any change.
The OpenRouter profiles adapt the original provider settings; compare all four
arms through the same provider rather than comparing them to published scores.

Before running, save:

- Switchyard commit and whether its working tree is clean.
- Dataset commit and the exact selected task IDs.
- Pier, Harbor, Codex CLI, Docker, Python, and Rust versions.
- Each generated config and its SHA-256.
- Provider model IDs, generation defaults, pricing snapshot, and account limits.
- Task, command, sandbox, and lifecycle timeouts; infrastructure retry policy.

Keep credentials out of configs and result bundles. Record the pricing source and
date, including input, output, cache-read, and cache-write rates where applicable.

## Generate the four arms

Run this from the repository root with Python 3.12 or newer. It writes under
`stage-eval/`, leaving the qualification profile unchanged. Each server exposes the
same public route ID, so the agent's requested model stays fixed across arms.

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import subprocess
import tomllib

source = Path('benchmark/routing-profiles/deepswe-v11-stage-router-luna-sol.toml')
text = source.read_text()
config = tomllib.loads(text)
route = config['routes']['switchyard']
assert route['picker'] == 'efficient_first'
assert route['confidence_threshold'] == 0.5
deployment = text[text.index('schema_version = 1'):]
prefix, stage = deployment.split('[routes.switchyard]\n', 1)
stage = '[routes.switchyard]\n' + stage
root = Path('stage-eval')
root.mkdir(exist_ok=True)
manifest = {'switchyard_commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'configs': {}}
for arm in ['efficient', 'capable', 'efficient-first', 'capable-first']:
    if arm in ['efficient', 'capable']:
        body = ('[routes.switchyard]\nid = "gpt-5.6-luna"\n'
                f'type = "passthrough"\ntarget = "{arm}"\n')
    else:
        body = stage.replace('picker = "efficient_first"',
                             f'picker = "{arm.replace("-", "_")}"', 1)
    content = prefix + body
    tomllib.loads(content)
    path = root / f'{arm}.toml'
    path.write_text(content)
    manifest['configs'][arm] = hashlib.sha256(content.encode()).hexdigest()
(root / 'configs.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
```

Both stage arms retain the qualification profile's five-result window, two-turn
capable hold, and handoff settings. They have no LLM judge. The fixed-model arms
use passthrough routes without stage handoff notes.

Build and validate without provider calls:

```bash
cargo build --locked -p switchyard-server
for arm in efficient capable efficient-first capable-first; do
  OPENROUTER_API_KEY=config-validation-only target/debug/switchyard-server \
    --config "stage-eval/$arm.toml" --dry-run
done
```

A dry run checks configuration loading. It does not establish provider acceptance
or model availability.

## Execute a bounded pilot

Use the [DeepSWE setup and Pier commands](README.md#deepswe-v11). Start with ten
sampled tasks and one trial per arm: 40 task attempts, plus any infrastructure
retries. Keep the same sample seed and verify the selected task IDs match across
arms. Run one arm at a time with a fresh server and separate result directories.
Use the qualification timeouts and network policy consistently.

For the selected arm, start the server with a real provider key supplied through
the environment:

```bash
arm=capable-first
mkdir -p "stage-eval/$arm"
target/debug/switchyard-server --config "stage-eval/$arm.toml" \
  --host 0.0.0.0 --port 443 \
  --routing-log-file "stage-eval/$arm/routing.jsonl"
```

In another terminal, run the same task selector for each arm:

```bash
pier run -p benchmark/datasets/deep-swe/tasks --agent codex \
  --model openai/gpt-5.6-luna --n-tasks 10 --sample-seed 0 \
  --ae OPENAI_BASE_URL=http://host.docker.internal:443/v1 \
  --ae OPENAI_API_KEY=unused
```

The host address and port follow the existing Docker Desktop example; use the
host's Docker-bridge address on Linux. Archive each Pier job before changing arms,
and preserve task IDs, verifier results, agent traces, server logs, and configs.
The network and timeout settings require the same harness setup as qualification;
the short command alone does not enforce them.

Use the pilot's measured tokens to estimate the cost of further runs. Select a
spending cap before live execution. If the pilot exceeds it, stop and report the
incomplete coverage. A successful pilot can justify a larger paired run with
repeated trials; ten tasks alone are too few for a reliable quality claim.

## Report the result

For every arm, report these quantities with their denominators:

- Strict success: reward-1 tasks divided by all selected tasks, counting missing
  verifier results and infrastructure failures as unsolved. Also list those
  failures separately.
- Measured token usage and cost, including routing-time calls, retries, and
  fallback attempts when the provider or harness exposes them. Mark unobserved
  charges as missing rather than zero.
- Task wall time and request latency distributions, with timeout and failure counts.
- Capable share by requests and tokens, stating whether counts use selected or
  actually served models. Report fallback separately.
- Paired task outcomes: both solved, only the router solved, only the fixed capable
  model solved, and neither solved.

Keep offline replay and live results in separate tables. Publish the frozen
inputs, deviations, missing data, and pilot cost estimate with the report. Compare
quality and spending against both fixed arms before recommending a picker or
threshold.
