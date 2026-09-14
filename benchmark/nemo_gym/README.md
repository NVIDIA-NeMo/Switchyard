# Evaluate Switchyard routing with NeMo Gym

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is a library for evaluating and improving models and agents, combining infrastructure for developing environments and running evaluation and training at scale with popular benchmarks and training environments.

This tutorial uses [MMLU-Redux 2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) to compare a fixed model with Switchyard routing.

Gym provides the evaluation substrate: it supplies tasks, runs the agent, and verifies answers to report rewards.
Switchyard sits in the model-request path, selecting which upstream model serves each request.

![Gym evaluation through LiteLLM and Switchyard Random routing](architecture.svg)

## Understand the wiring

Gym calls a LiteLLM endpoint through its `litellm_model` adapter. The Switchyard library is integrated in LiteLLM, which is the path we'll use in this example. As such, Switchyard does not run as a separate server here.

**LiteLLM defines the model groups.** This excerpt from [litellm.yaml](litellm.yaml) shows the candidates; provider settings are omitted here:

```yaml
model_list:
  - model_name: fixed
    litellm_params: {model: nvidia_nim/nvidia/nemotron-3-super-120b-a12b}
  - model_name: routed
    litellm_params: {model: nvidia_nim/nvidia/nemotron-3-super-120b-a12b}
  - model_name: routed
    litellm_params: {model: nvidia_nim/openai/gpt-oss-20b}
```

`fixed` and `routed` are names we chose, not special Gym modes. The fixed group has one candidate, so it always uses Super. The routed group has two candidates for Switchyard to choose from.

**Switchyard defines the selection policy.** [routes.toml](routes.toml) contains:

```toml
algorithm = "random"
seed = 6
```

LiteLLM uses the [Switchyard integration](../../examples/litellm/README.md) to choose a model using `routes.toml`. A small adapter records each choice and makes the response compatible with Gym.

**Gym requests a group, not a concrete model.** These excerpts show the model wiring inside the runner; **these are wrapped in a [run.sh](./run.sh) script we'll run later, and are not additional steps to execute in this tutorial**:

```text
gym eval run --benchmark mmlu-redux --model-type litellm_model --model fixed \
  ++policy_base_url=http://127.0.0.1:4000/v1 ++policy_api_key=unused

gym eval run --benchmark mmlu-redux --model-type litellm_model --model routed \
  ++policy_base_url=http://127.0.0.1:4000/v1 ++policy_api_key=unused
```
Note the following
-  `--model-type` selects the adapter
- `policy_base_url` points it at LiteLLM
- `--model` selects the group. The local proxy uses provider credentials from the environment, not Gym's placeholder key.
- The runner adds identical task limits and separate output/capture paths. The dataset, agent, verifier, temperature, and 4,096-token answer limit stay unchanged.

## 1. Set up

You need:

- Bash on Linux/macOS
- Git and curl
- [uv](https://docs.astral.sh/uv/)
- The [Rust toolchain prerequisites](../../docs/getting_started.md#prerequisites) for the current checkout bindings
- An NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/) with access to `openai/gpt-oss-20b` and `nvidia/nemotron-3-super-120b-a12b`

Run these commands in Bash from the Switchyard repository root. These one-time commands create a Gym checkout under `scratch/` at the tested `v0.6.0` release. Choose an unused `GYM_DIR` without spaces or shell metacharacters.

```bash
export GYM_DIR="$PWD/scratch/nemo-gym-litellm/Gym"
mkdir -p "$(dirname "$GYM_DIR")" &&
git clone https://github.com/NVIDIA-NeMo/Gym.git "$GYM_DIR" &&
git -C "$GYM_DIR" checkout v0.6.0 &&
uv tool run --from uv==0.11.29 uv sync \
  --directory "$GYM_DIR" --frozen --no-dev --python 3.13.14 &&
uv tool run --from uv==0.11.29 uv pip install --no-deps \
  --python "$GYM_DIR/.venv/bin/python" uv==0.11.29 &&
"$GYM_DIR/.venv/bin/uv" sync --project examples/litellm --locked --python 3.12
```

Gym and LiteLLM use separate Python environments. The proxy builds Switchyard bindings from this checkout; the native CLI workflow does not need Docker.

## 2. Run both conditions

The default is five tasks per condition, normally ten upstream calls. Inference can consume credits, and retries can add calls. Replace the placeholder below with your NVIDIA API key, then run. Pasting a key into this command may save it in shell history.

```bash
export NVIDIA_API_KEY="your-api-key"
bash benchmark/nemo_gym/run.sh
```

The [runner](./run.sh) prepares MMLU-Redux, starts the local LiteLLM proxy, evaluates fixed then routed, stops the proxy, and prints the comparison. It saves `comparison.txt` and other artifacts under `benchmark/nemo_gym/results/<timestamp>/`. Keep this unauthenticated development proxy local; do not share or publicly expose it.

In `run.sh`, one loop runs both conditions: only the model group and output paths change. Leave the configuration and prepared data unchanged until both runs finish.

Gym components can outlive the command briefly; let them finish shutting down before an immediate rerun.

## 3. Read the comparison

Start with coverage and selected models, then compare rewards, tokens, and latency. Here is an excerpt from the **two-task local stub test**, not a real-model benchmark:

```text
Pairing: matched=2, fixed-only=0, routed-only=0

Metric                                 fixed         routed
Paired rollouts                            2              2
Mean reward                            0.500          0.500
Terminal-answer tokens                    30             30
Gateway-reported tokens                   30             30

fixed selected models: {"nvidia_nim/nvidia/nemotron-3-super-120b-a12b": 2}
routed selected models: {"nvidia_nim/nvidia/nemotron-3-super-120b-a12b": 1, "nvidia_nim/openai/gpt-oss-20b": 1}
```

- **Pairing:** both conditions completed the same two tasks. Incomplete or invalid evidence is rejected instead of producing partial averages.
- **Selection:** fixed stayed on Super; routed used both models. A small Random run need not split evenly.
- **Reward and usage:** MCQA scores the boxed answer letter (correct = 1, wrong = 0). Token columns sum input and output tokens across rollouts, not just generated answers. The stub supplies answers and token counts, so these values demonstrate the report, not model quality or savings.

For real runs, weigh reward against usage and latency rather than treating fewer tokens as a win by itself. Tokens are not dollar costs. The default five-task prefix is a smoke test, not a representative MMLU-Redux score; Random is not capability-based routing.

## 4. Try a small change

- **Workload:** use a fresh results directory and adjust the task count:

  ```bash
  RESULTS_DIR="$PWD/benchmark/nemo_gym/results/my-run" \
    LIMIT=2 bash benchmark/nemo_gym/run.sh
  ```

- **Models:** edit [litellm.yaml](litellm.yaml), keeping one fixed candidate and two distinct routed candidates, including the fixed model. Keep per-model settings identical between conditions.
- **Routing:** change the seed in [routes.toml](routes.toml), keeping `algorithm = "random"`. A seed repeats assignments only for an identical request sequence; retries or concurrency can change them.
- **Another benchmark:** use `--benchmark NAME` in your own Gym calls against LiteLLM, with a compatible agent and verifier. This runner and comparator are MMLU-Redux-specific, not a general benchmark launcher.

See `bash benchmark/nemo_gym/run.sh --help` for profile, port, repeat, and concurrency options.

<details>
<summary>Saved files and token counts</summary>

Each `fixed/` and `routed/` folder contains:

- `rollouts.jsonl`: model responses and rewards.
- `rollouts_materialized_inputs.jsonl`: the tasks and settings used.
- `rollouts_failures.jsonl`: failed tasks, if any.
- `model-calls/` and `litellm-calls.jsonl`: model request logs.
- `run-provenance.json`: version and configuration details.

If something fails, start with that folder's `gym.log` or the result folder's `litellm.log`.

The "Gateway-reported tokens" column includes the "Terminal-answer tokens", so don't add them together. Missing token counts are unknown, not zero, and some provider retries may not appear in the totals. Random makes no classifier calls; routing time is already included in rollout latency.

To view the comparison again without calling the models, replace `my-run` with your results folder:

```bash
"$GYM_DIR/.venv/bin/python" benchmark/nemo_gym/compare.py \
  benchmark/nemo_gym/results/my-run/fixed benchmark/nemo_gym/results/my-run/routed
```

Keep the saved inputs because the source dataset can change. Request logs contain prompts and responses, so review them before sharing.

</details>

**Tested:** with Gym `v0.6.0`.
