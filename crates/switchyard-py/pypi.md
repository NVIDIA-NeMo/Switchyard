<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA-NeMo/Switchyard/main/assets/logo.png" alt="Switchyard" width="800">
</p>

# NVIDIA NeMo Switchyard

<p align="center">
  <a href="https://github.com/NVIDIA-NeMo/Switchyard/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NVIDIA-NeMo/Switchyard" alt="License: Apache-2.0"></a>
  <a href="https://github.com/NVIDIA-NeMo/Switchyard/releases"><img src="https://img.shields.io/github/v/release/NVIDIA-NeMo/Switchyard" alt="Latest GitHub release"></a>
  <a href="https://pypi.org/project/nemo-switchyard/"><img src="https://img.shields.io/pypi/v/nemo-switchyard" alt="PyPI: nemo-switchyard"></a>
  <a href="https://crates.io/crates/switchyard-libsy"><img src="https://img.shields.io/crates/v/switchyard-libsy?label=crates.io%20%28libsy%29" alt="crates.io: switchyard-libsy"></a>
  <a href="https://crates.io/crates/switchyard-server"><img src="https://img.shields.io/crates/v/switchyard-server?label=crates.io%20%28server%29" alt="crates.io: switchyard-server"></a>
  <a href="https://github.com/NVIDIA-NeMo/Switchyard/graphs/contributors"><img src="https://img.shields.io/github/contributors/NVIDIA-NeMo/Switchyard" alt="GitHub contributors"></a>
</p>

Switchyard is an open-source library that helps an AI agent choose which model handles each request. It combines efficient models with more capable models so you can balance task success, cost, and latency on your workload.

Use Switchyard through a gateway integration, try it with a local proxy, or embed it in your own harness. You choose the model pool. Switchyard supplies the routing decision. Your gateway or application owns the surrounding service.

## How it works

1. Configure the models your application can use and choose a routing algorithm.
2. The algorithm examines the request or the agent's recent tool activity. Some algorithms call a model to judge the task.
3. Your gateway or application sends the request to the selected model. Routing can change as the agent continues its work.

Evaluate the complete agent, model pool, and routing configuration against your single-model baseline. A cheaper model call does not guarantee a cheaper successfully completed task.

## Embed the library in your harness

Refer to the [embedding example](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/examples/libsy.py).

## Routing algorithms

Start with Auto. Choose Task, Execution, or Composite when you need more control. These names describe routing choices. The configuration keys are unchanged.

| Choice | How it chooses | TOML configuration |
| --- | --- | --- |
| **[Auto](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/overview/#auto)** | Uses the current default: Execution (Stage), efficient-first, with a confidence threshold of 0.5 and no classifier call. | `type = "auto"` |
| **[Task](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/llm_classifier_routing/)** | A model judges whether the efficient model can handle the task. | `type = "llm_classifier"`, `mode = "capability"` |
| **[Execution](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/stage_router_routing/)** | Uses recent tool activity and outcome signals to choose a model as the agent runs. | `type = "stage_router"` |
| **[Composite](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/composite_routing/)** | Combines Task and Execution: a classifier sets the default model tier when execution signals are uncertain. | `type = "composite"` |

Auto is a fixed preset in v0.3.0. It does not compare strategies at runtime. The TOML runner supports `type = "auto"`. For direct Python embedding, use `stage_router(picker="efficient_first", confidence_threshold=0.5)` for the same preset.

The [routing overview](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/overview/) retains the full catalogue, including [Plan/Execute](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/plan_execute_routing/), [escalation](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/escalation_router_routing/), [advisor](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/advisor_gate_routing/), [sub-agent](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/subagent_routing/), and [random](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/random_routing/) strategies. Integration support varies: check the gateway guide before choosing an algorithm.

[Prefill routing](https://nvidia-nemo.github.io/Switchyard/routing_algorithms/prefill_routing/) is designed for fast, efficient model selection using a model's prefill activations and a classifier trained on your own evaluation data.

## Evaluation and reference

[Compare benchmark results and configs](https://nvidia-nemo.github.io/Switchyard/benchmarks/) for Terminal-Bench 2.1 and DeepSWE.
For latency and routing overhead testing, see [Soak Testing](https://nvidia-nemo.github.io/Switchyard/operations/soak_test/).

### Further reading

- [Benchmark setup and profiles](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/benchmark/README.md): run your own baseline and routing comparisons.
- [Core concepts](https://nvidia-nemo.github.io/Switchyard/core_concepts/): clients, targets, routes, and model IDs.
- [TOML schema](https://nvidia-nemo.github.io/Switchyard/reference/toml_schema/): configuration fields and defaults.
- [Architecture](https://nvidia-nemo.github.io/Switchyard/architecture/): library and runtime components.
- [Installation](https://github.com/NVIDIA-NeMo/Switchyard/blob/main/INSTALLATION.md): package and platform requirements.
