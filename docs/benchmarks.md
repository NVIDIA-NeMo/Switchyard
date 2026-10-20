# Benchmarks

Task completion, cost, and runnable configs. All configs use `OPENROUTER_API_KEY`.

[Terminal-Bench 2.1](#terminal-bench-21) · [DeepSWE](#deepswe) · [Run a benchmark](#run-a-benchmark)

## Terminal-Bench 2.1

**Claude Opus 4.8 + GLM 5.2**

![Terminal-Bench 2.1 task completion versus cost.](assets/switchyard-cost-accuracy.svg)

| Strategy | Tasks solved | Cost per run | Config |
| --- | ---: | ---: | --- |
| LLM classifier | 79.40% | $109.04 | — |
| Stage | 79.03% | $109.81 | — |
| Escalation | 79.80% | $122.63 | [TOML](../benchmark/routing-profiles/tb21-escalation-opus-glm-deepseek.toml) |
| Jev (replay) | 82.02% | ≈$92.12 | [TOML](../benchmark/routing-profiles/tb21-jev-opus-glm.toml) |
| Opus 4.8 only | 77.15% | $148.83 | — |
| GLM 5.2 only | 74.16% | $71.31 | — |

## DeepSWE

**v1.1 · 113 tasks · Codex CLI · GPT-5.6 Luna + Sol**

![DeepSWE task completion versus cost.](assets/deepswe-cost-accuracy.svg)

| Strategy | Tasks solved | Cost per run | Config |
| --- | ---: | ---: | --- |
| Plan/Execute | 60.8% | $180.51 | [TOML](../benchmark/routing-profiles/deepswe-v11-plan-execute-luna-sol.toml) |
| Advisor | 57.5% | $208.00 | [TOML](../benchmark/routing-profiles/deepswe-v11-advisor-gate-luna-sol.toml) |
| Stage | 61.36% | $466.28 | [TOML](../benchmark/routing-profiles/deepswe-v11-stage-router-luna-sol.toml) |
| Sol only | 62.8% | $720.95 | — |
| Luna only | 52.2% | ≈$49 | — |

Luna: one run; others: three-run means. [Evaluation settings](../benchmark/DEEPSWE_V11_QUALIFICATION.md).

## Run a benchmark

- **Terminal-Bench 2.1:** [Setup](../benchmark/README.md#one-time-setup) · [Run](../benchmark/README.md#run-with-switchyard-routing)
- **DeepSWE:** [Run with Pier](../benchmark/README.md#deepswe-v11)
