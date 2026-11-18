# DeepSWE v1.1 qualification settings

Routing configurations:

- [Advisor gate: Luna executor with Sol reviews](routing-profiles/deepswe-v11-advisor-gate-luna-sol.toml)
- The [historical Plan/Execute configuration](#historical-planexecute-configuration) uses Sol as the planner and Luna as the executor.
- [Stage router: Luna efficient tier with Sol capable tier](routing-profiles/deepswe-v11-stage-router-luna-sol.toml)

## Hosted Harbor profiles

Both snapshots use Harbor 0.13.2 on Kubernetes with prepared task images and a
preinstalled Codex 0.154.0 bundle.

- The [September 10 baseline profile](harbor-profiles/deepswe-v11-harbor-0.13.2-20260910.yaml) records
  the Sol direct baseline setup. Its Codex config leaves hosted web search at the
  client default.
- The [October 6 repaired profile](harbor-profiles/deepswe-v11-harbor-0.13.2-20261006.yaml) selects the
  task's cached pnpm version, enables IPv6 loopback, runs Vitest browser tests
  in CI mode and disables Codex web search. Its matching hosted oracle passed all 113 tasks.

The hosted profiles capture committed and uncommitted edits with a temporary Git
index. An empty capture preserves the last nonempty patch, even if the agent has
since reverted its edits. The [local recipe](README.md#deepswe-v11) grades only
changes committed to `HEAD`.

Set `agents[0].model_name`, `OPENAI_BASE_URL` and reasoning effort for each run.
Use the provider model and endpoint for direct runs, or the route ID and Switchyard
endpoint for routed runs. The hosted runtime supplies the task, image and cluster
variables. For closed-book runs, allow only inference traffic and block hosted
web tools at the gateway.

## Shared settings

| Setting | Value |
| --- | --- |
| Workload | DeepSWE v1.1, all 113 tasks, benchmark revision 1 |
| Agent | Codex CLI 0.154.0 |
| Harness | Harbor 0.13.2 |
| Protocol | OpenAI Responses |
| Network | Closed book with public network access denied |
| Agent budget | 3 hours |
| Timeout multiplier | 2.0 |
| Command timeout | 10,800 seconds |
| Sandbox timeout | 8,400 seconds |
| Lifecycle timeout | 21,600 seconds |
| Infrastructure retries | 2 |
| Step and cost limits | Disabled |
| Strict scoring | Reward `1` divided by 113; failures and missing results score zero |
| Pricing snapshot | OpenRouter standard list prices from 2026-09-09, promotional pricing excluded |
| Pricing SHA-256 | `945490e47f9a4d5a234a7ee95b2db7a74bf88d32ced68d3d664dcb36a94bef19` |

## Routing-specific settings

| Setting | Advisor gate | Plan/execute | Stage router |
| --- | --- | --- | --- |
| Switchyard revision | `832374ac` | `08136d7d2ed4f862e5b3dd83488b3f120457d0ac` | `6a98efdbb63304ff1625f09a240a32381e396d28` |
| Configuration schema | 1 | 1 | 1 |
| Public route ID | `switchyard` | `switchyard` | `gpt-5.6-luna` |
| Luna role | Executor, reasoning effort `max` | Executor, reasoning effort `max` | Efficient tier |
| Sol role | Advisor, reasoning effort `max` | Planner, reasoning effort `max` | Capable tier |
| Concurrency | 10 tasks | 5, 5, and 8 tasks | 5 tasks |
| Independent runs | 3 | 3 | 1 |
| Reported score | Mean and sample standard deviation | Mean and sample standard deviation | One strict score |

The linked files contain the exact routing parameters, prompts, and model IDs. Their provider
blocks use OpenRouter so they can run outside the original evaluation environment. The qualification
runs used a different serving stack, so absolute results can differ even when the routing settings
are unchanged.

## Historical Plan/Execute configuration

The [archived TOML](https://github.com/NVIDIA-NeMo/Switchyard/blob/f979f6929f3504748d6792f5656426d2e20d623e/benchmark/routing-profiles/deepswe-v11-plan-execute-luna-sol.toml)
records the settings measured at revision
`08136d7d2ed4f862e5b3dd83488b3f120457d0ac`. The TOML file was published in a
later revision. It requires a version that still supports the removed
`plan_execute` route. These results belong to the standalone implementation;
the current [Plan/Execute configuration for Stage](../docs/routing_algorithms/stage_router_routing.md#plan-and-execute)
has no published benchmark results.
