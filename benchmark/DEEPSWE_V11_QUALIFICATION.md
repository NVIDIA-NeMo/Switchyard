# DeepSWE v1.1 qualification settings

Routing configurations:

- [Advisor gate: Luna executor with Sol reviews](routing-profiles/deepswe-v11-advisor-gate-luna-sol.toml)
- [Stage router: Luna efficient tier with Sol capable tier](routing-profiles/deepswe-v11-stage-router-luna-sol.toml)

## Shared settings

| Setting | Value |
| --- | --- |
| Workload | DeepSWE v1.1, all 113 tasks, benchmark revision 1 |
| Agent | Codex CLI 0.154.0 |
| Harness | Harbor 0.13.2 |
| Protocol | OpenAI Responses |
| Network | Closed book with public network access denied |
| Agent budget | 3 hours |
| Command timeout | 10,800 seconds |
| Sandbox timeout | 8,400 seconds |
| Lifecycle timeout | 21,600 seconds |
| Infrastructure retries | 2 |
| Step and cost limits | Disabled |
| Strict scoring | Reward `1` divided by 113; failures and missing results score zero |
| Pricing snapshot | OpenRouter list prices from 2026-09-09 |
| Pricing SHA-256 | `945490e47f9a4d5a234a7ee95b2db7a74bf88d32ced68d3d664dcb36a94bef19` |

## Routing-specific settings

| Setting | Advisor gate | Stage router |
| --- | --- | --- |
| Switchyard revision | `832374ac` | `6a98efdbb63304ff1625f09a240a32381e396d28` |
| Configuration schema | 1 | 1 |
| Public route ID | `switchyard` | `gpt-5.6-luna` |
| Luna role | Executor, reasoning effort `max` | Efficient tier |
| Sol role | Advisor, reasoning effort `max` | Capable tier |
| Concurrency | 10 tasks | 5 tasks |
| Independent runs | 3 | 1 |
| Reported score | Mean and sample standard deviation | One strict score |

The linked files contain the exact routing thresholds, turn windows, review limits, handoff
settings, and model IDs. Their provider blocks use OpenRouter so they can run outside the original
evaluation environment. The qualification runs used a different serving stack, so absolute results
can differ even when the routing settings are unchanged.
