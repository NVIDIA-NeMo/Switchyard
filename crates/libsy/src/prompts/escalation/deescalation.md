# Routing phase

When a routing phase marker is present, routing is reversible and the
phase rules below replace the earlier one-way escalation rule.

The routing input begins with one of these router-generated markers:

- `EFFICIENT_EVALUATION`: Review the efficient-tier response. Return
  `escalate: true` when the trajectory needs the strong tier; otherwise
  return `escalate: false`.
- `STRONG_EVALUATION`: Review the strong-tier response. Return
  `escalate: true` when the remaining work still needs the strong tier.
  Return `escalate: false` only when the difficult part is resolved and
  the remaining work is routine enough for the efficient tier.

The router, not the judge, applies confirmation counts and decides when
to change tiers. Judge only the phase named in the routing input.
