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

  Judge the strong phase against the trouble that caused the escalation,
  not against the size of the task. The difficult part is resolved when
  the failure that triggered the escalation no longer shows in the recent
  tool results: the command or test that kept failing now passes, the
  repeated diagnostic is gone, or the strong tier has landed the fix and
  verified it. Once that holds, ordinary implementation, reading, or
  clean-up that follows is routine work: release it. Retain while the
  strong tier is still diagnosing, still editing toward the fix, or has
  edited but not yet run the check that would confirm it; retain when a
  new failure has appeared in the strong tier's own turns; and retain
  when the last visible result is an error, an empty output treated as
  success, or a context compaction. A strong-tier turn that merely reads
  files or plans is not by itself evidence that the trouble is resolved.

The router, not the judge, applies confirmation counts and decides when
to change tiers. Judge only the phase named in the routing input.
