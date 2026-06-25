# Skill Distillation

Skill distillation turns selected past agent sessions into reusable guidance,
often a portable skill centered on `SKILL.md`. The model is not retrained.
Switchyard's role is the trace substrate: name the workflow, collect structured
session traces when that support lands, and keep enough provenance for a
separate distillation tool or harness-specific adapter to consume later.

The current implementation defines the product shape and saved configuration
only. It does not implement session capture, drafting, adoption, rollback,
external import, validation runs, or agent activation.

## V1 Workflow

The intended v1 flow is:

```text
run agents -> save session traces -> distill outside the proxy -> review/test -> adopt in the target harness
```

The user-facing command shape is:

```bash
switchyard configure --skill-distillation tooluniverse-trialqa
switchyard launch codex
switchyard configure --show
```

For now, `configure` stores only the namespace so future trace capture has a
stable grouping key. Distillation, review, adoption, and skill activation remain
future work and are expected to be harness-specific.

## Config

Skill distillation config is stored in `~/.config/switchyard/config.json` under
the top-level `skill_distillation` key:

```json
{
  "skill_distillation": {
    "namespace": "tooluniverse-trialqa"
  }
}
```

The config is intentionally small:

| Field | Default | Meaning |
|---|---|---|
| `namespace` | unset | Required when configuring the future trace collection namespace. |

Namespaces must be safe local path components: letters, numbers, dot,
underscore, and hyphen only. They cannot be `.` or `..`.

The top-level `skill_distillation` key is omitted when the workflow is not
configured. Use `switchyard configure --disable-skill-distillation` to remove
it without touching provider credentials. `namespace` is the only supported key
today; any extra manually edited keys are rejected instead of being treated as
dormant product behavior.

## Decisions

Generated outputs should stay portable and reviewable. A future candidate may
contain `SKILL.md` and human-readable support reports, but generated Python or
other executable artifacts are out of scope for the initial workflow.

Switchyard should not become a request-path memory system, and skill generation
should not run inside normal request handling. Distillation and activation are
closer to the target agent than to the proxy, so future implementations should
use harness-specific adapters rather than assuming one generated artifact can
be mounted everywhere.

The namespace is saved user configuration, not a per-request header. Future
request metadata may be captured as trace evidence, but individual requests
should not reconfigure skill distillation behavior.

Cheating prevention is explicit rather than automatic adoption. Drafts stay
staged by default, humans adopt them, every final rule should link back to
supporting sessions, and later review checks should flag answer leakage, source
IDs, URLs, benchmark shortcuts, and task-specific details before a skill is used.

## Planned Store Layout

Future work should use inspectable local files:

```text
<project>/.switchyard/skill-distillation/<namespace>/
  sessions/<session-id>/
    session.json
    turns.jsonl
    stats.json
  distillation-ledger.jsonl
  candidates/<version-id>/
    skill/SKILL.md
    report.md
  active/SKILL.md
  history.jsonl
```

The ledger should track which captured sessions have already been consumed by a
candidate. That lets future distillation default to "new traces not yet used"
instead of depending on a persistent lookback-count knob.
