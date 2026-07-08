# Skill Distillation

Skill distillation turns agent sessions into a reusable skill for the same
kind of work. The model is not retrained. Switchyard saves session history and
provides local storage contracts that an explicit workflow can use to produce,
validate, and activate a `SKILL.md` for the same namespace.

The launcher surface saves the namespace, defines the shared Rust contracts,
and automatically captures completed `switchyard launch` turns under the
current project. It still does not automatically run distillation, import
external runs, activate a candidate, or mount skills into launched agents.
Saved sessions and the local ledger are inputs for an explicitly orchestrated
distillation workflow.

The Python library also provides explicit building blocks for controlled local
experiments: it can normalize captured Switchyard sessions as immutable native
evidence, save and validate a skill candidate, activate it, and roll back to the
previous active bundle. These are library APIs, not a new launcher flag or an
automatic background workflow. See the container-free
[Nemotron Ultra LABBench2 TrialQA demo](../benchmark/TRIALQA_SKILL_DISTILLATION_DEMO.md)
for the end-to-end benchmark orchestration.

## Configure It

Choose a short namespace for the task or workflow you want the skill to learn:

```bash
switchyard configure --skill-distillation tooluniverse-trialqa
switchyard configure --show
```

The namespace is stored in the user config and can be updated without provider
credentials. To remove it:

```bash
switchyard configure --disable-skill-distillation
```

## Workflow and Current Boundaries

Session capture is automatic after configuration. The remaining stages require
an explicit caller today:

```text
configure a namespace
run an agent through switchyard launch
save the session under that namespace
  [automatic launcher capture ends here]
import or select completed evidence
create and validate an immutable skill candidate
activate or roll back the namespace's active bundle
mount that active skill into a later run
```

The store can create the first active bundle from a validated candidate and can
archive later active bundles for rollback. Candidate generation and launch-time
mounting remain responsibilities of the explicit orchestration layer.

Skill distillation is namespace-based. The namespace is saved user
configuration, not a per-request header. A future request may be recorded as
part of a saved session, but a single HTTP request should not change which skill
is being learned or used.

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

The config is intentionally small. There is no separate session-store knob:
the store is always project-local at
`.switchyard/skill-distillation/<namespace>/`.

| Field | Default | Meaning |
|---|---|---|
| `namespace` | unset | Name that groups saved sessions and generated skills for one workflow. |

Namespaces must be safe local path components: letters, numbers, dot,
underscore, and hyphen only. They cannot be `.` or `..`.

The top-level `skill_distillation` key is omitted when skill distillation is
not configured. `namespace` is the only supported key today. Extra manually
edited keys are rejected instead of being treated as inactive future options.

## Decisions

Generated output should stay portable and easy to review. Switchyard should
write `SKILL.md` and human-readable reports, not generated Python or other
executable files.

Skill generation should happen after sessions finish, not during normal model
request handling. Switchyard owns the saved sessions, local files, distillation
hooks, validation hooks, history, and launch-time skill loading. It should not
turn every request into a memory update.

Every candidate records which evidence supported it and must have passed
validation before activation. The store keeps the previous active bundle for
rollback. Content checks for answer leakage, source IDs, URLs, benchmark
shortcuts, and task-specific details belong to the validator used by the
orchestration layer.

## Store Layout

Session capture writes inspectable local files:

```text
<project>/.switchyard/skill-distillation/<namespace>/
  sessions/<session-id>/
    session.json
    turns.jsonl
    stats.json
  distillation-ledger.jsonl
  evidence/<evidence-id>/
    manifest.json
    evidence.json
    raw/
  evidence-ledger.jsonl
  candidates/<candidate-id>/
    manifest.json
    <skill-name>/SKILL.md
  reports/
  active/
    manifest.json
    <skill-name>/SKILL.md
  history/<archived-bundle>/
  history/activation-ledger.jsonl
```

`session.json` records the launch target, display model, strategy summary,
status, active skill path, and distillation handoff status. `turns.jsonl`
records normalized request and response turns, including messages, usage, and
routing metadata when available. `stats.json` records the final session stats.

The capture ledger tracks whether each saved session is pending future
distillation or was skipped because no completed turns were captured. Native
TrialQA imports use immutable evidence directories and a content-addressed
ledger. Candidate manifests hash every `SKILL.md`, name their source evidence,
and carry validation status. Activation revalidates the candidate and its local
evidence before publishing the bundle and recording the change.

## Rust Contracts

The `switchyard-skill-distillation` crate defines the source-neutral records
and extension points shared by future implementations and adapters. It does not
choose a provider, storage format, agent runtime, or model implementation.

The public contract includes:

- safe `SkillNamespace`, `TrajectoryId`, and `SkillVersionId` identifiers;
- versioned trajectories with task, execution, source, event, and outcome data;
- versioned skill candidates with required source-trajectory provenance and
  optional validation reports; and
- async `TrajectorySource`, `SkillDistiller`, `SkillValidator`, and `SkillStore`
  traits.

Provider-specific event data stays inside JSON payload and metadata fields.
The crate validates record invariants but does not redact source data, run a
model, write files, or activate a candidate by itself. Adapters and runtime
implementations own those operations.
