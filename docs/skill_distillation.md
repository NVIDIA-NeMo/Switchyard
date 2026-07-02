# Skill Distillation

Skill distillation turns agent sessions into a reusable skill for the same
kind of work. The model is not retrained. Switchyard saves the session history,
uses it to update a `SKILL.md`, and makes the active skill available to later
agent launches for the same namespace.

The current release adds the saved configuration and the Rust contracts that
later implementations will share. It does not yet save session files, run
distillation, update skills, import external runs, validate results, or mount
skills into launched agents.

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

## Intended Workflow

Once launcher support lands, the flow should be automatic:

```text
configure a namespace
run an agent through switchyard launch
save the session under that namespace
distill when the session ends
create or update the namespace's active SKILL.md
load that active skill in the next launch for the same namespace
```

If no skill exists yet, the first distilled session creates one. Later sessions
update the existing skill and keep enough history to inspect or roll back the
change.

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

The config is intentionally small:

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

The automatic workflow still needs guardrails. Every skill update should record
which sessions supported it, keep the previous active skill available for
rollback, and produce review output. Later checks should flag answer leakage,
source IDs, URLs, benchmark shortcuts, and task-specific details before a skill
is trusted.

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

The ledger should track which saved sessions have already contributed to a
skill update. That lets distillation use new sessions by default instead of
depending on a long-lived lookback-count setting.

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
