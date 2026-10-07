# Pi virtual model using Switchyard

This example registers `switchyard/auto`. Switchyard chooses between two planning
models using a System One classifier. Pi sends the completion requests and keeps
the router's state on the current session branch.

The planning model stays selected until the first successful `edit` or `write`.
The next request switches to the cheap implementation model. Direct requests,
including compaction summaries, use the cheap model. Thinking levels pass through
to Pi, which clamps them to the selected model's supported levels.

## Local setup

Requires Node.js 22.19 or newer, the repository's Rust toolchain, and Pi with the
virtual-model API. Development checks use Pi 1.0.4.

From the repository root:

```bash
npm --prefix bindings/typescript ci
npm --prefix bindings/typescript run build
npm --prefix examples/pi ci
npm --prefix examples/pi run typecheck
npm --prefix examples/pi test
```

Policy tests use local mocks. The loading smoke test uses Pi's public resource
loader to check that the extension and its native dependency load successfully.
The binding tests cover native routing with a loopback classifier.

Before a live run:

1. Set `TYPESAFE_API_KEY` for the classifier in `switchyard.toml`.
2. Sign in to the completion provider through Pi's `/login`.
3. Confirm the three model IDs in `switchyard-router.ts` exist in your Pi catalog.
   Update `TARGETS`, the planning target IDs in TOML, and the virtual model limits
   together when changing models.

Then run:

```bash
pi -e ./examples/pi/switchyard-router.ts --model switchyard/auto
```

This command can make paid classifier and completion calls. The example uses
`switchyard.toml` beside the extension by default. Set `SWITCHYARD_CONFIG` to an
alternate file path when testing another configuration.

## Configuration and ownership

The route ID is `switchyard/planner`. Its completion target names must be
`complex` and `standard`. The extension maps each target to a Pi `(provider, id)`
pair and checks that the returned model ID matches. The implementation model is
selected directly by the extension.

The TOML's completion client uses a loopback placeholder. Pi owns the actual
completion connection and credentials. Switchyard uses only its configured
classifier connection during routing. Its classifier calls and costs are outside
Pi's provider-call accounting.

The sample decision judge compares the capable model's advantage over the
standard model. This uses Switchyard's existing relative-advantage policy. Review
the sample evidence and cutoff for your models and tasks.

Pi persists `{ phase, target }` through branch changes, resume, and compaction.
The extension returns new state only when the phase or selected target changes.
An eligible previous planning model is reused before invoking the classifier.
A failed `edit` or `write` keeps the planning model.

Classifier execution failures select the default `standard` planning model.
Cancellation, configuration errors, missing Pi models, and invalid target mappings
are surfaced to the caller. The sample sets `fail_open = false` so execution
failures reach the extension; any completed classifier decision follows the
configured Switchyard policy.

For the binding's API and configuration restrictions, see
[`bindings/typescript`](../../bindings/typescript/README.md).
