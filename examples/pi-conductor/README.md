# Conductor for Pi

Capable reasoning, efficient execution.

Conductor makes the main Pi agent a read-only supervisor. It delegates edits and tests to one persistent worker in the same process and workspace. Switchyard routes supervisor requests to the capable model and child requests to the efficient model.

The supervisor can explore, delegate, inspect evidence, and redirect the same worker. The extension returns a checkpoint after the worker finishes a milestone or completes 40 responses. A checkpoint waits for the current tools to finish. It does not cancel an in-flight edit or test.

Each checkpoint includes:

- The cumulative Git diff against the original task base, including new files.
- Actual tool arguments and results, including command failures.
- The worker's summary, kept separate from the captured evidence.

Encrypted reasoning items pass unchanged between roles through the Responses API. Recipient reminders keep the supervisor and worker roles distinct. This requires compatible models and endpoints. It is not a claim that reasoning can be replayed across arbitrary providers.

## Run

Requires Node.js 22.19 or later, Pi 1.0.4, Git, and a Switchyard server with subagent routing.

```sh
cd examples/pi-conductor
npm ci
```

Set your upstream credential in the environment. Review `switchyard.toml` and adjust the endpoint and model IDs to ones available to you. Start Switchyard in another terminal:

```sh
switchyard-server --config /absolute/path/to/switchyard.toml --host 127.0.0.1 --port 4000
```

From a clean Git worktree, start Pi with the extension:

```sh
CONDUCTOR_BASE_URL=http://127.0.0.1:4000/v1 pi -e /absolute/path/to/pi-conductor/index.ts
```

Then enter `/conductor <your coding task>`. The main agent uses `conductor_delegate` for edits and tests. `/conductor-stop` stops the child and restores the previous model and tools. Evidence is written outside the worktree. Pi prints its location.

The worker runs with Pi's local permissions. Use a disposable worktree and an appropriate sandbox for untrusted tasks. This extension is not an operating-system sandbox or a network policy. Restart recovery is not implemented. Use a fresh Pi session for a separate task.

## Check

```sh
npm run check
npm test
npm run smoke
npm run smoke -- --cli
npm run smoke -- --worker-error
```

The scripted smoke uses real Pi sessions and a real Switchyard server with a local scripted upstream. It checks two delegations, worker continuity, safe checkpoints, evolving patches, model routing, reasoning replay, and per-response usage. It does not measure model quality.

For a small live smoke, use a compatible Responses endpoint:

```sh
SMOKE_UPSTREAM=https://your-endpoint.example/v1 \
SMOKE_API_KEY="$YOUR_API_KEY" \
SMOKE_CAPABLE_MODEL=your-capable-model \
SMOKE_EFFICIENT_MODEL=your-efficient-model \
npm run smoke -- --live
```

Usage artifacts contain tokens, not qualified dollar costs. The model registration has placeholder prices for Pi display. Do not interpret those as free inference. Missing terminal usage is flagged in `summary.json`. Transport retries still need upstream reconciliation before using this as a billing ledger.

The live smoke passed with Pi 1.0.4, GPT-6 Sol and GPT-6 Luna at max reasoning. It completed two delegations with 15 recorded responses, preserved reasoning in both directions, and passed independent checks of the edited code. That run used a local account-auth adapter for an existing ChatGPT connection. The adapter is not part of this package. The example configuration uses a standard API credential instead.

This package is a working integration candidate, not published on npm. The Pi harness needs its own benchmark qualification before using results from another harness as evidence for it.

## Implementation

- `index.ts` loads the extension.
- `src/conductor.ts` owns the delegation tool, child session, checkpoints, and cancellation.
- `src/evidence.ts` captures patches using a temporary Git index, without changing the user's index.
- `src/handoff.ts` preserves opaque reasoning and adds recipient role reminders.
- `switchyard.toml` uses existing passthrough and subagent routing. No new core routing algorithm is required.
