# switchyard-menubar

A macOS menu bar companion for a Switchyard server running in the background.

The server does the routing. This process only reads what the server already
wrote, so quitting it never affects traffic.

Click the glyph to see whether the server is answering, today's and this
week's requests and tokens, estimated savings, and which models served the
week. It can also restart the server and open either config file.

## Install

From the repository root:

```sh
make install-macos          # or install-macos-dry-run to see the steps first
make uninstall-macos
```

That builds `switchyard-server` and `switchyard-menubar` into
`~/.switchyard/bin`, writes a server config and menu bar settings, loads two
LaunchAgents, writes a `sy` Codex profile, and routes Codex.app through
Switchyard. Use the CLI profile with `codex -p sy`.

Your server config and menu bar settings are never overwritten once they
exist, so edits survive a reinstall. Logs are in `~/.switchyard/logs/`.

## Settings

`~/.switchyard/menubar.toml`. Any key may be omitted.

```toml
server_url = "http://127.0.0.1:4123"
routing_log = "~/.switchyard/routing.jsonl"
baseline_model = "gpt-5.6-sol"
refresh_seconds = 30

[prices."gpt-5.6-sol"]
input_per_mtok = 1.25
cached_input_per_mtok = 0.125
output_per_mtok = 10.0
```

## How savings are computed

The server writes one JSONL record per call, with the model that answered and
its token counts.

- **Actual** is every call priced at the model that served it, including
  Switchyard's own classifier calls.
- **Baseline** is the caller-facing calls only, priced as if each had used
  `baseline_model`. Classifier calls have no baseline counterpart, because
  without Switchyard they would not happen.

Savings are the difference, so routing overhead counts against the figure and
a bad day shows a negative number. Dollar figures stay hidden until every
model seen has a price, so a partial table cannot mislead.

These are list-price estimates. On a ChatGPT login there is no per-token bill,
so read them as "what this traffic would have cost at API rates".

## Codex profiles

`codex --profile sy` reads `~/.codex/sy.config.toml`. A `[profiles.sy]` table
inside `config.toml` is legacy config the CLI now refuses to start with, so the
installer removes one if an older version left it there.

The profile sets only `model` and `model_provider`. It deliberately leaves
`approval_policy` and `sandbox_mode` alone, because routing should not quietly
change how Codex asks before it acts.

**Codex.app cannot use the profile.** The installer builds
`~/.codex/config.sy.toml` from your original config, backs up that original to
`~/.codex/config.toml.direct`, and installs the routed config as
`~/.codex/config.toml`. `make uninstall-macos` restores the original and keeps
a copy of the active config before replacing it.
