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
LaunchAgents, and writes a `sy` Codex profile. Use the CLI profile with
`codex -p sy`.

Your server config and menu bar settings are never overwritten once they
exist, so edits survive a reinstall. Logs are in `~/.switchyard/logs/`.

## Settings

`~/.switchyard/menubar.toml`. Any key may be omitted.

```toml
server_url = "http://127.0.0.1:4123"
routing_log = "~/.switchyard/routing.jsonl"
baseline_model = "gpt-5.6-sol"
refresh_seconds = 30

pricing_file = "~/.switchyard/prices.json"
```

The server's `--pricing-file` and the menubar's `pricing_file` use the same JSON file.
Key each entry by the model ID in the routing log. Rates are in USD per million tokens:

```json
{"gpt-5.6-sol":{"input":1.25,"cache_read":0.125,"cache_write":1.25,"output":10}}
```

These example rates preserve the installer's previous assumptions. Replace them with
what you pay. The [pi guide](../../docs/integrations/pi.md#estimate-the-cost) explains
cache-write rates and how to add other models. Rates load at startup; restart both
applications after editing them.

## How savings are computed

Actual spending is the sum of the costs saved in the routing log, including routing
calls. Changing prices does not change saved costs. The baseline is an estimate of
what the answer traffic would have cost using `baseline_model`; it excludes routing
calls. The server and menubar share the code for cost totals and token pricing, including
cache-read and cache-write rates.
The baseline assumes the same token and cache counts as the observed answer traffic.

The menubar shows estimated savings only when actual costs and the baseline estimate
are complete. A recorded `cost: null` keeps spending unknown. A missing baseline rate
or required one-hour cache-write rate keeps the baseline unknown. Missing or invalid answer
usage also keeps the baseline unknown. Token counts still
appear when savings cannot be calculated. The log can be read while the server is offline.

Codex's `codex-auto-review` model keeps the installer's baseline rates by default.
Update that entry when changing `baseline_model` if you want those reviews to have
the same estimated cost with and without Switchyard. ChatGPT subscription traffic
may have no per-token bill; token-rate estimates are comparisons at the supplied rates.

## Migrate older settings and logs

Inline `[prices]` settings and log records without a `cost` field require migration.
Stop the server and menubar before migrating. From the repository root, run:

```sh
cargo run -p switchyard-costs -- \
  ~/.switchyard/menubar.toml ~/.switchyard/routing.jsonl ~/.switchyard/migrated-costs
```

The destination directory must not exist. The command writes `menubar.toml`,
`prices.json`, and `routing.jsonl` there and leaves the originals untouched. It preserves
existing costs, including null, and adds estimates only to records without a cost field.
Missing usage or rates leave the reconstructed cost unknown. Malformed or unfinished
records stop migration; failed migration outputs are removed.

The command uses the rates in the old settings. It cannot recover historical provider
charges or historical prices. It maps `input_per_mtok` to `input` and `cache_write`,
`output_per_mtok` to `output`, and `cached_input_per_mtok` to `cache_read` (or the input
rate when omitted). This preserves the old cache-write assumption. Review those rates
for your providers; add `cache_write_1h` when needed. Reconstruction is always marked
`estimated`, even when it uses historical prices.

Review the generated files, then start the menubar with the generated settings path.
Those settings point at the generated log and pricing file. Update the server's
LaunchAgent to use those same paths for `--routing-log-file` and `--pricing-file` before
restarting it. Keep the originals as backups. Reinstalling preserves existing files;
it does not migrate them.

## Codex profiles

`codex --profile sy` reads `~/.codex/sy.config.toml`. The installer writes this
standalone profile and keeps your existing `~/.codex/config.toml` unchanged.
Reinstalling backs up the profile before replacing it if its contents change.

The profile sets `model`, `model_provider`, and the Switchyard provider details.
It leaves `approval_policy` and `sandbox_mode` to your existing Codex settings.

Codex.app cannot use the profile. The installer leaves its config unchanged.
For older installs that replaced `config.toml`, `make uninstall-macos` restores
`config.toml.direct` and keeps a copy of the active config before replacing it.
