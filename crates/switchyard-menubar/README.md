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

Codex's "Approve for me" reviews count too. The server config that the
installer writes sends them to `codex-auto-review` on the ChatGPT backend. With
a ChatGPT login, Codex sends these reviews to `codex-auto-review` even without
Switchyard, so the installer prices that model at the `baseline_model` rates.
Each review then adds the same amount to the actual cost and to the baseline,
so reviews do not change the dollar amount saved. They do lower the percentage
a little, because the baseline grows. If you change `baseline_model`, update
this price to match.

The menu bar needs a price for the model ID that the routing log records for
reviews, which is the reviewer target's `id`. Without that price, the menu bar
hides the Saved row for every period that includes a review. The installer does
not overwrite an existing `composite.toml` or `menubar.toml`, so an existing
install needs the route and the price added by hand.

Codex's "Approve for me" reviews need a price too. Use the reviewer target's
`id`, which is the model ID recorded in the routing log. The default reviewer
route uses `codex-auto-review`. With a ChatGPT login, these reviews also happen
without Switchyard. Add that ID to the price table at the `baseline_model`
rates, including the cached input rate. Each review then adds the same cost to
actual and baseline. Dollars saved stay the same; the percentage drops as the
baseline grows. Update this price when you change `baseline_model`.

If a reviewer uses another model ID, add a price for that ID. Any period with
an unpriced review hides its savings and shows the missing-price hint. Existing
`menubar.toml` files need these prices added by hand.

These are list-price estimates. On a ChatGPT login there is no per-token bill,
so read them as "what this traffic would have cost at API rates".

## Codex profiles

`codex --profile sy` reads `~/.codex/sy.config.toml`. The installer writes this
standalone profile and keeps your existing `~/.codex/config.toml` unchanged.
Reinstalling backs up the profile before replacing it if its contents change.

The profile sets `model`, `model_provider`, and the Switchyard provider details.
It leaves `approval_policy` and `sandbox_mode` to your existing Codex settings.

Codex.app cannot use the profile. The installer leaves its config unchanged.
For older installs that replaced `config.toml`, `make uninstall-macos` restores
`config.toml.direct` and keeps a copy of the active config before replacing it.
