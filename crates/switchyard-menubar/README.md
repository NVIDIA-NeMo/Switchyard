# switchyard-menubar

A macOS menu bar companion for a Switchyard server running in the background.

The server does the routing. This process reads what the server wrote and
edits the server config only when you ask it to, so quitting it never affects
traffic.

Click the glyph to see whether the server is answering, today's and this
week's requests and tokens, estimated savings, and which models served the
week. It can also restart the server, open either config file, and change a
route's algorithm and models.

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
config_file = "~/.switchyard/composite.toml"
launchd_label = "com.nvidia.switchyard.server"
baseline_model = "gpt-5.6-sol"
refresh_seconds = 30

[prices."gpt-5.6-sol"]
input_per_mtok = 1.25
cached_input_per_mtok = 0.125
output_per_mtok = 10.0
```

## Change a route's algorithm and models

Click the glyph, then **Change routing…**. The window reads the server config
named by `config_file`.

1. Pick a route. The window shows its algorithm and the model for each role.
2. Pick an algorithm: `passthrough`, `random`, `llm_classifier`, `composite`,
   `stage_router`, `advisor`, `plan_execute`, or `auto`. The window says what
   each one does and which roles it needs. For example, `composite` needs a
   judge, a capable model, and an efficient model.
3. For each role, pick an LLM client from the config, then a model. The model
   box lists the models from the client's `GET /models` endpoint. Type to
   filter the list, or type a model ID that is not listed.
4. Click **Apply**.

Apply checks the new config with
`switchyard-server --config <file> --dry-run`, using the `switchyard-server`
installed next to the menu bar app. If the check fails, the window shows the
server's error and the file does not change. If it passes, the app saves the
file, keeps the old one as `<file>.switchyard-backup.<timestamp>`, restarts
the server with `launchctl kickstart -k gui/<uid>/<launchd_label>`, and says
whether the server answers `/health`. If `config_file` is a symlink, the app
writes the file that the link points to, puts the backup next to that file,
and leaves the link in place.

The app changes as little of the file as it can:

- Comments, formatting, and every table the change does not touch stay as
  they were.
- A role keeps its target when that target already names the chosen model.
  If another target names that model on that client, the route uses it.
  Otherwise the app changes the route's own target in place, so settings such
  as `extra_body` and `omit_body_fields` stay.
- The app copies a target instead of changing it when another route uses it,
  or when an earlier role in the same Apply already took it. For example, if
  you move the efficient model up to Capable and pick a new efficient model,
  Capable takes the old efficient target, and Efficient gets a copy of that
  target with the new model.
- A changed or copied target also keeps its model-specific settings. For
  example, a target with `omit_body_fields = ["reasoning_effort"]` keeps that
  setting when it switches from a Claude model to a GPT model. Check these
  settings after you move a role to another model family.
- Switching algorithms keeps the route's `id`, `context_window`,
  `tool_calling`, `reasoning`, and `vision`, and keeps `subagents` when the new
  algorithm accepts it (`passthrough`, `stage_router`, and `composite`). It
  removes the old type's other settings and writes the settings the new type
  requires with the values from the routing docs, such as
  `confidence_threshold = 0.5`. Edit the file to tune them.
- Targets that no route uses any more stay in the file.

The window cannot show a custom-mode `llm_classifier` route or a route of
another type. Applying to such a route replaces its settings with the
algorithm you pick.

The check runs with the menu bar app's environment. If a client reads its
key from `api_key_env`, the menu bar's LaunchAgent needs that variable too.
Otherwise the check fails and the app saves nothing.

If a chosen model has no price in `menubar.toml`, the window says so.
Savings stay hidden until you add the price and restart the menu bar app.

### Model lists

The app saves every model list it fetches in `model-lists.json`, in the same
directory as the settings file it was started with. With the default settings
file, that is `~/.switchyard/model-lists.json`. The file holds the model IDs
and the time each list was fetched, keyed by the list's URL. It never holds a
key.

When a role needs a list, the window uses the list it already has, or else
the one in `model-lists.json`. Only when neither exists does the app send one
`GET /models` request. After a list is in `model-lists.json`, the app does not
fetch it again on its own, even after a restart. If a fetch fails and the
window has no list to show, the app tries again the next time you open the
window. Clients whose models share a URL share one list and one request.

If the app cannot write `model-lists.json`, the window still uses the fetched
list and shows "Could not save the list" with the error. Because the list is
not in the file, the app fetches it again after a restart.

Click **Refresh models** to fetch the lists that the window's roles use again,
one request per URL. The result area then shows each list's model count or
error. If a fetch fails, the window keeps the list it has and shows the error
under it.

When a role's model box is empty or holds a listed model ID, the note under
it shows the list's model count and age, such as "10 models, listed 2 minutes
ago. Type to filter." While you type, the note says how many models match,
such as "2 of 10 models match." When nothing matches, it says "No listed
model matches. Apply uses the ID as typed."

### Keys for model lists

The app never writes a key to a file. For a client with `api_key_env`, it
reads that variable from its own environment. A client with `forward_auth`
stores no key, and a LaunchAgent does not get the variables from your shell
profile. So when a list needs a key and none is available, the role's note
says so, and the window shows a key field under the roles. Paste the key and
click **Save key**. The app saves the key in your login Keychain as
"Switchyard model list", with the client's `base_url` as the account, and
fetches the list with it. Clients with the same `base_url` share the key.

If macOS cannot save the key, or cannot hand over a saved one, the window
shows the Keychain's error. You can always type a model ID that is not
listed.

The app lists models with the system `curl`. It writes the key to curl's
stdin, never to its command line, and runs curl with `-q`, so curl ignores
`~/.curlrc`. Without `-q`, a `verbose` line in that file would print the key
into the error text that the window shows.

The app uses the Keychain rather than your login shell's environment.
Reading that environment means starting your shell from the app, which runs
your whole shell profile and fails if the profile waits for input. The
Keychain needs no shell, and macOS asks before another app reads the key.
After you reinstall the menu bar app, macOS may ask once whether the new
build may read it.

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
