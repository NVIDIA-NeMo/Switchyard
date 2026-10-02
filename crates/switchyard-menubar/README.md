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
4. Click **Apply**, or press Return. Escape or Cmd-W closes the window.

When a route has more roles than the screen can show, such as a `random`
route with many models, the roles scroll, and the buttons stay on the screen.

Apply checks the new config with
`switchyard-server --config <file> --dry-run`, using the `switchyard-server`
installed next to the menu bar app. If the check fails, the window shows the
server's error and the file does not change. If it passes, the app saves the
file, keeps the old one as `<file>.switchyard-backup.<timestamp>`, and
restarts the server with `launchctl kickstart -k gui/<uid>/<launchd_label>`.
A second backup in the same second gets `-2` added to its name, so a backup
never replaces another one. If another program changes the file while Apply
checks it and writes the backup, Apply saves nothing, removes that backup, and
asks you to click Apply again. If `config_file` is a symlink, the app writes
the file that the link points to, puts the backup next to that file, and
leaves the link in place.

The result area lists what happened, most important first: the saved file
and its backup, the restart and whether the server answers `/health` within
10 seconds, notes about the change, and how many routes passed the check. If
the server does not answer, the result says how to put the backup back.

The app changes as little of the file as it can:

- Comments, formatting, and every table the change does not touch stay as
  they were.
- A role keeps its target when that target already names the chosen model.
- If another target names that model on that client, the route uses it when
  the route would get the same `system_prompt`, `reasoning_effort`,
  `extra_body`, and `omit_body_fields` from it. The route also uses it when
  `reasoning_effort`, `extra_body`, or `omit_body_fields` differ, because the
  server rejects two targets that name one model on one client with
  different values for those. When the role then shares the target with
  another route, the result says so.
- Otherwise the app changes the route's own target in place, so settings such
  as `extra_body` and `omit_body_fields` stay.
- The app copies a target instead of changing it when another route uses it,
  or when an earlier role in the same Apply already took it. For example, if
  you move the efficient model up to Capable and pick a new efficient model,
  Capable takes the old efficient target, and Efficient gets a copy of that
  target with the new model.
- A changed or copied target also keeps its model-specific settings. For
  example, a target with `omit_body_fields = ["reasoning_effort"]` keeps that
  setting when it switches from a Claude model to a GPT model. `--dry-run`
  does not catch a setting that the new model rejects. So when a target moves
  to another model family, such as from `gpt-…` to `claude-…`, or to a client
  with another `format`, the result lists the `omit_body_fields`,
  `reasoning_effort`, and `extra_body` that the target kept, or says that it
  has none. The app does not change them; edit the file if the new model
  needs other values.
- Switching algorithms keeps the route's `id`, `context_window`,
  `tool_calling`, `reasoning`, and `vision`, and keeps `subagents` when the new
  algorithm accepts it (`passthrough`, `stage_router`, and `composite`). It
  removes the old type's other settings and writes the settings the new type
  requires with the values from the routing docs, such as
  `confidence_threshold = 0.5`. The result lists the settings it removed, and
  the backup still has them. Edit the file to tune them.
- Targets that no route uses any more stay in the file.

The window cannot show the settings of a custom-mode `llm_classifier` route
or of a route whose `type` is not in the Algorithm list. Applying to such a
route replaces its settings with the algorithm and models you pick.

The check runs with the menu bar app's environment, not your shell's. If a
client reads its key from `api_key_env`, the menu bar app's LaunchAgent needs
that variable too. Otherwise the check fails and the app saves nothing. The
window says so under each role that uses such a client, and again in the
result when the check fails. To add the variable, put it under
`EnvironmentVariables` in
`~/Library/LaunchAgents/com.nvidia.switchyard.menubar.plist`, then load the
agent again:

```sh
launchctl bootout gui/$UID/com.nvidia.switchyard.menubar
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.nvidia.switchyard.menubar.plist
```

That puts the key in the plist file as plain text, and `make install-macos`
writes the plist again without it. A client with `forward_auth = true` needs
no key in the config, because the server sends each caller's own key.

Apply does not add prices, because the app has no price source besides
`menubar.toml`. If a chosen model has no price there, the result names the
model. Savings stay hidden until you add the price and restart the menu bar
app.

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

Click **Refresh models** to fetch every list that the window's roles use,
even a cached one. The app sends one request per URL, and fetches the URLs at
the same time, so a slow URL does not hold back the others. Each role's note
changes as its list arrives, and the result area then shows each list's
model count or error. If a fetch fails, the window keeps the list it has and
shows the error under it.

When a role's model box is empty or holds a listed model ID, the note under
it shows the list's model count and age, such as "10 models, fetched 2
minutes ago. Type to filter." While you type, the note says how many models
match, such as "2 of 10 models match." When nothing matches, it says "No
listed model matches. Apply uses the ID as typed."

### Keys for model lists

The app never writes a key to a file. To list a client's models, it uses the
first key it finds: the key you just typed, then the variable named by the
client's `api_key_env` in the menu bar app's own environment, then the login
Keychain item for the client's `base_url`. A LaunchAgent does not load your
shell profile, so a key you export in `~/.zshrc` is not in the app's
environment. A `forward_auth` client has no key in the config, because the
server sends each caller's own key upstream.

When the app finds no key, the role's note says so and a key field appears
under the roles. Paste the API key for the `base_url` that the field names,
and click **Save key**. The app first lists the models with the key. If the
models endpoint rejects it, the app does not save it. Otherwise the app saves
the key in your login Keychain as "Switchyard model list", with the client's
`base_url` as the account. Clients with the same `base_url` share the key.
The app uses a saved key only to list models at that `base_url`. It refuses
a key with a line break, because a line break would add a request header.

If macOS cannot save the key, or cannot read a saved one, the window shows
the Keychain's error. You can always type a model ID that is not listed.

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
