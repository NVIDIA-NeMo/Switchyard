# switchyard-menubar

The macOS app shows usage, installs coding-tool settings, and opens local
worktree sessions. Its menu bar shows server status and estimated savings.

The server does the routing. This process reads what the server wrote and
edits the server config only when you ask it to, so quitting it never affects
traffic.

Click the glyph to see whether the server is answering, today's and this
week's requests and tokens, estimated savings, and which models served the
week. The glyph turns faint when the server does not answer. The menu can also
restart the server, open either config file, and open the routes window, where
you change a route's algorithm and models. When a menu item fails, a dialog
shows the error.

## Install

From the repository root:

```sh
make install-macos          # or install-macos-dry-run to see the steps first
make uninstall-macos
```

`make install-macos` builds `switchyard-server` and `switchyard-menubar` into
`~/.switchyard/bin` and installs `~/Applications/Switchyard.app`. It writes a
server config and menu bar settings, loads two LaunchAgents, and writes a `sy`
Codex profile. Use the CLI profile with `codex -p sy`. Open Switchyard from
Finder or Spotlight to see Usage, or choose **Install…** to set up a coding tool.

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

The app checks this file when the menu refreshes and after each menu click. A
changed file takes effect without a restart. When `config_file` changes while
the routes window is open and no Apply is running, the window loads the new
file and drops changes that you have not applied.

## Edit a route's algorithm and models

Click the glyph, then **Edit routes…**. The window reads the server config
named by `config_file`.

1. Pick a route in the list at the left. The routes that choose between models
   come first, and the routes that send every request to one model come
   after them. A dot after a route's name means that you changed the route
   and have not applied the change.
2. Pick an algorithm. The window names each one in plain words, such as
   "Judge plus stage router", and shows its `type` in parentheses. Under the
   name, it says what the algorithm does. Each role has its own line of text
   that says what the role does. For example, `composite` needs a judge, a
   capable model, and an efficient model.
3. For each role, pick an endpoint, which is an LLM client from the config,
   then a model. The endpoint box shows the client's name and the request
   format it speaks. Hover over it to see its URL and whose key it uses. The
   model box lists the models from the endpoint's `GET /models` URL. Type to
   search the list, or type a model ID that is not listed.
4. Click **Apply**, or press Return. Click **Revert** to drop your changes to
   the open route. Escape or Cmd-W closes the window.

The window keeps what you picked on a route while you look at other routes.
Apply writes only the open route. If you close the window and open it again,
it drops changes that you did not apply, unless an Apply is still running.
Choosing **Edit routes…** while the window is open brings it to the front and
keeps your changes.

When a route has more roles than the screen can show, such as a `random`
route with many models, the roles scroll, and the buttons stay on the screen.

### Endpoints without a model list

The ChatGPT Codex endpoint, `chatgpt.com`, has no model list that an API key
can read, and the app never holds your ChatGPT login. So the window does not
ask for a key for it. The model box for that endpoint lists the models that
your config already names on it, together with the models in Codex's own
saved list, `$CODEX_HOME/models_cache.json` or else
`~/.codex/models_cache.json`. The menu bar app reads `CODEX_HOME` from its
own LaunchAgent environment, not from your shell. While the model box is
empty or holds a listed ID, the note under it says where the list comes from
and how old Codex's list is. You can type any other model ID.

### Blocks that another tool writes

Some tools write part of the server config between marker lines that look like
`# >>> name >>>` and `# <<< name <<<`. A route inside such a block shows a
warning, because that tool overwrites changes made here the next time it
runs. New targets go after the last target outside every block, so the next
rewrite of a block cannot delete them. When every target is inside a block, a
new target goes before the first table of the file. When a route outside a
block ends up using a target inside one, the result says so.

### Apply

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

While Apply runs, **Quit** shows a message and does not quit. Quitting between
the save and the restart could leave a saved config that the server has not
loaded.

The result area lists what happened, most important first: the saved file
and its backup, the restart and whether the server answers `/health` within
10 seconds, notes about the change, and how many routes passed the check. If
the server does not answer, the result says how to put the backup back.

When the change sends each caller's own login (`forward_auth = true`) to a
different host than before, the result says so. A caller that has no login for
the new host gets HTTP 401. For example, the Codex app's ChatGPT login does not
work on a gateway such as `gateway.example.com`.

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
model. Savings stay hidden until you add the price to `menubar.toml`.

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
it shows the list's model count, its host, and its age, such as "249 models on
gateway.example.com, fetched 6 days ago. Type to search." While you
type, the note says how many models match, such as "2 of 10 models match."
When nothing matches, it says "No listed model matches. Apply uses the ID as
typed."

For an `anthropic_messages` client, the request adds `limit=1000`, the most
that Anthropic returns in one page. Without `limit`, Anthropic returns 20
models.

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
and click **Save key**, or press Return in the field. The app first lists
the models with the key. If the models endpoint rejects it, the app does not
save it. Otherwise the app saves the key in your login Keychain as
"Switchyard model list", with the client's `base_url` as the account.
Clients with the same `base_url` share the key. The app uses a saved key
only to list models at that `base_url`. It refuses a key with a line break,
because a line break would add a request header.

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
a bad day shows a negative number. Today's dollar figures and this week's
each stay hidden until every model seen in that period, and the baseline
model, has a price, so a partial table cannot mislead. While this week's
figures are hidden, the menu names the models that have no price.

Codex's "Approve for me" reviews count too. The server config that the
installer writes sends them to `codex-auto-review` on the ChatGPT backend. With
a ChatGPT login, Codex sends these reviews to `codex-auto-review` even without
Switchyard, so the installer prices that model at the `baseline_model` rates,
including the cached input rate.
Each review then adds the same amount to the actual cost and to the baseline,
so reviews do not change the dollar amount saved. They do lower the percentage
a little, because the baseline grows. If you change `baseline_model`, update
this price to match.

The menu bar needs a price for the model ID that the routing log records for
reviews, which is the reviewer target's `id`. Without that price, the menu bar
hides the Saved row for every period that includes a review. The installer does
not overwrite an existing `composite.toml` or `menubar.toml`, so an existing
install needs the route and the price added by hand.

If a reviewer uses another model ID, add a price for that ID. Any period with
an unpriced review hides its savings and shows the missing-price hint.

These are list-price estimates. On a ChatGPT login there is no per-token bill,
so read them as "what this traffic would have cost at API rates".

## Codex profiles

`codex --profile sy` reads `~/.codex/sy.config.toml`. The installer writes this
standalone profile and keeps your existing `~/.codex/config.toml` unchanged.
Reinstalling backs up the profile before replacing it if its contents change.

The profile sets `model`, `model_provider`, and the Switchyard provider details.
It leaves `approval_policy` and `sandbox_mode` to your existing Codex settings.

Codex.app cannot use the profile. `make install-macos` leaves its
`config.toml` unchanged. To change the defaults used by Codex.app and Codex CLI,
choose **Codex app and CLI defaults** in the app’s **Install…** view.
For older installs that replaced `config.toml`, `make uninstall-macos` restores
`config.toml.direct` and keeps a copy of the active config before replacing it.

## Update the app from source

**Update from source…** opens Terminal and runs `scripts/macos/install.sh` from
the checkout used for the current install. Keep that checkout and a Rust
toolchain available. Update the checkout first to choose the code you want to
build. Routing settings and prices stay saved. The installer signs the app ad
hoc for local use; the app has no Developer ID signature or notarization.

## Install coding-tool settings

**Install…** shows where Codex, Claude Code, and Pi are installed, their user
settings, and their configured model. Pick a public route on the local HTTP
server and click **Install / refresh**. Codex CLI gets `sy.config.toml` and uses
`codex -p sy`. The Codex app option changes the defaults in `config.toml` for
both the app and CLI. Claude Code gets `settings.json` environment settings. Pi gets a custom provider in `models.json` and defaults in
`settings.json`. Unrelated settings remain. The original files are backed up
beside each file as `<filename>.switchyard-original`. If a file did not exist,
`<filename>.switchyard-original-missing` records its absence. **Restore original**
keeps a copy of the current file, then restores the backup or removes the file
that was originally absent.

The app checks all selected settings files before replacing any of them. Each
replacement is atomic. If a later replacement fails, the app tries to restore
earlier files. A crash can leave Pi’s two files out of sync, and rollback can
fail. Another process running as the same user can also change a file after the
app checks it. The original backups remain available for manual recovery.

Restart the coding tool after changing its defaults. Project settings, shell
variables, and command-line options can override them.

Subscription routes forward the caller's saved login to its own provider.
Choose Codex for a ChatGPT route or Claude Code for an Anthropic route. Pi's
custom provider uses routes whose API credentials the server owns. Configure
those credentials on the server before installing its route. The app never
reads or copies OAuth tokens, switches accounts in running sessions, or combines
subscription quotas. Claude settings with an explicit API key or bearer token
must be cleared before installing a subscription route. Shell overrides still
need to be checked with Claude's `/status`.

## View session usage

Usage lists the actual upstream model, public route, timestamp, input, cached,
and output tokens for each completed model call. Select a recorded session ID
to filter the view. A recorded turn ID groups related calls; a turn may contain
several model calls. Older records and tools that send no IDs say **Not recorded**.
The viewer reads the last 8 MiB of the log plus one byte to check whether the
first record is complete. It keeps at most 5,000 complete records and excludes
partial records. It labels a limited view and counts unreadable records.
Classifier calls appear as routing overhead. It displays neither prompts nor responses.
Observed tokens do not report provider quota, reset windows, or authoritative
billing. Refresh reads the log again.

## Start sessions with separate logins and worktrees

Codex CLI and Claude Code support named accounts. Enter a name and click
**Add account…** to open the tool’s own login command in Terminal. Complete the
login, click **Refresh**, and select the account. Switchyard keeps these account
directories under `~/.switchyard/accounts`; the coding tool owns their login
files. The selected account determines where **Install / refresh** saves settings
and which login a new session uses. Existing sessions keep their login.

To run several tasks with one coding tool, enter a Git project path under
Install and click **New worktree session**. Each session starts from that
checkout's HEAD on a new branch in `~/.switchyard/worktrees` and opens in Terminal.
Uncommitted files stay in the original checkout. Each launched session has its
own model settings; launching another session does not change user defaults.
Codex and Claude launches send a session ID for usage correlation. Pi stores
its session under the private agent directory beside the worktree; its requests
need an integration that sends session and turn IDs to appear as a correlated
session. Otherwise their usage is still listed under **All sessions**.
Use `git worktree list` and `git worktree remove` to manage the checkouts after
saving or discarding changes. Successful launches keep their Git branches and
the private settings directories beside the worktrees until you remove them. Named
account directories also remain until you remove them. If launch setup fails,
the app tries to remove the worktree, branch, and private settings it created;
cleanup can fail.

The app does not manage task dependencies, resume history, remote workspaces,
or review comments.
