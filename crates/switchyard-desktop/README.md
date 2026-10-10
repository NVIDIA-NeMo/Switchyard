# Switchyard desktop and terminal apps

The macOS and Windows desktop apps and terminal app show usage, edit routes, install
coding-tool settings, and open sessions in separate Git worktrees. The server
handles routing and continues when the app quits. The desktop app uses Tauri 2;
the terminal app uses Ratatui.

## Install and update

Install the app from the repository root with the repository's pinned Rust
toolchain (currently Rust 1.99). Install Rust through rustup and keep Cargo on
`PATH`. Native dependencies also need a C compiler and linker: Xcode Command
Line Tools on macOS, or GCC/Clang and the system development libraries on Linux.
Cargo downloads dependencies on the first build. The default Codex profile
requires Codex CLI 0.134.0 or newer and a ChatGPT login. Sign in with Codex
before using it.

```sh
cargo desktop install
```

Run the same command to replace an existing installation. The command builds
the current checkout; update the checkout first to install newer code.

On macOS, the Rust installer builds the desktop app and server, installs
`~/Applications/Switchyard.app`, and loads two per-user LaunchAgents. It signs
the app ad hoc for local use. The app has no Developer ID signature or
notarization. Reinstalling keeps existing server settings, prices, accounts, and
log history. The installer moves older `menubar.toml` settings to `desktop.toml`
when the new file is absent and replaces the old app LaunchAgent.

The installer builds the binaries and validates the server config before
stopping services. On macOS, it also signs the replacement app before stopping
services. Each file replacement is atomic, but the complete installation is not.
If installation fails after services stop, fix the reported error and run `cargo
desktop install` again. Install and uninstall share a lock for your home
directory. If another installation holds it, wait for that command to finish and
retry.

By default, the source installer writes `~/.codex/sy.config.toml` for
`codex -p sy` and keeps `~/.codex/config.toml` unchanged. Set `SY_PROFILE` to
choose another profile name. The installer writes `$SY_PROFILE.config.toml` in
the Codex config directory. For example:

```sh
export SY_PROFILE=composite-gpt-6-sol-gpt-6-luna
cargo desktop install
codex -p "$SY_PROFILE"
```

Remove the app and recorded profile with `cargo desktop uninstall`. Uninstall
uses the saved installation location and profile. Conflicting shell overrides
require an explicit `uninstall --settings FILE` selection. Uninstall keeps an
exact recovery copy of the selected profile before removal. Restore coding-tool
routing from **Connect tools** first; custom and account settings remain and
may still point at the removed server. Preview removal of a selected installation
without changing files or services:

```sh
cargo desktop uninstall --settings FILE --dry-run
```

Set `SY_MODEL` to choose an existing public route ID. New installs use
`composite-gpt-6-sol-gpt-6-luna`. Existing server configs stay saved, so set
`SY_MODEL` to one of their routes when installing with a custom config. The
profile leaves `approval_policy` and `sandbox_mode` to your existing Codex
settings. Reinstalling backs up the profile before updating its managed keys. It
preserves unrelated profile settings. Updates from Settings preserve the entire
profile. `SY_PROFILE` must start with an ASCII letter or digit, use only ASCII
letters, digits, underscores, or hyphens, and contain at most 128 characters.
For example, the installer rejects `SY_PROFILE=team.dev` before creating app or
config files.

Open the app from Finder or Spotlight. Closing the window removes the running
app from the Dock and keeps the tray menu. Opening the window from the tray
restores the Dock icon. A shortcut you pinned to the Dock stays there.
The tray includes **Connect tools…**, the three models with the most recorded
tokens, **View model usage…**, and **Settings…**. Model totals cover the
retained log and include classifier calls. A final row combines any remaining
models. Clicking a model row opens Overview filtered to that model. Choose
**Open Switchyard** to return or **Quit Switchyard** to exit. Quit waits for a
queued or running operation to finish. The server continues running after the
app quits.

**Settings → Review update…** shows the checkout and planned changes. **Build
and reinstall** starts the Rust installer from the checkout used for this
installation. It builds that checkout's current commit and preserves the current
Codex profile. Review the checkout, local changes, and planned actions before
starting. Settings shows the update stage and outcome. The updater keeps running
when installation stops the desktop app. Only one update runs at a time.
Progress and errors go to `~/.switchyard/logs/update.log` (under `SY_HOME` when
customized). Keep the checkout and a Rust toolchain available. Update the
checkout first to choose the code you want to build.

Preview installation with `cargo desktop install --dry-run`. On Linux, the same
command installs the terminal app and a systemd user service for the server.
Linux installation requires an active systemd user session. Containers,
SSH sessions without a user service manager, and distributions without systemd need
separate service setup. The repository targets x86-64-v3 CPUs on Linux x86_64
and Neoverse N1 CPUs on Linux aarch64; older CPUs may not run the binaries.
The desktop GUI runs on macOS and Windows. Windows requires the MSVC build tools, WebView2, and a Task Scheduler user session.
On Linux, the terminal app provides usage views and config editing. Login,
session launch, and open-file actions require macOS or Windows. Run coding-tool
login commands and open files yourself on Linux. Model discovery needs `curl`;
session creation on macOS and Windows also needs Git and the selected coding tool.
With a custom Linux `SY_HOME`, pass the settings file explicitly:

```sh
"$SY_HOME/bin/switchyard-desktop" "$SY_HOME/desktop.toml" --tui
```

Run the terminal app with:

```sh
~/.switchyard/bin/switchyard-desktop --tui
# This command runs the terminal app from a checkout.
cargo run -p switchyard-desktop -- --tui
```

Tab changes sections. The footer lists navigation keys. In Routes, `/` searches
and the arrow keys select matching routes; `d` deletes a route only after you
type its name. In Coding tools, `p` previews a diff, `x` selects a custom file,
and `i` shows the preview before a second `i` installs the route. Restore asks
you to type `RESTORE` before changing coding-tool settings. Changing the tool,
account, or route clears the preview. In Routes, press `e` to edit the algorithm
and model roles. Tab changes form fields, Ctrl-U clears a field, arrow keys
choose algorithms and endpoints, Enter reviews the changes, and Escape cancels.
Route Save, session Launch, Restore, and source Update show a review before
typed confirmation. Cursor keys, Home, End, Delete, and paste edit text. Failed
submissions keep the form and values. The terminal restores its screen and input
mode on exit. `--print` prints the usage summary without opening either app.
`--tui` cannot be combined with `--print` or `--validate-toml`.

## Routes

The route library searches public model IDs, algorithms, endpoints, and model
choices. Select a route to edit its algorithm and model roles. The app opens one
editor at a time, even when the config contains hundreds of routes. **Refresh
suggestions** loads the endpoint's list; you can also type an unlisted model ID.
ChatGPT's private Codex endpoint uses the saved Codex model list and configured
targets. **Review and save…** shows the changes before **Save and restart**
applies them. The app keeps drafts during the app session until saving or
**Discard unsaved edits…**. Discarding changes only resets the editor. **Delete
route…** removes the selected route after confirmation, preserves its targets
and endpoint settings, and uses the same validation, backup, and restart as
saving. Refresh keeps drafts, focus, and scroll. Drafts belong to their config
file. Selecting another file keeps the drafts for the original file. Switching
algorithms preserves each algorithm's full set of choices.

**Save and restart** checks the complete edited file with the bundled
server's `--dry-run`. It preserves comments and unrelated tables, saves a backup
beside the config, and restarts the server. A failed check leaves the file
unchanged. A restart failure after saving is reported separately. The app re-reads
the config and rejects concurrent changes detected before replacement. A draft
from a different server config file is rejected before saving.

Switching algorithms writes the new algorithm's default settings and removes
settings specific to the old algorithm. Targets keep their existing model
settings, such as `extra_body` and `omit_body_fields`; the review names those
settings before saving when a model family changes. Check them if the new
provider rejects requests. Shared targets are copied when an edit would change
another route.

API keys entered for model lists stay in macOS Keychain or Windows Credential Manager. The app sends keys to
curl through stdin and omits them from frontend snapshots. An endpoint must be
configured to use authentication before it accepts a model-list key. An entered
key applies only to that endpoint's exact `base_url`. On macOS and Windows, submitting a key
always requests a fresh model list; a rejected key is not saved and leaves
the cached list unchanged. Model-list keys do not replace the server's
`api_key_env` credentials. The app and server LaunchAgents do not load shell
startup files.

On other platforms, the app rejects entered model-list keys; set the environment
variable named by the endpoint's `api_key_env` and make it available to the app
instead.

## Install coding-tool settings

**Connect tools** has one coding-tool selector for Codex CLI, Codex app, Claude
Code, and Pi. macOS finds the Codex app by its registered bundle identity,
including copies renamed to `ChatGPT.app`. Windows reads the current user's
`OpenAI.Codex` MSIX registration to find the app. Choose a route and settings location
to compare the current and proposed settings and their exact file paths.
Credential values stay hidden. The preview changes no files. **Apply routing…**
shows a confirmation before saving. Apply compares the reviewed file contents
with the current files and requires a new review if they changed. Progress
appears while the action runs. One dismissible notification shows the result,
with backup paths and diagnostics in **Details**. If the settings already match,
the button reads **Routing already applied**. An optional absolute file path
selects settings in another location. For Pi, choose `models.json`; the app also
uses `settings.json` beside it. A custom file cannot be combined with a saved
account.

The app's **Connect tools** page defaults to `sy.config.toml` for Codex CLI and
`codex -p sy`. The source installer's `SY_PROFILE` choice does not change this
destination. Select a custom file to update another profile. With the normal
user settings location, the Codex app option changes shared defaults in
`config.toml` for both the app and CLI. A custom file or saved account changes
only that destination; the coding tool must load those settings separately.
Claude Code gets environment settings in `settings.json`. Pi gets a custom
provider in `models.json` and defaults in `settings.json`. Unrelated settings
remain.

The app backs up each original file as `<filename>.switchyard-original`. If a
file did not exist, `<filename>.switchyard-original-missing` records its
absence. **Review restore…** lists the files and original backups. **Restore
settings** keeps a copy of each current file, then restores its backup or
removes the file if it was originally absent. The result lists the recovery
copies. A successful restore retires the original backup, so the next
installation backs up your current settings. If backup cleanup fails, retry
Restore before installing another route.

Install and Restore check every selected file before replacing any file.
Malformed settings and symlink settings files are rejected before writes. Each
replacement is atomic. If a later replacement fails, the app tries to restore
earlier files. A crash can leave Pi's two files out of sync, and rollback can
fail. Another process running as the same user can also change a file after the
app checks it. A failed replacement keeps original backups for manual recovery.
If backup cleanup fails after restoration, the remaining receipts keep the
restored contents available for a retry. Recovery copies remain until you remove
them.

Caller login routes require a matching coding tool and provider: Codex for
ChatGPT or Claude Code for Anthropic. Pi requires a route whose API credentials
the server owns. Configure those credentials on the server before installing
the route. Claude settings with a user API key or bearer token must be cleared
before installing a subscription route. Switching from an API route installed
by Switchyard removes its placeholder credentials only when the existing model
settings and local endpoint match that installation. If the local endpoint has
changed, restore the original settings first. Restart the coding tool after
changing its defaults. Project settings, shell variables, and command-line
options can override them; check Claude's `/status` for shell overrides.

For older installs that replaced `config.toml`, `cargo desktop uninstall` restores
`config.toml.direct` and keeps a copy of the active config before replacing it.

## Start sessions with separate logins and worktrees

Codex CLI and Claude Code support named accounts. Expand **Sign in to another
subscription account…** and enter a name. **Open login in Terminal…** opens the
tool's own login command. Complete the login, refresh, and select the account.
Switchyard keeps account directories under `~/.switchyard/accounts`; the coding
tool owns their login files. The selected account determines where **Connect
tools** saves settings and which login a new session uses. Existing sessions
keep their login. Switchyard does not read, copy, or rotate OAuth tokens or
combine subscription quotas.

In **Sessions**, choose a route and login, enter an absolute Git project path,
and click **Review session**, then **Open Terminal…**. Each session starts from
that checkout's HEAD on a new branch in `~/.switchyard/worktrees` and opens in
Terminal. Uncommitted files stay in the original checkout. Each launched session
has its own model settings; launching another session does not change user
defaults. Codex app uses its own workspace controls; choose Codex CLI for
worktree sessions.

Codex and Claude launches send a session ID for usage correlation. Pi stores its
session under the private agent directory beside the worktree. Its requests
need an integration that sends session and turn IDs to appear as a correlated
session. Otherwise, Usage lists its calls under **All sessions**.

Use `git worktree list` and `git worktree remove` to manage checkouts after
saving or discarding changes. Successful launches keep their branches and
private settings directories until you remove them. Named account directories
also remain until you remove them. If launch setup fails, the app tries to
remove the worktree, branch, and private settings it created; cleanup can fail.

The app does not show provider reset windows or manage task dependencies,
session resume, remote workspaces, or review comments. Provider limits still apply.

## Usage

**Usage** lists the actual model, public route, timestamp, input, cached, and
output tokens, and recorded session and turn IDs for each completed model call.
Filter by session or search for a turn, model, or route. A user turn may contain
several calls. Classifier calls are marked as routing overhead. Missing IDs
appear as **Session not recorded** or **Turn not recorded** in the desktop app
and **Not recorded** in the terminal app. Switchyard does not infer them from
prompts.

Recent history reads the last 8 MiB of the log plus one byte to check whether the
first record is complete. It keeps at most 5,000 complete records and excludes
partial records. Usage labels a limited view and counts unreadable records.
Overview starts with model totals. Use **Period** to select today, the past
seven local calendar days, or all retained history. **Compare** switches between
models, routes, and sessions; **Measure** switches between tokens and calls.
The chart shows the ten largest totals, and the table lists every matching
name with exact counts. Click a name to see its recent calls with the same
period filter. The daily chart shows usage by model for the past seven days.

Model, route, and daily totals use the incremental full-log reader. Session
comparisons and per-turn details use the bounded recent history and label its
limits. Replacing or truncating the log resets full-log totals. The terminal
app's hourly sparkline still uses recent history.

Charts use Apache ECharts 6.1.0, bundled in `ui/vendor` with its Apache license
and NOTICE. They load locally and require no CDN or frontend build tool.
The minified dependency adds about 1.1 MB before compression. Tooltips render
on the canvas, and tables provide readable counts and keyboard links.
The layout follows free references from [Carbon](https://www.carbondesignsystem.com/building-blocks/data-visualization/dashboards),
[Tabler](https://docs.tabler.io/ui/getting-started), and
[Primer](https://primer.style/product/components/page-layout/).

Routing history stores no prompt or response text.
Observed tokens do not report provider quotas, reset windows, or authoritative
billing. Both apps refresh usage at the configured `refresh_seconds` interval.

## Settings

The app reads `~/.switchyard/desktop.toml` on refresh and before each operation:

```toml
server_url = "http://127.0.0.1:4123"
routing_log = "~/.switchyard/routing.jsonl"
config_file = "~/.switchyard/composite.toml"
launchd_label = "com.nvidia.switchyard.server"
baseline_model = "gpt-5.6-sol"
refresh_seconds = 30
```

Use **Settings → Open server config** to edit advanced routing options.
Model lists are cached beside the app settings in `model-lists.json`.
Usage remains readable when the route config is missing or malformed.

## How savings are computed

The server writes one JSONL record per call, with the model that answered and
its token counts.

- **Actual** is every call priced at the model that served it, including
  Switchyard's own classifier calls.
- **Baseline** is the caller-facing calls only, priced as if each had used
  `baseline_model`. Classifier calls have no baseline counterpart, because
  without Switchyard they would not happen.

Savings are the difference, so routing overhead counts against the figure and a
bad day shows a negative number. Each period's cost and savings estimates stay
hidden until every model recorded in that period and the baseline model have
prices. Overview names missing model prices and explains estimates under **How
cost estimates work**.

Codex's "Approve for me" reviews count too. The server config that the
installer writes sends them to `codex-auto-review` on the ChatGPT backend. With
a ChatGPT login, Codex sends these reviews to `codex-auto-review` even without
Switchyard, so the installer prices that model at the `baseline_model` rates,
including the cached input rate.
Each review then adds the same amount to the actual cost and to the baseline,
so reviews do not change the dollar amount saved. They do lower the percentage
a little, because the baseline grows. If you change `baseline_model`, update
this price to match.

The app needs a price for the model ID that the routing log records for reviews,
which is the reviewer target's `id`. Without that price, the app hides the
savings estimate for every period that includes a review. The installer does not
overwrite an existing `composite.toml` or `desktop.toml`, so an existing install
needs the route and the price added by hand.

If a reviewer uses another model ID, add a price for that ID. Any period with
an unpriced review hides its savings and shows the missing-price hint.

These are list-price estimates. On a ChatGPT login there is no per-token bill,
so read them as "what this traffic would have cost at API rates".
