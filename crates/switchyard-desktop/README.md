# Switchyard desktop and terminal apps

The macOS and Windows desktop apps and terminal app show usage, edit routes, install
harness settings, and open sessions in separate Git worktrees. The server
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
log history.

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
exact recovery copy of the selected profile before removal. Restore harness
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
Left-click the tray icon to see whether the server responds, today's and this
week's requests, tokens, spending, and savings, and each model's share of this
week's tokens. A note explains missing savings, such as a baseline model without
rates. On macOS the popup opens like a status item menu: its left edge follows
the icon, its top sits just below the menu bar, and it uses the system font and
the system light or dark appearance. The popup rereads the routing log every
`refresh_seconds` (at least 5) while it is open. **Restart server** and **Open
server config…** run from the popup and show their result there. **Settings…**,
**Connect tools…**, and **Open Switchyard** open the corresponding desktop view
and close the popup. **Quit Switchyard** quits the app. Escape or a click outside
the popup dismisses it. Clicking the tray icon again also closes it.

**Settings → Appearance** chooses **System**, **Dark**, or **Light** for the app
window. System follows the device's light or dark appearance. Existing saved
Canary preferences select Dark.

On macOS, the navigation sidebar shows native regular glass where supported,
with the native sidebar material as a fallback. The popup uses the same glass
effect with a popover material fallback. The page header, editors, charts, and
tables use solid backgrounds. The app checks macOS Reduce Transparency and Reduce
Motion settings every second in both windows. Reduce Transparency covers the
native material with opaque backgrounds; a failed accessibility check also
restores opaque backgrounds. Windows uses opaque backgrounds.

Right-click the tray icon for the same summary as text rows, followed by
**Restart server**, **Open server config…**, **Settings…**, **Open
Switchyard**, **Connect tools…**, and **Quit Switchyard**. The menu rereads the
routing log every `refresh_seconds` (at least 5), like the earlier menu bar app.
A failed restart opens the app window with the error. Choose
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
On Linux, login and session launch open the terminal program named by `TERMINAL`,
or a detected terminal program. Open-file actions use `xdg-open`. Model discovery
needs `curl`; session creation needs Git and the selected harness.
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

The terminal app runs every action that the desktop app sends. It has seven
pages: Overview, Calls, Models, Routes, Harnesses, Sessions, and Settings. When
the full page names do not fit, the tab bar shows the first three letters of
each name. Overview shows the server, today's and the past seven days' calls,
spending, and savings, calls per hour, and the top models; it has no keys of its
own. The status bar shows the current page's keys, and `?` opens a window that
lists the keys of every page:

| Page | Key | What it does |
|---|---|---|
| Every page | `1`–`7`, Tab, Shift-Tab, ←, → | Switch pages |
| Every page | `r`, F5 | Reload usage, calls, and server health |
| Every page | `?` | Show or hide the list of keys |
| Every page | `q`, Ctrl-C | Quit the app; the server keeps running |
| Calls | ↑, ↓, `j`, `k`, PgUp, PgDn, `g`, `G` | Select a call |
| Calls | `/` | Filter by model, route, session, or turn |
| Calls | `s` | Show one session at a time, then all sessions again |
| Calls | Enter | Open the selected call's details |
| Calls | Esc | Clear the filter and the session |
| Models | `p`, `]`, `[` | Switch between today, the past 7 days, and the whole log |
| Routes | ↑, ↓, `j`, PgUp, PgDn, `g`, `G` | Select a route |
| Routes | `/` | Filter by public model ID, config name, algorithm, endpoint, or model |
| Routes | Esc | Clear the filter |
| Routes | `e` | Edit the route's algorithm and models |
| Routes | `a` | Create a route |
| Routes | `n` | Rename the route |
| Routes | `c` | Duplicate the route |
| Routes | `d` | Delete the route |
| Routes | `m` | Load the model IDs that an endpoint offers |
| Routes | `k` | Save an endpoint's model-list key, then load its model IDs |
| Harnesses | ↑, ↓, `j`, `k`, PgUp, PgDn, `g`, `G` | Choose the route to install |
| Harnesses | `t` | Select the next harness |
| Harnesses | `a` | Select the next saved account, then the harness's own settings again |
| Harnesses | `x` | Use a custom settings file; leave it empty to use the detected settings |
| Harnesses | `p` | Preview the settings diff |
| Harnesses | `i` | Review and apply routing to the harness |
| Harnesses | `u` | Review and restore the original settings |
| Harnesses | `n` | Sign in to another subscription account |
| Harnesses | `l` | Sign in to the selected account again |
| Harnesses | `o` | Open the selected account folder in the editor |
| Harnesses | `d` | Archive the selected account |
| Sessions | ↑, ↓, `j`, `k`, PgUp, PgDn, `g`, `G` | Choose the route for a new session |
| Sessions | `t` | Select the next harness for a new session |
| Sessions | `a` | Select the next saved account for a new session |
| Sessions | `s` | Select the next recorded session |
| Sessions | `l` | Review and launch a worktree session |
| Sessions | `n` | Sign in to another subscription account |
| Sessions | `o` | Open the selected session folder in the editor |
| Sessions | `d` | Review and remove the selected session |
| Settings | ↑, ↓, `j`, `g`, `G` | Select an action |
| Settings | Enter | Run the selected action |
| Settings | `o` | Open the server config |
| Settings | `c` | Check the server config |
| Settings | `b` | Restart the server |
| Settings | `a` | Open the app settings |
| Settings | `e` | Change the baseline model and refresh interval |
| Settings | `p` | Open the model prices |
| Settings | `k` | Check the saved server keys |
| Settings | `s` | Save a server key |
| Settings | `i` | Import a server key from your login shell |
| Settings | `x` | Remove a saved server key |
| Settings | `u` | Review an update from source |
| Settings | `l` | Open the update log |
| Settings | `v` | Open the server logs folder |

The Harnesses and Sessions pages share the selected harness, account, and
route, and the Routes filter limits the routes that they offer. The Harnesses
page shows the settings preview only while the harness, account, and route
still match it.

Actions that change files ask you to type a confirmation, and most show a
review first. Editing, creating, renaming, and duplicating a route ask for
`SAVE`. Deleting a route asks for its public model ID, and archiving an account
asks for the account name. Applying routing asks for `INSTALL`, restoring
settings asks for `RESTORE`, launching a session asks for `LAUNCH`, removing a
session asks for `REMOVE`, and source updates ask for `UPDATE`. Restarting the
server asks you to press `y`. In a review, ↑ and ↓ scroll the preview, Enter
confirms the typed word, and Esc cancels.

While you filter calls or routes, Enter keeps the filter, Esc clears it, Ctrl-U
erases the text, and Ctrl-W erases the last word. Short messages appear in the
status bar, and Esc dismisses them. Errors and warnings stay until you dismiss
them. A warning means an action did only part of its work, such as saving a
route when the server could not restart. Long messages and results with details
open a window that ↑ and ↓ scroll and Esc, Enter, or `q` closes.

In forms, Tab and Shift-Tab change fields, ↑ and ↓ choose algorithms and
endpoints, and Enter submits. Esc cancels, and Ctrl-U clears a field. The left
and right arrow keys, Home, End, Backspace, Delete, and paste edit text. Key
fields show one bullet per character. A failed submission keeps the form and
its values. The terminal restores its screen and input mode on exit. `--print`
prints the usage summary without opening either app. `--tui` cannot be combined
with `--print` or `--validate-toml`.

## Routes

**Create route…** asks for the public model ID that harnesses request, the config
name for the route's table, the routing algorithm, and one endpoint and model for
each model the algorithm uses. For example, **Stage router** asks for a capable
and an efficient model, and **Random split** starts with two models and can add
more. The config name follows the public model ID until you edit it. Switching
algorithms keeps the models that both algorithms use.

Every choice field in the app is a searchable list. Type any letters of a name in
order, such as `g6l` for `gpt-6-luna`, and separate words to match each one.
Arrow keys move through the list, Enter picks, and Escape closes it. Model fields
also accept an ID that the list does not show.

The route library searches public model IDs, algorithms, endpoints, and model
choices. Select a route to edit its algorithm and model roles. The app opens one
editor at a time, even when the config contains hundreds of routes. **Refresh
suggestions** loads the endpoint's list; you can also type an unlisted model ID.
ChatGPT's private Codex endpoint uses the saved Codex model list and configured
targets. The diagram shows the selected algorithm's model roles; click a role
to focus its endpoint field. At narrow widths, **Route library** and **Selected
route** switch between the route list and selected editor. **Review changes…**
shows the changes before **Save and restart** applies them. The app keeps drafts
during the app session until saving or
**Discard unsaved edits…**. Discarding changes only resets the editor. **Delete
route…** removes the selected route after confirmation, preserves its targets
and endpoint settings, and uses the same validation, backup, and restart as
saving. Refresh keeps drafts and scroll and restores focus to matching form fields.
Drafts belong to their config file. Selecting another file keeps the drafts for
the original file. Switching algorithms preserves each algorithm's full set of choices.

**Save and restart** asks the installed helper to check the complete edited file
with the installed server's `--dry-run`. The check uses the saved server keys,
which are the keys the server starts with. If the edited file names a key that
is not saved, the check fails. See [Server Keys](../../INSTALLATION.md#server-keys).
The app preserves comments and unrelated tables, saves a backup
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
`api_key_env` keys, which you save under **Settings → Server keys**. The app and
server LaunchAgents do not load shell startup files.

On other platforms, the app rejects entered model-list keys; set the environment
variable named by the endpoint's `api_key_env` and make it available to the app
instead.

## Server keys

**Settings → Server keys** lists each variable that the server config names
with `api_key_env` and says whether a key is saved for it. The installed server
reads only these saved keys, never your shell's variables. To save a key, type
or paste it in the hidden field and press Enter. Select **Show key** to check
what you typed, or choose **Import from shell** to read the variable once from
your login shell. **Remove…** deletes a saved key after you confirm. The
installed helper saves each key in macOS Keychain or Windows Credential Manager.
The app passes the key to the helper through standard input and never stores
it. Restart the server to use a changed key. After a change, the section shows
**Restart server now**.

## Install harness settings

**Connect tools** has one harness selector for Codex CLI, Codex app, Claude
Code, and Pi. A harness is an app or CLI that sends model requests through
Switchyard. macOS finds the Codex app by its registered bundle identity,
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
only that destination; the harness must load those settings separately.
Claude Code gets environment settings in `settings.json`. Pi gets a custom
provider in `models.json` and defaults in `settings.json`. Unrelated settings
remain.

Codex compacts a thread when its usage nears the model's context window, but it
does not know a Switchyard route ID. When the route sets `context_window` in the
server config, the Codex settings get `model_context_window` with that value, and
Pi's `contextWindow` uses it too. Without it, Codex never compacts on the route,
and Pi uses 128,000. A `model_catalog_json` entry for the route takes precedence
over `model_context_window`, so a catalog that lists the route with the wrong
window makes Codex compact at that size. Correct the catalog entry instead.

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

Caller login routes require a matching harness and provider: Codex for
ChatGPT or Claude Code for Anthropic. Pi requires a route whose API credentials
the server owns. Configure those credentials on the server before installing
the route. Claude settings with a user API key or bearer token must be cleared
before installing a subscription route. Switching from an API route installed
by Switchyard removes its placeholder credentials only when the existing model
settings and local endpoint match that installation. If the local endpoint has
changed, restore the original settings first. Restart the harness after
changing its defaults. Project settings, shell variables, and command-line
options can override them; check Claude's `/status` for shell overrides.

## Start sessions with separate logins and worktrees

Codex CLI and Claude Code support named accounts. Expand **Sign in to another
subscription account…** and enter a name. **Open login in Terminal…** opens the
harness’s own login command. Complete the login, refresh, and select the account.
Switchyard keeps account directories under `~/.switchyard/accounts`; the harness
owns their login files. The selected account determines where **Connect
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

Usage displays up to 100 calls per page. **Previous calls** and **Next calls** move
through the filtered results. **Copy filtered calls** copies all matching calls,
including calls on other pages. Sidebar navigation stays available while the app
refreshes the displayed data.

**Usage** lists the actual model, public route, timestamp, input, cached, and
output tokens, and recorded session and turn IDs for each completed model call.
Filter by session or search for a turn, model, or route. A user turn may contain
several calls. Classifier calls are marked as routing overhead. Missing IDs
appear as **Session not recorded** or **Turn not recorded** in the desktop app
and **Not recorded** on the terminal app's Calls page. Switchyard does not
infer them from prompts.

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
limits. Replacing or truncating the log resets full-log totals. The calls-per-hour
chart on the terminal app's Overview page also uses recent history.

Charts use Apache ECharts 6.1.0, bundled in `ui/vendor` with its Apache license
and NOTICE. They load locally and require no CDN or frontend build tool.
The minified dependency adds about 1.1 MB before compression. Tooltips render
on the canvas, and tables provide readable counts and keyboard links.

Routing history stores no prompt or response text.
Observed tokens do not report provider quotas, reset windows, or authoritative
billing. Both apps refresh usage at the configured `refresh_seconds` interval.

## Settings

The app reads `~/.switchyard/desktop.toml` on refresh and before each operation:

```toml
server_url = "http://127.0.0.1:4123"
routing_log = "~/.switchyard/routing.jsonl"
config_file = "~/.switchyard/composite.toml"
# This optional file replaces the bundled price table.
# pricing_file = "~/.switchyard/prices.toml"
launchd_label = "com.nvidia.switchyard.server"
baseline_model = "gpt-6-sol"
refresh_seconds = 30
```

Use **Settings → Open server config** to edit advanced routing options and
**Settings → Open model price override** to edit a configured price file. Without an
override, **About model prices** explains how to configure one.
Model lists are cached beside the app settings in `model-lists.json`.
Usage remains readable when the route config is missing or malformed.

## How costs and savings are computed

The server writes one JSONL record per call, with the model that answered, its
token counts, and its cost. The server and app use the same
[bundled rates](../switchyard-costs/prices.toml) by default. To replace the table,
pass `--pricing-file PATH` to the server and set `pricing_file` to the same TOML or JSON
file in app settings. The [Pi cost guide](../../docs/integrations/pi.md#estimate-the-cost)
describes the file format. If an explicitly configured file is missing or invalid,
the server and app report an error instead of using bundled rates.

For each answer, the server records the charge that the provider reports, from
`usage.cost` in the response or, for a buffered LiteLLM response, from the
`x-litellm-response-cost` header. Otherwise it records an estimate from those
rates. The server cannot see the headers of classifier and judge calls, so
those calls use `usage.cost` or the estimate. The app adds up these recorded
costs and does not price calls itself. The log covers the calls that the server
observed; a request cancelled before its answer finished may have no record.

- **Cost** is every recorded call, including Switchyard's own classifier calls.
- **Baseline** is the caller-facing calls only, with their tokens priced at the
  `baseline_model` rates from the selected price table. Classifier calls have no
  baseline counterpart, because without Switchyard they would not happen.

Savings are the difference, so routing overhead counts against the figure and a
bad day shows a negative number. A period shows savings only when every call in
it has a recorded cost, every answer has token counts, and the selected table has
rates for `baseline_model`, including `cache_write_1h` when answers wrote to a
one-hour cache. Calls recorded before the server priced calls have no cost, and
neither do calls to a model that has no rates and no provider charge. Overview counts these calls and
explains costs under **How costs work**.

The server loads rates when it starts, so restart it after editing an override
file. Bundled rates change only when you install a version with updated rates.
Each record keeps the cost from when its call finished, so editing rates does
not change past totals.

The bundled table includes OpenAI and Claude text-token rates in USD per million
tokens. For GPT-5.6 and newer, `cache_write` uses OpenAI's 30-minute rate, which
is 1.25 times the input rate. Older OpenAI models use the ordinary input rate
for separately counted writes, with no extra write fee. Claude rates distinguish
five-minute writes (`cache_write`) from one-hour writes (`cache_write_1h`).
Estimates exclude long-context and service-tier adjustments, OpenAI's 12-hour
cache-write pricing, tool fees, and time-based cache storage. Entries with provider
prefixes use list-rate estimates; billed charges may differ.

Installation uses bundled rates without creating a price file and preserves
existing custom price files.
For older `[prices]` settings, remove the table to use bundled rates, or convert
it to the shared TOML or JSON rate format and select it with `pricing_file`. The
installer does not migrate old settings.

Codex's "Approve for me" reviews count too. The server config that the
installer writes sends them to `codex-auto-review` on the ChatGPT backend. With
a ChatGPT login, Codex sends these reviews to `codex-auto-review` even without
Switchyard, so the bundled rates price that model at the `gpt-6-sol`
rates. Each review then adds the same amount to the cost and to the baseline,
so reviews do not change the dollar amount saved. They do lower the percentage
a little, because the baseline grows. If you change `baseline_model`, use a
custom replacement table with matching `codex-auto-review` rates.

The server needs rates for the model ID that the routing log records for
reviews, which is the reviewer target's `id`. Without them, a review that the
provider does not price has no recorded cost, and the app hides the savings for
every period that includes it. The installer does not overwrite an existing
`composite.toml` or an existing price override. Older installations may need the
review route added by hand. The bundled table includes `codex-auto-review`; a
custom replacement table must include its own reviewer rates. If a reviewer
uses another model ID, add rates for that ID.

These are list-price estimates unless the provider reports a charge. On a
ChatGPT login there is no per-token bill, so read them as "what this traffic
would have cost at API rates".
