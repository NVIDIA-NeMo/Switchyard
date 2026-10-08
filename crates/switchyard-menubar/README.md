# Switchyard desktop and terminal apps

The macOS desktop app and terminal app show usage, edit routes, install
coding-tool settings, and open sessions in separate Git worktrees. The server
handles routing and continues when the app quits. The desktop app uses Tauri 2;
the terminal app uses Ratatui.

## Install and update

Run these commands from the repository root:

```sh
make install-macos
make uninstall-macos
```

The installer builds the app and server, installs `~/Applications/Switchyard.app`,
and loads two per-user LaunchAgents. It signs the app ad hoc for local use;
the app has no Developer ID signature or notarization. Reinstalling keeps
existing server settings, prices, accounts, and log history.

The installer writes `~/.codex/sy.config.toml` for `codex --profile sy` and keeps
`~/.codex/config.toml` unchanged. Set `SY_PROFILE` to choose a descriptive profile
name and `SY_MODEL` to choose an existing public route ID. New installs use
`composite-gpt-6-sol-gpt-6-luna`. Existing server configs stay saved, so set
`SY_MODEL` to one of their routes when installing with a custom config. The
profile leaves `approval_policy` and `sandbox_mode` to your existing Codex
settings. Reinstalling backs up the profile before replacing changed contents.

Open the app from Finder or Spotlight. Closing the window keeps the tray menu.
Choose **Open Switchyard** to return or **Quit Switchyard** to exit. The app
rejects Quit while an operation is queued or running; try again after it finishes.

**Settings → Update from source…** opens Terminal to rebuild and reinstall from
the checkout used for this installation. It builds that checkout's current commit
and keeps `SY_PROFILE` and `SY_MODEL`. Keep the checkout and a Rust toolchain
available. Update the checkout first to choose the code you want to build.

Run the terminal app with:

```sh
~/.switchyard/bin/switchyard-menubar --tui
# This command runs the terminal app from a checkout.
cargo run -p switchyard-menubar -- --tui
```

Tab changes sections. The footer lists navigation keys. In Routes, press `e` to
edit the algorithm and model roles. Tab changes form fields, Ctrl-U clears a
field, Enter submits, and Escape cancels. The terminal restores its screen and
input mode on exit. `--print` prints the usage summary without opening either
app. `--tui` cannot be combined with `--print` or `--validate-toml`.

## Routes

Each route label shows its public model ID, algorithm, endpoint name, and model
choices. In **Routes**, choose an algorithm and enter the endpoint and model for
each role. **Refresh models** loads the endpoint's list; you can also type an
unlisted model ID. ChatGPT's private Codex endpoint uses the saved Codex model
list and configured targets. The app keeps drafts until **Apply route and
restart** or **Revert draft**. Refresh keeps drafts unless the settings select
a different server config file.

**Apply route and restart** checks the complete edited file with the bundled
server's `--dry-run`. It preserves comments and unrelated tables, saves a backup
beside the config, and restarts the server. A failed check leaves the file
unchanged. A restart failure after saving is reported separately. Apply re-reads
the config and rejects concurrent changes detected before replacement. A draft
from a different server config file is rejected before saving.

Switching algorithms writes the new algorithm's default settings and removes
settings specific to the old algorithm. Targets keep their existing model
settings, such as `extra_body` and `omit_body_fields`; the result names those
settings when a model family changes. Check them if the new provider rejects
requests. Shared targets are copied when an edit would change another route.

API keys entered for model lists stay in macOS Keychain. The app sends keys to
curl through stdin and omits them from frontend snapshots. An endpoint must be
configured to use authentication before it accepts a model-list key. Submitting
a key always requests a fresh model list; a rejected key is not saved and leaves
the cached list unchanged. Model-list keys do not replace the server's
`api_key_env` credentials. The app and server LaunchAgents do not load shell
startup files.

## Install coding-tool settings

**Install…** shows the current model and endpoint for Codex CLI, Codex app,
Claude Code, and Pi. Choose a route and click **Install / update**. Codex CLI
gets `sy.config.toml` for `codex -p sy`. The Codex app option changes the defaults
in `config.toml` for both the app and CLI. Claude Code gets environment settings
in `settings.json`. Pi gets a custom provider in `models.json` and defaults in
`settings.json`. Unrelated settings remain.

The app backs up each original file as `<filename>.switchyard-original`. If a
file did not exist, `<filename>.switchyard-original-missing` records its absence.
**Restore previous settings** keeps a copy of the current file, then restores
the backup or removes a file that was originally absent.

Install and Restore check every selected file before replacing any file.
Malformed settings and symlink settings files are rejected before writes. Each
replacement is atomic. If a later replacement fails, the app tries to restore
earlier files. A crash can leave Pi's two files out of sync, and rollback can
fail. Another process running as the same user can also change a file after the
app checks it. The original backups remain available for manual recovery.

Caller login routes require a matching coding tool and provider: Codex for
ChatGPT or Claude Code for Anthropic. Pi requires a route whose API credentials
the server owns. Configure those credentials on the server before installing
the route. Claude settings with an explicit API key or bearer token must be
cleared before installing a subscription route. Restart the coding tool after
changing its defaults. Project settings, shell variables, and command-line
options can override them; check Claude's `/status` for shell overrides.

For older installs that replaced `config.toml`, `make uninstall-macos` restores
`config.toml.direct` and keeps a copy of the active config before replacing it.

## Start sessions with separate logins and worktrees

Codex CLI and Claude Code support named accounts. Enter a name and click **Add
account / open login** to open the tool's own login command in Terminal. Complete
the login, refresh, and select the account. Switchyard keeps account directories
under `~/.switchyard/accounts`; the coding tool owns their login files. The
selected account determines where Install saves settings and which login a new
session uses. Existing sessions keep their login. Switchyard does not read,
copy, or rotate OAuth tokens or combine subscription quotas.

In **Sessions**, choose a route and login, enter an absolute Git project path,
and click **Launch session**. Each session starts from that checkout's HEAD on a
new branch in `~/.switchyard/worktrees` and opens in Terminal. Uncommitted files
stay in the original checkout. Each launched session has its own model settings;
launching another session does not change user defaults. Codex app uses its own
workspace controls; choose Codex CLI for worktree sessions.

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
appear as **Not recorded**; Switchyard does not infer them from prompts.

Recent history reads the last 8 MiB of the log plus one byte to check whether the
first record is complete. It keeps at most 5,000 complete records and excludes
partial records. Usage labels a limited view and counts unreadable records.
Overview totals use a separate incremental reader, so the history limit does
not truncate weekly savings. Routing history stores no prompt or response text.
Observed tokens do not report provider quotas, reset windows, or authoritative
billing. Both apps refresh usage at the configured `refresh_seconds` interval.

## Settings

The app reads `~/.switchyard/menubar.toml` on refresh and before each operation:

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

Savings are the difference, so routing overhead counts against the figure and
a bad day shows a negative number. Today's and this week's dollar figures stay hidden until every model seen in that period, and the baseline
model, has a price, so a partial table cannot mislead. While this week's
figures are hidden, the Overview names the models that have no price.

Codex's "Approve for me" reviews count too. The server config that the
installer writes sends them to `codex-auto-review` on the ChatGPT backend. With
a ChatGPT login, Codex sends these reviews to `codex-auto-review` even without
Switchyard, so the installer prices that model at the `baseline_model` rates,
including the cached input rate.
Each review then adds the same amount to the actual cost and to the baseline,
so reviews do not change the dollar amount saved. They do lower the percentage
a little, because the baseline grows. If you change `baseline_model`, update
this price to match.

The app needs a price for the model ID that the routing log records for
reviews, which is the reviewer target's `id`. Without that price, the app
hides the Saved row for every period that includes a review. The installer does
not overwrite an existing `composite.toml` or `menubar.toml`, so an existing
install needs the route and the price added by hand.

If a reviewer uses another model ID, add a price for that ID. Any period with
an unpriced review hides its savings and shows the missing-price hint.

These are list-price estimates. On a ChatGPT login there is no per-token bill,
so read them as "what this traffic would have cost at API rates".
