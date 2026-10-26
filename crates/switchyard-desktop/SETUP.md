# Desktop setup

Run `cargo desktop install` from the source checkout to build and install the
desktop app and server. On Linux, open the installed app with
`~/.switchyard/bin/switchyard-desktop --tui`. If you set `SY_HOME`, use its `bin`
directory instead.

Setup opens when neither the app settings file nor its default server config
exists. Existing files skip setup, including after reinstall. Invalid existing
files show an error so you can repair them in Settings. Routing logs and price
files alone do not skip setup.

Choose an existing server config or create one from bundled defaults. Select
the server address, routing log, optional price file, baseline model, and refresh
interval, then save. Use absolute file paths and an `http://` server address.
Creating a server config never replaces an existing file.
The server waits for saved app settings before starting. Save any required API
keys in **Settings → Server keys**. Use **Check config** in Settings, then restart
the server. The managed server listens on `127.0.0.1`; changing the address does
not expose it to the network.

**Settings → Configuration** uses the same form. Saving preserves comments in
the app settings file. Restart the server after changing its selected files or
port. The installed server reads those selections when it starts.

**Run setup again** reopens the form. **Remove app preferences** shows the exact
settings file before removal and checks that its contents have not changed
since review. Removal keeps a recovery copy and preserves server configs, saved
keys, history, accounts, and coding-tool settings. Use **Connect tools** to
restore a coding tool's settings.

In the terminal app, open Settings with `7`. Press `f` to edit configuration or
`d` to review removal of app preferences.

A fresh installation leaves the Codex profile unchanged. After setup saves a
server config, reinstalling creates or updates the Codex profile, which defaults
to `~/.codex/sy.config.toml`. The installer keeps a recovery copy of a changed
profile.
