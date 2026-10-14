# Use Switchyard with YardMaster

[YardMaster](https://github.com/bricke/YardMaster) is a self-hosted gateway that
lets a team share one `switchyard-server`. Each person gets their own token,
the provider keys stay on the server, and usage is broken down by person, route and
model. YardMaster starts and supervises `switchyard-server` itself and writes its
TOML from a setup wizard. This page was tested with YardMaster 0.2.0, which
builds Switchyard v0.3.0.

YardMaster is a separate project under the AGPL-3.0 license. It runs Switchyard as
an unmodified binary.

## How it uses Switchyard

```text
tools (Claude Code, Codex, SDKs…) ──ym_ token──▶ YardMaster gateway ──▶ switchyard-server ──▶ providers
```

- **Identity.** The gateway checks the person's `ym_` token, removes `Authorization`,
  `x-api-key` and cookies, and sets `x-switchyard-origin` to the person's name. It
  prefixes `x-switchyard-session-id` with the person, so session affinity never spans
  two people. Leave `forward_auth` off: client keys never reach Switchyard.
- **Config changes.** YardMaster writes the candidate TOML and checks it with
  `switchyard-server --dry-run`. On apply, it stops the server with `SIGTERM`, lets
  in-flight requests drain for `--shutdown-timeout`, starts it on the new TOML and
  polls `/health`. If the server does not come back healthy, YardMaster restarts it
  on the previous TOML and shows the tail of its stderr.
- **Usage.** YardMaster reads the routing log (`--routing-log-file`) for per-request
  records, and scrapes `/metrics` and `/v1/stats` for live charts. It stores metadata
  only, never prompts or responses.

## Run

```bash
docker run -d --name yardmaster --restart unless-stopped \
  -p 8080:8080 -p 8443:8443 \
  -v yardmaster-data:/data \
  -e YARDMASTER_SECRET_KEY="$(openssl rand -base64 32)" \
  ghcr.io/bricke/yardmaster:0.2
docker logs yardmaster        # shows a one-time setup code
```

Open `http://<host>:8080`, enter the setup code, and follow **Setup** to add
providers, models and routes. People then point their tools at the gateway with
their own token. See the
[YardMaster install guide](https://github.com/bricke/YardMaster/blob/main/docs/install.md).
