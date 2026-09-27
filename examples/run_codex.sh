# Local Setup with Codex + Switchyard: composite router (Terra + Stage, Sol/Luna)
#
# Terra classifies the user's turn and sets the tier. Stage drives the tool
# loop underneath it. Everything runs on your own ChatGPT login, no API key.
#
# Prereqs: Rust/Cargo, Codex CLI already logged in.

cargo build -p switchyard-server --release

cat > /tmp/composite.toml <<'EOF'
schema_version = 1

[llm_clients.chatgpt_backend]
format = "openai_responses"
base_url = "https://chatgpt.com/backend-api/codex"
forward_auth = true

[targets.capable]
id = "gpt-5.6-sol"
llm_client = "chatgpt_backend"

[targets.efficient]
id = "gpt-5.6-luna"
llm_client = "chatgpt_backend"

# chatgpt.com/backend-api/codex is Codex CLI's own private endpoint, not the
# public OpenAI Responses API. It 400s unless store=false and stream=true are
# set explicitly, and it rejects max_output_tokens outright, so the
# classifier's own token cap has to be dropped before the request goes out.
[targets.terra]
id = "gpt-5.6-terra"
llm_client = "chatgpt_backend"
extra_body = { store = false, stream = true }
omit_body_fields = ["max_output_tokens"]

[routes.switchyard]
id = "switchyard"
type = "composite"

[routes.switchyard.classifier]
target = "terra"
base_threshold = 0.5
classify_trigger = "user_turn"

[routes.switchyard.stage]
capable_target = "capable"
efficient_target = "efficient"
confidence_threshold = 0.5
EOF

./target/release/switchyard-server --config /tmp/composite.toml --dry-run
./target/release/switchyard-server --config /tmp/composite.toml --port 4123 &

cat > ~/.codex/composite_test.config.toml <<'EOF'
model = "switchyard"
model_provider = "sy_composite"
approval_policy = "never"
sandbox_mode = "workspace-write"

[model_providers.sy_composite]
name = "Switchyard Composite"
base_url = "http://127.0.0.1:4123/v1"
wire_api = "responses"
requires_openai_auth = true
EOF

codex --profile composite_test

# What's happening
#
# Terra reads the user's turn once and sets the tier, held for every tool
# call until the next user turn. Stage's own signals still run underneath
# and can escalate mid-turn on a genuine failure, they just start from
# Terra's tier instead of the picker's static default.
#
# Verify
#
# Check the server's log or curl -s http://127.0.0.1:4123/v1/stats.
# selected_model should follow Terra's read of the prompt, not just default
# to Luna, and a tool failure mid-turn should still push it to Sol.
#
# Known quirks
#
# chatgpt.com/backend-api/codex is Codex CLI's own private endpoint, not the
# public OpenAI Responses API, and it only accepts requests shaped the way
# the real Codex CLI sends them: store=false and stream=true set explicitly,
# and no max_output_tokens field at all (it 400s on that one outright). The
# terra target's extra_body/omit_body_fields above exist to make the
# classifier's request match that shape; capable/efficient don't need them
# because they forward the caller's own Responses body verbatim.

