# Switchyard Profile V2 Design

Audience: contributors adding or reviewing Switchyard profiles in Rust, Python, or bindings.

Prerequisites: familiarity with `ChatRequest`, `ChatResponse`, `LlmTarget`, async execution, and
the existing Switchyard request path. You do not need to know the older factory, bundle, graph, or
processor pipeline internals to understand v2.

## Summary

V2 is the simpler component model for Switchyard as a whole. The first implementation lives in
`switchyard-components-v2`, but the idea is not Rust-only: Python should converge on the same
profile-centered model instead of keeping a separate factory/bundle architecture.

V2 replaces the older habit of decomposing every behavior into separate request processors,
backends, response processors, graph fragments, bundles, and factories with one central runtime
unit:

```text
Profile
```

A profile owns the behavior for one addressable Switchyard mode. That can be passthrough,
random routing, latency-aware routing, a classifier stage-router, observability-only middleware, or a
future agent-facing policy. The profile decides how a request is prepared, which backend is called,
what response cleanup is needed, and where observability is recorded.

The goal is not to make Switchyard less capable. The goal is to stop spreading one behavior across
many abstractions when most contributors need to reason about one thing: "what happens when this
profile receives a request?"

## Why V2 Exists

The original Switchyard shape was even looser than today's Python chain: a linked list of
"strategies" that accepted `Any` and returned `Any`. That made early experiments easy, but it also
made behavior hard to type, validate, compose, and review. The Python chain was the first cleanup:
split the untyped strategy chain into explicit request, backend, and response roles:

```text
request-side work -> LLMBackend -> response-side work
```

That shape fixed real problems and made incremental porting possible, but it also carried Python
object boundaries into places where they no longer help. A single product feature could require
edits in several places:

- request processor config,
- request processor implementation,
- backend config,
- backend implementation,
- response processor config,
- response processor implementation,
- graph or table wiring,
- factory or bundle glue,
- tests that needed to know the glue shape.

For new contributors, that is the wrong entry point. They should not need to learn an orchestration
framework before implementing a routing policy, whether they are writing the implementation in Rust
or exposing it through Python.

V2 starts from the product-level object instead:

```text
one profile config -> one profile runtime -> one request lifecycle
```

This makes the code easier to read, easier to test, and harder to accidentally over-abstract.

The config side follows the same principle. A config file should describe endpoints, targets, and
profiles. It should not ask users to assemble request processors, response processors, graph nodes,
bundles, or backend wrappers by hand.

## Core Mental Model

The v2 runtime has two surfaces:

- an object-safe runtime surface for code that needs to run a profile or request a
  decision without selected-backend dispatch;
- a typed hook surface for code that wants to inspect or embed the profile's request and response
  hooks.

In Rust, that target shape is:

```rust
#[async_trait]
pub trait Profile: Send + Sync {
    async fn run(&self, input: ProfileInput) -> Result<ChatResponse>;

    async fn decide(&self, request: RoutingRequest) -> Result<RoutingDecision> {
        Err(SwitchyardError::DecisionUnsupported {
            profile_id: request.decision_profile.profile_id,
        })
    }
}

#[async_trait]
pub trait ProfileHooks: Send + Sync {
    type ProcessedRequest: Send + Sync;

    async fn process(&self, input: ProfileInput) -> Result<Self::ProcessedRequest>;

    async fn rprocess(
        &self,
        processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse>;
}
```

Conceptually:

```text
process(request)              -> profile-specific processed request
run(request)                  -> final response
decide(routing_request)       -> routing decision without selected-backend dispatch
rprocess(processed, response) -> processed response
```

`Profile` is the erased runtime contract. Servers, config loaders, and generic profile tables
can store `Box<dyn Profile>` and call `run()` or `decide()` without knowing a profile's private
request state. `decide()` defaults to a typed unsupported-profile error so the one configured
runtime registry can serve both full requests and capability-checked decision calls.

`ProfileHooks` is the typed authoring and embedding contract. The associated `ProcessedRequest`
type is the profile-owned wrapper around the prepared request and any request-side decision the
profile made. It replaces untyped metadata bags and avoids forcing every profile into a shared
decision enum.

These methods are three related entry points into the same profile, not three required stages of a
hidden pipeline:

- `run()` is the authoritative serving path. It owns the complete request lifecycle and is the
  method a Switchyard server should call.
- `decide()` is the decision-only path. Supporting profiles reuse their request-side policy but
  stop before selected-backend dispatch; other profiles return a typed unsupported error.
- `process()` is the request-side library hook. It returns the prepared request plus any typed
  profile-specific state needed later, such as a random-routing decision or an initial
  latency-service target selection.
- `rprocess()` is the response-side library hook. It receives the processed request state and the
  backend response so the profile can perform cleanup, normalization, or accounting without a
  side-channel.

`run()` should normally call `process()` and `rprocess()` when those hooks express the same
lifecycle cleanly. If `process()` seems unable to carry the state that `run()` needs, the first fix
is to make that profile's `ProcessedRequest` type richer, not to add a generic metadata map. A
profile should bypass its hooks only when the lifecycle genuinely cannot be represented by a
request-side wrapper, and that reason should be documented on the concrete profile.

The self-contained serving flow looks like this:

```text
HTTP / SDK wire request
    |
    | endpoint + translation layer
    v
ChatRequest
    |
    | Profile::run()
    |   - process into profile-specific request state
    |   - prepare or rewrite the backend request
    |   - choose target/backend
    |   - call backend
    |   - normalize response
    |   - record stats/errors
    v
ChatResponse
    |
    | endpoint + translation layer
    v
HTTP / SDK wire response
```

`process()` and `rprocess()` are intentionally still present. They are not a return to the old
pipeline hierarchy. They exist because Switchyard can be embedded as middleware in systems that
own their own transport or model runtime:

```text
caller-owned request object
    |
    | caller converts or already has a ChatRequest
    |
    v
ChatRequest
    |
    | Profile::process()
    v
profile-specific processed request
    |
    | processed.request()
    v
prepared ChatRequest
    |
    | caller-owned HTTP client, model runner, batcher, or agent runtime
    v
ChatResponse
    |
    | Profile::rprocess(processed, response)
    v
processed ChatResponse
    |
    | caller returns or translates response
    v
caller-owned response object
```

In middleware mode, the reusable pieces are the profile's request hook, response hook, shared
backend adapters, translation layer, and explicit helpers such as stats recorders. The caller owns
transport and the actual model call. The profile still owns the request and response behavior it
exposes through those hooks.

The important rule is that `process()` must return more than a bare `ChatRequest` when the profile
makes a meaningful request-side decision. For example, random routing should return a
`RandomRoutingProcessedRequest` containing both the rewritten request and the routing decision.
Latency-service routing can return a `LatencyServiceProcessedRequest` containing the rewritten
request and the selected target candidate. Those wrappers are profile-specific types, not variants
of one universal metadata object.

Do not implement unsupported hooks with `panic!`, `unimplemented!()`, or runtime
`NotImplementedError`. If every profile must expose the hook, make the hook total with a meaningful
identity wrapper for simple profiles. Decision-only routing is the explicit exception: it stays on
the object-safe profile contract so one registry owns each runtime, and unsupported profiles return
the typed `DecisionUnsupported` error instead of being rebuilt in a parallel capability registry.

## What A Profile Owns

A profile owns the policy boundary. If a reviewer asks, "where is this behavior implemented?", the
answer should usually be one module under `src/profiles/`.

For example:

```text
src/profiles/passthrough.rs
src/profiles/random_routing.rs
src/profiles/latency_service/
```

Each profile should keep these pieces close together:

- the user-facing config struct,
- validation that is specific to that config,
- runtime construction,
- request-side behavior,
- the processed-request wrapper returned by `process()`,
- any profile-specific request decision type,
- backend selection and calls,
- response-side behavior,
- stats and error accounting,
- focused tests for the profile's invariants.

Shared engines are still allowed. Random routing can reuse `RandomRoutingEngine`. Latency routing
can split health polling and selection into small private modules. The distinction is that these
helpers are implementation details of a profile, not a second public orchestration model.

## What V2 Removes From The Contributor Path

V2 is designed so profile authors do not have to build or understand these concepts:

- `ProxyContext` as a mutable cross-stage side channel,
- request processor trait objects,
- response processor trait objects,
- graph fragments as a profile output format,
- table builders as the primary authoring API,
- middleware bundles,
- factories,
- served-model side channels.

Those ideas may still exist elsewhere during migration or for compatibility. They should not be
the design vocabulary for new profile code.

For now, v2 does not treat backward compatibility as a requirement. If an older Python factory,
bundle, route shape, or wrapper only exists to preserve the previous object hierarchy, it is a
candidate for removal rather than preservation. Temporary aliases can be added later only when a
release decision explicitly needs them; they should not shape the core design.

## Deliberate Helpers Instead Of Implicit Middleware

Helpers are still valuable. The difference is that v2 makes helper use visible at the profile
boundary.

Stats are the clearest example. In the older model, stats could be hidden behind separate request
processors, response processors, or a stats backend wrapper. That hid the control flow:

```text
request processor records start
backend records selection
response processor records usage
```

The profile author then had to know which side channel connected those pieces. That side channel
was often `ProxyContext`.

In v2, a profile records stats at the semantic point where it knows the truth:

```rust
let backend_started_at = Instant::now();
let response = match selected_backend.call(processed.request().clone()).await {
    Ok(response) => response,
    Err(error) => {
        self.record_error(&processed)?;
        return Err(error);
    }
};

let backend_latency_ms = backend_started_at.elapsed().as_secs_f64() * 1000.0;
self.record_success(&processed, Some(backend_latency_ms))?;

let usage = response.body().map(usage_from_body).unwrap_or_default();
self.record_usage(&processed, usage, Some(total_latency_ms), Some(routing_overhead_ms))?;
```

This is more explicit, and that is the point. The profile knows:

- which target was selected,
- which model should receive accounting,
- which profile-specific decision should receive accounting,
- which failures happened before a response existed,
- which latency is backend time versus profile overhead.

A stats helper should reduce mechanical work, not hide the lifecycle. A good helper answers,
"how do I record this known event consistently?" A bad helper answers, "where did this event come
from?" by forcing the reader to chase implicit middleware.

## Backend Reuse

V2 does not require rewriting every backend immediately. The crate currently adapts existing native
OpenAI and Anthropic backends through a small `ProfileBackend` surface that does not expose
`ProxyContext`.

That adapter is a migration tool. The target direction is:

```text
Profile
    owns routing and lifecycle
Backend
    owns provider call behavior
Translation
    owns wire-format conversion
```

Backends remain a reasonable abstraction because provider calls are real reusable behavior. The
problem v2 fixes is not "all abstractions are bad." The problem is splitting one profile's policy
across too many orchestration objects.

## Config Shape

Each profile config lives beside its runtime and declares a stable serialized type. In Rust, the
macro handles the repetitive serde and config-resolution surface:

```rust
#[profile_config("random-routing")]
pub struct RandomRoutingProfileConfig {
    #[profile_target]
    pub strong: LlmTarget,
    #[profile_target]
    pub weak: LlmTarget,
    #[serde(default = "default_strong_probability")]
    pub strong_probability: f64,
    #[serde(default)]
    pub rng_seed: Option<u64>,
}
```

The macro standardizes the boring parts:

- clone/debug/serde derives,
- strict unknown-field rejection,
- a stable `PROFILE_TYPE`,
- `profile_type()`,
- profile-owned parsing from serialized config,
- `#[profile_target]` resolution from config IDs into concrete `LlmTarget` values.

It should not hide profile-specific meaning. Cross-field validation belongs in normal Rust when it
is policy-specific. For example, validating duplicate latency targets is easier to review as plain
code than as macro magic.

Python should follow the same rule even if the mechanism is a decorator or class registration
instead of a Rust proc macro. A profile's config belongs next to the profile implementation; it
should not live in a distant central factory file.

The intended config authoring rule is:

```text
shared config behavior can be macro-generated
profile policy stays in the profile module
```

## User-Facing Config Model

The v2 config loader is the user-facing entry point for profile construction. It accepts the same
schema as YAML, JSON, or TOML, with `${VAR}` interpolation before typed deserialization:

```yaml
endpoints:
  nvidia:
    base_url: https://integrate.api.nvidia.com/v1
    api_key: ${NVIDIA_API_KEY}
    timeout_secs: 120.0

# Targets are named by the model id they serve.
targets:
  nvidia/frontier-model:
    endpoint: nvidia
    model: nvidia/frontier-model
    format: openai

  nvidia/fast-model:
    endpoint: nvidia
    model: nvidia/fast-model
    format: openai

profiles:
  direct:
    type: passthrough
    target: nvidia/fast-model

  smart-stage-router:
    type: stage_router
    capable: nvidia/frontier-model
    efficient: nvidia/fast-model
    fallback_target_on_evict: nvidia/frontier-model
    picker: capable_first
    confidence_threshold: 0.7
```

The intended load path is:

```text
YAML / JSON / TOML
    |
    v
ProfileConfigDocument
    |
    v
ProfileConfigPlan
    |
    v
Box<dyn Profile>
```

`ProfileConfigDocument` is the parsed file shape. It is allowed to contain config-facing
references, such as `target: nvidia/fast-model` or `endpoint: nvidia`. It is not runtime-ready.

`ProfileConfigPlan` is the resolved plan. It has concrete `LlmTarget` values and typed profile
configs. The design deliberately does not rebuild the old graph/table/factory system. The plan is
only the point where a parsed config has been validated enough to build one profile or all profiles.

The `ProfileConfig` trait is the build contract that keeps runtime construction explicit. The macro
can generate parsing and target-resolution glue, but every profile config still has to say how it
turns into its runtime profile. The concrete runtime must implement the object-safe `Profile`
runtime contract and the typed `ProfileHooks` embedding contract; config plans erase only to
`Profile` for serving.

The config split should be enforced with a trait shaped like this:

```rust
pub trait ProfileConfig: ProfileConfigDefinition {
    type Runtime: Profile + ProfileHooks + 'static;

    fn build(&self) -> Result<Self::Runtime>;

    fn build_boxed(&self) -> Result<Box<dyn Profile>> {
        Ok(Box::new(self.build()?))
    }
}
```

That means a profile config can rely on macro-generated parsing, but it cannot skip runtime
construction. The `#[profile_config]` macro emits a compile-time assertion that the config
implements `ProfileConfig`, so the compiler requires each config to provide `build()`. Because
`type Runtime: Profile + ProfileHooks`, a profile that only implements `run()` but leaves
`process()` or `rprocess()` out does not satisfy the design; it should provide real hook behavior
or the hook should move to a narrower capability trait.

```rust
impl ProfileConfig for RandomRoutingProfileConfig {
    type Runtime = RandomRoutingProfile;

    fn build(&self) -> Result<Self::Runtime> {
        let router = RandomRoutingEngine::new(
            RandomRoutingProcessorConfig::new(self.strong.clone(), self.weak.clone())
                .with_strong_probability(self.strong_probability)?
                .with_rng_seed(self.rng_seed),
        )?;

        Ok(RandomRoutingProfile {
            router,
            strong_backend: native_target_backend(self.strong.clone())?,
            weak_backend: native_target_backend(self.weak.clone())?,
            stats: profile_stats_accumulator(),
        })
    }
}
```

This is deliberate. The macro handles the repetitive config surface. The profile author still owns
the runtime decisions: validation, backend construction, poller construction, stats handles, and
any profile-specific resources.

The central config loader should own only generic work:

- format detection and loading,
- YAML/JSON/TOML parsing,
- environment interpolation,
- shared endpoint inheritance,
- target resolution,
- dispatch by profile `type`,
- building requested profile runtimes.

Profile modules own everything profile-specific:

- which config fields exist,
- which fields are target references,
- cross-field validation,
- backend/runtime construction,
- profile-specific error semantics.

This is why `#[profile_target]` matters. The file can say `strong: strong`, but the runtime config
still receives a real `LlmTarget`. The profile author writes the natural runtime type, while the
macro handles the file-facing ID indirection.

The generated `profile_types!` registry is the narrow central list of supported profile config
types. It is acceptable because it is one line per profile and only dispatches to profile-owned
parsing/building. It should not grow into a central hand-written schema with duplicated fields.

## Complete Profile Shape

The Rust shape should make the full lifecycle visible in one profile module:

```rust
#[profile_config("random-routing")]
pub struct RandomRoutingProfileConfig {
    #[profile_target]
    pub strong: LlmTarget,
    #[profile_target]
    pub weak: LlmTarget,
    pub strong_probability: f64,
    pub rng_seed: Option<u64>,
}

impl ProfileConfig for RandomRoutingProfileConfig {
    type Runtime = RandomRoutingProfile;

    fn build(&self) -> Result<Self::Runtime> {
        validate_probability(self.strong_probability)?;

        Ok(RandomRoutingProfile {
            router: RandomRoutingEngine::from_config(self)?,
            strong_backend: native_target_backend(self.strong.clone())?,
            weak_backend: native_target_backend(self.weak.clone())?,
            stats: profile_stats_accumulator(),
        })
    }
}

pub struct RandomRoutingProfile {
    router: RandomRoutingEngine,
    strong_backend: TargetBackend,
    weak_backend: TargetBackend,
    stats: StatsAccumulator,
}

pub struct RandomRoutingProcessedRequest {
    request: ChatRequest,
    decision: RandomRoutingDecision,
}

impl ProfileProcessedRequest for RandomRoutingProcessedRequest {
    fn request(&self) -> &ChatRequest {
        &self.request
    }
}

#[async_trait]
impl ProfileHooks for RandomRoutingProfile {
    type ProcessedRequest = RandomRoutingProcessedRequest;

    async fn process(&self, request: ChatRequest) -> Result<Self::ProcessedRequest> {
        let (request, decision) = self.route_request(request)?;
        Ok(RandomRoutingProcessedRequest { request, decision })
    }

    async fn rprocess(
        &self,
        _processed: &Self::ProcessedRequest,
        response: ChatResponse,
    ) -> Result<ChatResponse> {
        Ok(response)
    }
}

#[async_trait]
impl Profile for RandomRoutingProfile {
    async fn run(&self, request: ChatRequest) -> Result<ChatResponse> {
        let started_at = Instant::now();
        let processed = self.process(request).await?;
        let backend = self.backend_for(&processed.decision)?;

        let response = match backend.call(processed.request().clone()).await {
            Ok(response) => response,
            Err(error) => {
                self.record_error(&processed.decision)?;
                return Err(error);
            }
        };

        self.record_success_and_usage(&processed.decision, &response, started_at)?;
        self.rprocess(&processed, response).await
    }
}
```

The equivalent Python shape should follow the same object boundary. Python may use decorators
instead of Rust macros, but it should not return to factory dictionaries or hidden bundles:

```python
ProcessedRequestT = TypeVar("ProcessedRequestT")


class Profile(Protocol):
    async def run(self, request: ChatRequest) -> ChatResponse: ...


class ProfileHooks(Protocol[ProcessedRequestT]):
    async def process(self, request: ChatRequest) -> ProcessedRequestT: ...

    async def rprocess(
        self,
        processed: ProcessedRequestT,
        response: ChatResponse,
    ) -> ChatResponse: ...


@profile_config("random-routing")
@dataclass
class RandomRoutingProfileConfig:
    strong: LlmTarget
    weak: LlmTarget
    strong_probability: float = 0.5
    rng_seed: int | None = None

    def build(self) -> "RandomRoutingProfile":
        validate_probability(self.strong_probability)
        return RandomRoutingProfile(
            router=RandomRoutingEngine(
                strong=self.strong,
                weak=self.weak,
                strong_probability=self.strong_probability,
                rng_seed=self.rng_seed,
            ),
            strong_backend=native_target_backend(self.strong),
            weak_backend=native_target_backend(self.weak),
            stats=profile_stats_accumulator(),
        )


@dataclass
class RandomRoutingProcessedRequest:
    request: ChatRequest
    decision: RandomRoutingDecision


class RandomRoutingProfile(Profile[RandomRoutingProcessedRequest]):
    def __init__(
        self,
        *,
        router: RandomRoutingEngine,
        strong_backend: TargetBackend,
        weak_backend: TargetBackend,
        stats: StatsAccumulator,
    ) -> None:
        self.router = router
        self.strong_backend = strong_backend
        self.weak_backend = weak_backend
        self.stats = stats

    async def process(self, request: ChatRequest) -> RandomRoutingProcessedRequest:
        request, decision = self._route_request(request)
        return RandomRoutingProcessedRequest(request=request, decision=decision)

    async def run(self, request: ChatRequest) -> ChatResponse:
        started_at = monotonic()
        processed = await self.process(request)
        backend = self._backend_for(processed.decision)

        try:
            response = await backend.call(processed.request)
        except SwitchyardError:
            self._record_error(processed.decision)
            raise

        self._record_success_and_usage(processed.decision, response, started_at)
        return await self.rprocess(processed, response)

    async def rprocess(
        self,
        processed: "RandomRoutingProcessedRequest",
        response: ChatResponse,
    ) -> ChatResponse:
        return response
```

The language-specific syntax can differ. The architecture should not. A profile config builds one
profile runtime, and the profile runtime owns the visible request lifecycle.

## How A Request Flows In Existing Profiles

### Passthrough

Passthrough has no routing state, so its `run()` can be the straightforward lifecycle:

```text
process(request)
    |
    v
PassthroughProcessedRequest { request }
    |
    v
single target backend call
    |
    v
record success, error, latency, and usage
    |
    v
rprocess(processed, response)
```

This is the simplest profile and should remain the reference for a one-target policy.

### Random Routing

Random routing selects between configured targets. Its request-side decision belongs in its own
processed-request type:

```text
process(request)
    |
    +--> rewritten request model
    |
    +--> RandomRoutingDecision
             |
             v
        selected backend call
             |
             v
        stats for selected decision
```

The key is that `RandomRoutingProcessedRequest` is not a generic metadata wrapper. It is the
random-routing profile's own type. If another profile makes a different kind of request-side
decision, it gets a different processed-request type.

### Latency Service

Latency-service routing owns a health cache and chooses among multiple targets:

```text
health cache
    |
    v
process(request) selects initial target candidate
    |
    v
call selected backend
    |
    +--> success: record usage and return
    |
    +--> failure: record error, exclude target, retry another target
```

The hot path reads cached health. Health polling is explicit through `poll_once()` or external
health injection, which keeps serving free of accidental latency-service network calls.
`LatencyServiceProcessedRequest` may contain the prepared request and the initial target selection,
but the full retry loop still belongs in `run()` because retries update per-call exclusion state as
backend calls fail.

## Why This Helps Contributions

V2 makes the contribution unit match the feature unit. A contributor adding a routing policy should
be able to open one profile module and answer:

- What does the config look like?
- What gets validated?
- How does request mutation work?
- Which backend is called?
- What happens on backend error?
- What stats are recorded?
- Can this profile be used as middleware-only?
- What tests define the behavior?

That is a smaller review surface than a graph of processors, bundles, factories, and table entries.

It also makes tests more direct. Instead of testing whether factory glue assembled the right
sequence of objects, tests can call:

```rust
let processed = profile.process(request).await?;
profile.run(request).await?;
profile.rprocess(&processed, response).await?;
```

This encourages adversarial tests around actual behavior:

- invalid config is rejected,
- request models are rewritten correctly,
- selected target state does not leak across concurrent calls,
- backend failures record stats and propagate errors,
- retries do not hit the same failed target when alternatives remain,
- middleware-only hooks do not accidentally call a backend.

## Adding A New Profile

Start with the smallest profile-owned implementation.

1. Create `src/profiles/<name>.rs` or `src/profiles/<name>/mod.rs`.
2. Define `<Name>ProfileConfig` with `#[profile_config("<serialized-type>")]`.
3. Mark config fields that refer to targets with `#[profile_target]`.
4. Implement the `ProfileConfig` build contract for the config type.
5. Put profile-specific validation near `build()`.
6. Build the runtime from fully resolved targets and existing backend helpers.
7. Define `<Name>ProcessedRequest` for the output of `process()`.
8. Include the prepared `ChatRequest` and any profile-specific request-side decision in that type.
9. Implement `ProfileHooks` and `Profile` for `<Name>Profile`.
10. Keep retry loops, backend calls, and mutable per-call state local to `run()`.
11. Add tests for config validation, hook behavior, full `run()` behavior, stats, errors, and
   concurrency if the profile has mutable state.
12. Add the config type to the generated profile type list.
13. Export the config and runtime from `src/profiles/mod.rs` and `src/lib.rs`.

For Python-defined profiles, the same shape applies:

1. Put the profile config and runtime in a profile module, not in a global factory directory.
2. Use the shared profile registration path so config loading can discover it.
3. Keep validation and target resolution close to the profile.
4. Define a typed processed-request object for the profile's `process()` output.
5. Implement the same `process`, `run`, and `rprocess(processed, response)` lifecycle.
6. Make config fields explicit and typed rather than accepting arbitrary factory dictionaries.
7. Delete old factory or bundle glue instead of wrapping it indefinitely.

The first version should be boring. Add private helpers only when they make the profile easier to
read. Do not add a new public abstraction because two profiles happen to share five lines.

## What Not To Add Back

Avoid reintroducing old shapes under new names:

- A "profile graph" that must be built before a profile can run.
- A "profile bundle" that only wraps request and response hooks.
- A hidden stats layer that records without being visible in the profile's `run()`.
- A context object whose main job is carrying state between self-owned stages.
- A generic "profile metadata" object used as a new untyped side channel.
- A central factory registry that every new profile has to touch manually.

If a new helper is needed, prefer one of these forms:

- a private function inside one profile module,
- a small shared function with no lifecycle ownership,
- a reusable engine that performs pure policy logic,
- a backend adapter that only owns provider call mechanics.

## Migration Boundary

`switchyard-components-v2` is intentionally separate from the older
`switchyard-components` crate while the shape settles. During migration, v2 can reuse proven
engines and backend code from the older crate, but new profile code should not adopt the older
public shape.

The same applies to Python. The Python package can keep working while profile v2 lands, but the
target state is not "Rust has profiles while Python keeps factories." The target state is one
profile model, with Python using bindings or Python-defined profiles that obey the same lifecycle.

The practical migration stance is:

```text
preserve behavior
do not preserve unnecessary object structure
do not preserve backward compatibility by default
```

This means a v2 profile can reuse a mature random-routing engine, stats accumulator, or native
backend implementation while still presenting a flatter runtime surface.

## Staleness Risks

This document will need updates when:

- the temporary `ProfileBackend` adapter is replaced by native v2 backend ownership,
- config loading for v2 profiles becomes the primary server path,
- Python bindings expose v2 profiles directly or Python-defined profiles adopt the same lifecycle,
- stats recording moves from explicit profile calls to a deliberate helper with an equally visible
  lifecycle,
- new profiles introduce a legitimate pattern that this document does not cover.

## PyO3 Binding Caveat

The associated `ProcessedRequest` type is deliberately good Rust, but it is not directly
object-safe across PyO3. Generic Python bindings should therefore split the surface the same way the
Rust traits do.

The serving path is easy to expose generically: Python can hold a `Profile` wrapper backed by
`Arc<dyn Profile>` and call `run(request)` for any Rust profile returned by config loading.
That is the right default for servers, launchers, and generic profile tables.

The hook path needs one more layer. Python cannot call a fully generic `dyn ProfileHooks` without
erasing the associated processed-request type. Bindings should expose either:

- an opaque `ProcessedRequest` Python handle backed by an erased Rust wrapper, with a `.request`
  accessor for caller-owned model calls, or
- macro-generated concrete Python classes for each registered profile and its concrete
  `<Name>ProcessedRequest` type.

Both approaches keep the core invariant: the processed-request value is typed and profile-owned in
Rust. Python should not receive a generic metadata dictionary, and `switchyard-components-v2`
should not depend on PyO3. PyO3 registration belongs in `switchyard-py`, ideally generated from the
same profile registry that lists the supported profile config types.
