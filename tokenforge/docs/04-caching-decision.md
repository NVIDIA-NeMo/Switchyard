# The caching gap — build, source, or drop

**Context.** The M360 architecture diagram shows a "Cache & shaping — prefix,
semantic" box in the control-plane layer. Switchyard satisfies none of it: no
semantic cache, no response cache, no embeddings, no vector store. `cached_tokens`
is pass-through of the *provider's* prompt cache, and `session_cache.py` is an LRU
pin store for routing affinity.

The pushback was: don't just flag it, resolve it. Here is the resolution.

> **Research caveat.** The competitive/vendor scan for this decision was cut short
> by a session limit. Version numbers, maintenance status, and current vendor
> feature sets below are **from prior knowledge and must be re-verified** before
> any of it reaches customer material. The *architecture* and *monetization*
> reasoning does not depend on that scan and stands on its own.

## 1. The core insight: these are two different products

"Prefix, semantic" reads like one box. It is two, at two different layers, with
different owners, different risk profiles, and different answers.

| | Prefix / KV cache | Semantic cache |
| --- | --- | --- |
| What it reuses | attention KV state for a shared token prefix | a previous *response* for a similar prompt |
| Correctness | **exact** — same tokens, same math | **approximate** — returns a different question's answer |
| Where it belongs | **the inference server** | a gateway / control plane |
| Who provides it | vLLM automatic prefix caching, TensorRT-LLM, SGLang RadixAttention, NVIDIA Dynamo KV-aware routing and KV offload | you, or a cache vendor |
| Switchyard's role | already reports the result as `cached_tokens` | none |
| Decision | **source — do not build** | **defer — see §3** |

Collapsing them into one diagram box is the actual error. Fix the diagram first.

## 2. Prefix caching — source it, and monetize the observation

Prefix/KV caching cannot be implemented at a gateway. The gateway does not hold
GPU memory or attention state. It happens in the inference server, and the NVIDIA
stack the product co-sells with already does it — vLLM automatic prefix caching,
TensorRT-LLM, and Dynamo's KV-aware routing and KV-cache offload are all in this
space. **Re-verify current capabilities before quoting them.**

TokenForge's job is therefore not to cache, but to **see and price the caching
that already happens**. That is already implemented:

- `ai.tokens.cached_read` and `ai.tokens.cache_write` are meters in the catalog.
- The rate card prices cached reads separately (`TokenRate.cached`) and cache
  writes separately (`cache_write`), because the supplier does.
- `cost_details.{base_input, cached_input, cache_write}` flows from the intake
  record into the margin calculation.

This is a stronger position than building a cache, and it is defensible in front
of NVIDIA: *we do not duplicate your inference-layer optimizations, we make them
appear on an invoice.* A customer who turns on prefix caching sees their margin
per token improve in the TokenForge dashboard. That is the FinOps product.

**Diagram fix:** rename the box to **"Cache economics — observe & price"** and
place prefix caching in the inference layer where it actually lives.

## 3. Semantic caching — defer, and here is why that's the right call

Semantic caching is genuinely a gateway concern, and commercial gateways
(Portkey, Kong AI Gateway, LiteLLM, Cloudflare AI Gateway, Helicone and others)
ship it, so there is a table-stakes argument. **Re-verify who ships what.**

Building it is not hard: an embedding model, a vector store (Redis or pgvector),
a similarity threshold, a TTL. A competent team ships a working version in weeks.

**The reason to defer is not difficulty. It is that semantic caching is
structurally hostile to this product's two lead verticals and to its own revenue
model.**

### 3.1 It breaks the revenue model

A semantic cache hit serves the customer a valuable answer while consuming
**zero** upstream tokens. Under per-token pricing, revenue for that request is
zero and margin is undefined. The better the cache performs, the more revenue it
destroys — while the customer's *perceived* value is unchanged or better.

This is not a bug to work around; it is a pricing question that must be answered
*before* the feature is built:

| Option | Effect |
| --- | --- |
| Bill nothing on a hit | cache success directly cannibalises revenue |
| Bill the full token price as if uncached | ~100% margin, but it is billing for tokens never consumed — indefensible to an auditor and fatal in FSI |
| **Bill a distinct `ai.cache.hit` SKU at a discount** | honest, defensible, and margin-positive |

The third is the answer, and it is a **product opportunity, not a workaround**:
a cache hit becomes a priced governance service — "we served this from cache, you
paid 15% of token price, you saved 85%" — which is precisely the FinOps value
proposition, itemised on the invoice. Add `ai.cache.hit` to the meter catalog and
a `cache_hit_price_usd` to the rate card when the feature is built.

### 3.2 It is a correctness risk in the lead verticals

A semantic cache deliberately returns *a different question's answer* when the
questions are similar enough. The two lead verticals are **FSI** and **sovereign
AI**. "Similar enough" is not an acceptable standard for a regulated financial
answer or a government one, and the immutable audit trail — a core
differentiator — becomes actively misleading if the audit record shows a prompt
that never produced the response served.

If it ships, it must be **opt-in per route, off by default, and never enabled on
a contract with an audit obligation.**

### 3.3 It is a cross-tenant data-leakage risk

A shared cache across tenants is a confidentiality breach waiting to happen: one
tenant's response served to another. Correct mitigation is a **tenant-scoped
cache namespace** — but tenant scoping largely destroys the hit rate that
justified the cache, since the volume for cross-prompt similarity within one
tenant is far lower. Prompts also routinely contain PII, so cache keys and stored
responses inherit the full data-residency and retention regime.

Vendors reporting high hit rates are typically measuring single-tenant,
high-repetition workloads. **Do not quote third-party hit rates for a
multi-tenant deployment.**

## 4. Decision

| Item | Decision | Phase |
| --- | --- | --- |
| Prefix / KV caching | **Source** from the inference layer. Meter and price the result. Already implemented. | A (done) |
| Cache economics reporting | Surface cached-read savings and margin uplift per tenant | A |
| `ai.cache.hit` SKU + cache-hit pricing model | Design now, so the pricing answer exists before the feature | B |
| Semantic cache | **Defer to Phase C.** Opt-in per route, tenant-scoped namespace, off by default, prohibited on audit-obligated contracts | C |
| Response cache (exact-match) | Consider ahead of semantic — exact-match has none of the correctness risk and much of the cost benefit | C |
| Prompt shaping (`max_tokens` clamp, context trimming) | **Build** — already partly implemented in the policy processor | B |

**Exact-match response caching deserves a closer look than it usually gets.** It
carries no correctness risk whatsoever — identical input, identical output — and
in agentic workloads with retry loops and idempotent tool calls, repetition rates
are high. It captures a meaningful share of the benefit with none of the
liability. If a cache ships in Phase C, ship exact-match first.

## 5. What to say to a customer

> "Prefix and KV caching happen in the inference layer — vLLM, TensorRT-LLM,
> Dynamo — and they do it better than a gateway could. TokenForge meters and
> prices what they save, so cache efficiency shows up as margin on your invoice
> rather than as an unexplained cost variance. Semantic caching is on the roadmap
> as an opt-in, tenant-scoped capability; we do not enable approximate answer
> reuse by default on regulated workloads, and we will not quote you a hit rate
> measured on someone else's single-tenant traffic."

That is a stronger answer than claiming a semantic cache, and it survives a
solution-architect review — which, per the GTM playbook, is a gate that must be
airtight.
