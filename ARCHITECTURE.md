# Architecture

## Crate boundaries

```
                 ┌────────────────────────────┐
                 │  nimbus-infer-rs (independent repo, std-only)
                 │  Model · KVCache · Engine  │   numeric kernel: untouched
                 └──────────────▲─────────────┘
                                │ path dependency (../nimbus-infer-rs)
┌───────────────────────────────┴────────────────────────────────────────┐
│ nimbus-edge                                                            │
│                                                                        │
│  ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ nimbus-edge-cli  │──▶│ nimbus-edge-server│──▶│ nimbus-edge-core  │  │
│  │ serve/models/    │   │ axum routes · SSE │   │ openai DTOs       │  │
│  │ status/bench     │   │ auth · errors     │   │ chat templates    │  │
│  └──────────────────┘   └───────────────────┘   │ registry          │  │
│          HTTP-free          HTTP-only           │ scheduler         │  │
│                                                 │ NimbusBackend     │  │
│                                                 └─────────┬─────────┘  │
└───────────────────────────────────────────────────────────┼────────────┘
                                                            │
                                                 nimbus_infer_rs::Engine
```

- **nimbus-edge-core** — everything HTTP-agnostic: the OpenAI-compatible
  request DTOs with strict validation, chat templates, the model registry
  (import/inspect/remove, checksums, path safety), the scheduler (bounded
  queue + bounded concurrency + cancellation) and the `Backend` trait with
  its nimbus-infer-rs adapter (`NimbusBackend`, including the prefix cache).
- **nimbus-edge-server** — HTTP only: routes, SSE shaping, bearer auth, the
  error-to-status mapping. It never touches the engine; it speaks to the
  scheduler's event channel.
- **nimbus-edge-cli** — clap front end, model registry commands, bench, host
  facts (std-only FFI: `GetProcessMemoryInfo` / `/proc`).
- **nimbus-edge-tests** — integration tests over the real server stack.

The inference core stays a separate repo with zero dependencies. Nimbus Edge
adds only four direct dependencies to its side of the boundary (serde,
serde_json, sha2, tokio — plus axum/clap/reqwest in the outer layers); the
numeric kernel is never forked or reimplemented here.

## Request lifecycle

```
POST /v1/chat/completions
  │ strict DTO parse (deny_unknown_fields) → known-unsupported-field check
  │ template render (or 400 no_chat_template) → param validation
  ▼
Scheduler::submit ──queue full?──▶ 429 rate_limit_error (Retry-After)
  │ bounded VecDeque; dead entries swept on submit/dispatch
  ▼ dispatcher: inflight < max_concurrency?
spawn_blocking ──── one thread per running request
  │ NimbusBackend::generate
  │   encode(prompt) → prefix-cache lookup (token-prefix match)
  │   Engine::generate_with { on_token, is_cancelled, prefill }
  │     │ per token: incremental UTF-8 decode → stop-string scan → delta
  │     └ cancelled? → stop between steps, return partial
  ▼ mpsc channel (bounded, backpressure via blocking_send)
handler / SSE body ──▶ OpenAI JSON chunks or chat.completion object
```

**Cancellation correctness.** A request owns two receivers: the scheduler's
event channel and a oneshot "liveness" receiver. Both live inside the SSE
body (or the handler future, for non-streaming). When the client goes away,
hyper drops the body/future, both receivers drop, and two things happen at
once: queued jobs are swept (`tx.is_closed()`), and the running generation's
`is_cancelled` closure fires (`oneshot::Sender::is_closed()`), stopping the
engine between decode steps. A slow consumer is backpressured by the bounded
channel instead of buffering unboundedly.

**Stop strings** are matched on the accumulated decoded text every token, so
they work regardless of how the vocabulary encodes special markers; the
output is truncated at the earliest cut and `finish_reason` becomes `stop`.

## Concurrency and KV isolation

`Engine::generate_with(&self, ...)` is an immutable borrow: the engine holds
no mutable global state, and every call builds its own `KVCache`. Concurrent
requests therefore run on independent threads with independent KV state —
isolation is structural, not discipline. The scheduler caps concurrency
(default 1) because decode is memory-bandwidth bound and the engine already
parallelizes internally (see the core repo's threading notes).

This is explicitly **not** continuous batching: requests never share a
forward pass. When/if the core grows a batched step API with per-sequence
KV, the `Backend` trait is the only seam that changes.

## Prefix cache

After a generation, the engine returns its final KV cache
(`GenerationOutput.cache`). The backend truncates it to the prompt length
(`KVCache::truncate`) and stores it as a `PrefillState` entry keyed by the
prompt's token sequence. A later request whose prompt tokenizes with that
sequence as a strict token prefix (multi-turn chat) continues from the
cached prefill. Two invariants, both test-enforced:

1. a cache hit produces byte-identical output to a fresh prefill;
2. reuse requires a *token* prefix that leaves at least one token to
   prefill — a full-cover cache can never be misused for a different
   continuation.

Entries hold full-context KV allocations; the count is bounded
(`--prefix-cache-entries`, default 2) and oversized models can be excluded.

## Model registry

`models import` validates before writing (config parses with the engine's
own parser, weights file present, tokenizer complete), copies files while
hashing them (SHA-256 per file), and records strict metadata
(`edge-meta.json`, `deny_unknown_fields`). Model names are the traversal
defense: `[A-Za-z0-9._-]` only, and every registry path is canonicalized and
containment-checked against the registry root. `models inspect` reads only
the safetensors *header* (u64 length + JSON), never the weights.

## Extension points

- **Chat templates**: add a variant to `ChatTemplate` in core plus snapshot
  tests; select it per model via `--chat-template`/metadata.
- **Model formats** (GGUF etc.): the registry validates the
  `nimbus-checkpoint-v1` layout and `NimbusBackend::load` is the single
  construction point for engines. A new format becomes a new loader behind
  the same `Backend` trait plus a `format` tag in the metadata — no
  scheduler or HTTP changes.
- **Batching**: a batched step API in the core slots into `Backend`
  without touching the HTTP layer.
