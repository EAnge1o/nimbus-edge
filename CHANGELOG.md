# Changelog

All notable changes to nimbus-edge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org).

## [0.2.0] — 2026-10-30

### Added

- `stream_options: {"include_usage": true}` (chat, streaming): one final
  `choices: []` usage chunk before `data: [DONE]`, OpenAI-compatible;
  verified against the official Python SDK.
- Opt-in CORS (`--cors-origins "*" | origin,list`) with preflight handling
  so browser clients (Open WebUI, LibreChat) can connect directly; default
  remains off (see SECURITY.md).
- `GET /` self-describing root endpoint.
- Request logging via `tracing` (initialized in `serve`; `RUST_LOG`
  respected, `info` default).
- `scripts/check.ps1` / `check.sh`: the full CI gate as one local command.
- `docs/demo.svg` + `docs/make_demo_svg.py`: README screenshot rendered
  from real command output (house style of the Nimbus family).
- Real-model verification: gpt2-124M and SmolLM2-135M-Instruct converted
  from HuggingFace, served end-to-end, benchmarked
  ([benchmarks/](benchmarks/README.md)); gpt2 greedy output is
  token-identical to the Python engine.

### Fixed (engine, nimbus-infer-rs — separate repo)

- The RoPE frequency table is now built in f64 and stored f32, matching the
  Python engine bit-for-bit; a pure-f32 pipeline drifted ~1 ulp per entry,
  which accumulated across layers on deep models.

## [0.1.0] — 2026-10-30

First release.

### Added

- OpenAI-compatible HTTP server (`nimbus-edge serve`): `/healthz`,
  `/v1/models`, `/v1/chat/completions`, `/v1/completions` with streaming
  (SSE) and non-streaming responses, bearer-token auth, OpenAI-style error
  bodies, and strict rejection of unsupported parameters.
- Bounded request queue with `max_queue_size` (429 when full) and
  `max_concurrency`; client-disconnect and explicit cancellation stop
  generations between decode steps and remove queued entries.
- Per-request KV isolation by construction (no shared forward pass; not
  continuous batching, and documented as such).
- Prompt-prefix KV cache (opt-out) with a byte-identical-output guarantee,
  hit/miss counters on `/healthz`.
- Model registry: `models import|list|inspect|remove` with import-time
  validation, per-file SHA-256 checksums, strict metadata, name/path-traversal
  defenses, and a header-only safetensors inspector.
- Chat templates: ChatML (SmolLM2-compatible) with snapshot tests; base
  models without a template refuse `/v1/chat/completions` with a clear
  error.
- `nimbus-edge status` (local build/host info, server probe) and
  `nimbus-edge bench` (prefill/decode/TTFT/peak-RSS, 3-run median, JSON +
  human output).
- Windows x64 and Linux x64 release artifacts (zip/tar.gz + SHA256) with a
  bundled tiny model, CI matrix, and an artifact smoke test.
- Integration, scheduler, black-box and snapshot test suites; all tests run
  from a generated random-weight checkpoint with no network access.

### Engine (nimbus-infer-rs, separate repository)

Minimal, backward-compatible extensions this project depends on, kept out of
this codebase:

- `Engine::generate_with` — streaming callback, cooperative cancellation,
  prefill reuse; `generate_ids` unchanged.
- `Engine::prefill` + `GenerationOutput.cache` for prefix caching.
- `KVCache: Clone` and `KVCache::truncate`.
- safetensors `data_offsets` bounds check (corrupt files error, not panic).
