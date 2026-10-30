# Nimbus Edge

[![ci](https://github.com/EAnge1o/nimbus-edge/actions/workflows/ci.yml/badge.svg)](https://github.com/EAnge1o/nimbus-edge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Nimbus Edge** is a CPU-first, local-first, single-binary LLM inference
server for Windows and Linux. It wraps the
[nimbus-infer-rs](https://github.com/EAnge1o/nimbus-infer-rs) engine — a
from-scratch, std-only Rust inference core (GPT-2 + LLaMA architectures,
byte-level BPE, KV cache, hand-written AVX2/FMA kernels, int8/int4
quantization) — with an Ollama-style command line and an OpenAI-compatible
HTTP API. No GPU, no cloud, no daemon framework: download one binary, run
one command, `POST /v1/chat/completions`.

Nimbus Edge never vendors or forks the inference core; it consumes it as a
crate. See [ARCHITECTURE.md](ARCHITECTURE.md) for the boundary.

![nimbus-edge serving SmolLM2-135M-Instruct: import, serve, and a real chat
completion rendered from actual command output](docs/demo.svg)

*The screenshot is rendered from real command output (`docs/make_demo_svg.py`),
not staged — SmolLM2-135M-Instruct answering over the local HTTP API.*

---

## 60-second quick start

```bash
# 1. Get the binary: download a release for your platform (see "Install"),
#    or build from source:
cargo build --release
# The binary is target/release/nimbus-edge(.exe).

# 2. Import a model (any nimbus-infer checkpoint). Source builds use the
#    fixture from the sibling core checkout; release artifacts bundle it as
#    models/tiny-gpt2:
nimbus-edge models import     --path ../nimbus-infer-rs/tests/fixtures/tiny-gpt2     --name tiny-gpt2 --chat-template chatml

# 3. Serve it:
nimbus-edge serve --model tiny-gpt2
# nimbus-edge v0.1.0 | model tiny-gpt2 (gpt2, fp32, weights 3.2 MB, ...) | context 256 tokens
# listening on http://127.0.0.1:11434 | max_concurrency 1 | max_queue_size 32 | auth off

# 4. Talk to it:
curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"tiny-gpt2","messages":[{"role":"user","content":"hello"}]}'
```

`/v1/chat/completions` needs a model with a chat template; `/v1/completions`
works with any checkpoint. The bundled `tiny-gpt2` is a **random-weight
fixture for smoke-testing the API** — its text output is gibberish by
design. Real usable models are listed under [Models](#models).

## Install

**Option A — prebuilt binaries (recommended).** Every release ships:

- `nimbus-edge-vX.Y.Z-windows-x64.zip` + `.sha256`
- `nimbus-edge-vX.Y.Z-linux-x64.tar.gz` + `.sha256`

Each artifact contains the binary, `README.md`, `LICENSE`, and a bundled
`models/tiny-gpt2` fixture so it works offline. Verify the SHA256 file
before running. `cargo install` is still available for developers, but it is
not the primary distribution.

**Option B — from source.** nimbus-edge builds against a *sibling* checkout
of the inference core (a path dependency — nothing is vendored):

```bash
git clone https://github.com/EAnge1o/nimbus-edge.git
cd nimbus-edge
git clone --depth 1 https://github.com/EAnge1o/nimbus-infer-rs.git ../nimbus-infer-rs
cargo build --release
cargo test --workspace --features nimbus-edge-core/test-support
```

Requires a stable Rust toolchain (1.75+). Windows builds on the MSVC target
in CI; windows-gnu works too.

## Models

Nimbus Edge runs **nimbus-infer checkpoints**: a directory with

```
config.json
weights.safetensors        (or model.safetensors)
tokenizer/vocab.json       (optional but required for text I/O)
tokenizer/merges.txt
```

That format covers models converted with the Python engine's
`python -m nimbus_infer convert`, e.g. GPT-2 family and LLaMA-family
architectures (RoPE/RMSNorm/SwiGLU/GQA included).

Verified end-to-end through the HTTP API on this machine (2026-10-30,
converted from HuggingFace via the Python engine):

| model | arch | chat template | verified |
|---|---|---|---|
| gpt2-124M | gpt2 | none (base model) | real completion, greedy parity vs the Python engine (32/32 tokens) |
| SmolLM2-135M-Instruct | llama (GQA 9/3, RoPE) | chatml | real chat completion — "The capital of France is Paris." |
| tiny-gpt2 (bundled fixture) | gpt2 | chatml (optional) | random weights; exercises the full API in tests and smoke |

Measured throughput for these models: [benchmarks/README.md](benchmarks/README.md)
(gpt2-124M fp32 decode ≈ 40 tok/s on an i7-12700H, int8 ≈ 56 tok/s — same
engine, HTTP layer adds no per-token cost).

GGUF is **not** supported in v1 — see the extension point note in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Import / manage

```bash
nimbus-edge models import --path <checkpoint-dir> --name <name> \
    [--chat-template none|chatml] [--quant fp32] [--overwrite]
nimbus-edge models list [--json]
nimbus-edge models inspect <name>     # metadata + config + tensor list, no weights loaded
nimbus-edge models remove <name>
```

- Import validates the checkpoint (config parse, weights file, tokenizer
  completeness) and records **per-file SHA-256 checksums** in
  `edge-meta.json`.
- Model names are restricted to `[A-Za-z0-9._-]` (1–64 chars, starting
  alphanumeric) and every registry access is containment-checked against the
  registry root — no path traversal.
- The chat template choice lives in the model metadata. **Base models like
  GPT-2/distilgpt2 have no chat template**; import them with the default
  `none` and use `/v1/completions`. ChatML (`<|im_start|>`/`<|im_end|>`, the
  SmolLM2 format) is the built-in template for chat models.
- Default registry: `~/.nimbus-edge/models` (override: `--models-dir` or
  `NIMBUS_EDGE_MODELS_DIR`).

You can also serve a checkpoint directory directly without importing:
`nimbus-edge serve --model /path/to/checkpoint` (chat template `none`
unless an `edge-meta.json` sidecar sits in the directory).

## Serving

```bash
nimbus-edge serve --model <name-or-path> [--host 127.0.0.1] [--port 11434]
    [--quant fp32|int8|int4] [--max-concurrency 1] [--max-queue-size 32]
    [--auth-token <token>] [--threads N] [--prefix-cache/--no-prefix-cache]
```

- Listens on **127.0.0.1 by default**; pass `--host 0.0.0.0` deliberately if
  you want LAN exposure.
- Optional bearer auth: `--auth-token` or env `NIMBUS_EDGE_AUTH_TOKEN`.
  When set, `/v1/*` requires `Authorization: Bearer <token>`; `/healthz`
  stays open for liveness probes.
- `--threads N` pins the engine's internal kernel threads (default: all
  cores; decode GEMVs cap themselves at 4 — see the core repo's notes on
  memory-bandwidth-bound decode).
- `--cors-origins "*"` (or a comma-separated list) enables CORS for browser
  clients — Open WebUI, LibreChat and friends work against it like against
  Ollama. Off by default; see [SECURITY.md](SECURITY.md).

## API: the compatible subset

This server implements a **documented subset** of the OpenAI API — not full
compatibility, and it does not pretend otherwise. Unknown request fields and
known-but-unimplemented fields return a **400 with an OpenAI-style error
body**, never a silent ignore.

| endpoint | status |
|---|---|
| `GET /healthz` | implemented (server-local shape: model, queue, cache stats) |
| `GET /v1/models` | implemented (lists the served model) |
| `POST /v1/chat/completions` | implemented (streaming + non-streaming) |
| `POST /v1/completions` | implemented (streaming + non-streaming) |

| parameter | support |
|---|---|
| `model` | required; must match the served model (404 otherwise) |
| `messages` | system/user/assistant, string content |
| `prompt` | string, or array with exactly one element |
| `temperature` | `[0, 2]`; `0` = greedy |
| `top_p` | `(0, 1]` |
| `max_tokens` | `1..=16384`; clamped to the model context if larger |
| `stop` | string or up to 4 strings, matched on decoded text |
| `seed` | non-negative integer (xoshiro256++) |
| `stream` | `true` → SSE (`data: {...}\n\n`, terminal `data: [DONE]`) |
| `user` | accepted, ignored (caller metadata) |
| `n`, `logprobs`, `tools`, `response_format`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `stream_options`, `echo`, `suffix`, `max_completion_tokens`, … | **400 `unsupported_parameter`** |

Error shape: `{"error": {"message", "type", "param", "code"}}` with
`invalid_request_error` (400), `invalid_request_error/model_not_found` (404),
`rate_limit_error` (429, with `Retry-After: 1`), `api_error` (500).
Mid-stream failures are emitted as `data: {"error": {...}}` followed by
`data: [DONE]` so SSE clients see them.

Known limitations of the subset (also surfaced as errors, not guessed):

- content arrays (multimodal messages) and roles outside
  system/user/assistant are rejected;
- streaming `usage` requires the client to opt in with
  `stream_options.include_usage` (as in the OpenAI API);
- `finish_reason` is `stop` (stop string matched), `length` (token budget),
  or — a local extension, documented here — `cancelled` (generation was
  cancelled mid-flight);
- token-level `stop` ids are not exposed; stop sequences are matched on
  decoded text.

### OpenAI SDK examples

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="not-needed")

# non-streaming
r = client.chat.completions.create(
    model="tiny-gpt2",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=32,
)
print(r.choices[0].message.content)

# streaming
for chunk in client.chat.completions.create(
    model="tiny-gpt2",
    messages=[{"role": "user", "content": "hello"}],
    max_tokens=32,
    stream=True,
):
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

Node:

```js
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://127.0.0.1:11434/v1", apiKey: "not-needed" });

const stream = await client.chat.completions.create({
  model: "tiny-gpt2",
  messages: [{ role: "user", content: "hello" }],
  max_tokens: 32,
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0].delta?.content ?? "");
```

## Queue, cancellation, concurrency

- Requests wait in a **bounded FIFO queue** (`--max-queue-size`, default 32)
  and at most `--max-concurrency` (default 1) generate at once. A full queue
  rejects immediately with **429** (`queue_full`, `Retry-After: 1`) — it
  never grows without bound.
- Each running request owns an **independent engine-side KV cache**; there
  is no shared forward pass, so concurrent requests cannot contaminate each
  other's state (this is a deliberately safe bounded-concurrency queue, not
  continuous batching — see ARCHITECTURE.md).
- **Cancellation is cooperative and complete**: if a client disconnects
  (stream or non-stream) or the request is cancelled, queued entries are
  removed without running, and a running generation stops at the next decode
  step instead of burning CPU. All of this is covered by integration tests.

## Caching

Two in-process caches, both opt-out-able:

- **Loaded model** stays resident for the process lifetime (weights,
  tokenizer).
- **Prefix KV cache** (on by default, `--prefix-cache-entries 2`): after a
  generation, the prompt's KV state is kept (truncated to the prompt, so
  entries are pure prompt-prefix states) and later requests whose prompt
  extends a cached prompt token-for-token reuse it instead of re-prefilling.
  This is the standard multi-turn chat pattern. Hits are masked behind a
  correctness invariant — output must be byte-identical to a fresh prefill,
  and tests enforce it. Hit/miss counters are exposed on `/healthz`
  (`prefix_cache`). Each entry costs one full-context KV allocation
  (~KV-size per sequence; shown in the serve banner), bounded by
  `--prefix-cache-entries`; disable with `--no-prefix-cache`.

Nimbus Edge deliberately does **not** depend on nimbus-kv (the multi-node
Raft store): a local single-process server has no cross-node cache to share.

## Diagnostics and benchmarks

```bash
nimbus-edge status                 # build info, host facts, registry summary
nimbus-edge status --url http://127.0.0.1:11434   # probe a running server
nimbus-edge bench --model <name-or-path> [--prompt TEXT] [--max-new-tokens 64] \
    [--runs 3] [--quant fp32|int8|int4] [--format human|json] [--json-out FILE]
```

`bench` reports prefill tok/s, decode tok/s, first-token latency, process
peak RSS, and the full host/build record — 3 runs, median, warmup included.
All numbers are measured; nothing is estimated. Reproduction instructions
and the honest status of cross-engine comparisons live in
[benchmarks/](benchmarks/README.md).

## Testing

```bash
cargo test --workspace --features nimbus-edge-core/test-support   # tests only
scripts/check.ps1   # or scripts/check.sh — the full CI gate locally:
                    # fmt + clippy -D warnings + tests + release build
```

covers: request parsing and error responses, registry import/list/inspect/
remove and path-traversal defenses, chat-template snapshots, non-streaming
and SSE completions, client-disconnect cancellation, queue-full 429,
multi-concurrent KV isolation, model-not-found and corrupted-file handling,
scheduler unit tests with a stub backend, and a black-box test that drives
the real binary end to end. All tests use a random-weight tiny checkpoint
generated in-process — no network, no model downloads.

## Current limitations (honest list)

- **GGUF is not supported** — nimbus-infer checkpoints only.
- **No continuous batching** — bounded concurrency with per-request KV, by
  design and documented as such.
- The compatible API subset is narrow by intent (see table above).
- Base models without a chat template cannot serve `/v1/chat/completions`.
- Prefix cache reuse requires the new prompt to extend a cached one at a
  token boundary; a mid-token text boundary misses (correctly) and prefills
  fresh.
- Single process, single node; no multi-tenancy, no web UI, no
  distributed inference, no training.
- The bundled `tiny-gpt2` has random weights: it validates the API, it does
  not produce meaningful text.

## The Nimbus family

Nimbus Edge is the serving layer of a from-scratch LLM stack:

- [nimbus-infer](https://github.com/EAnge1o/nimbus-infer) — the original
  NumPy-only engine (tokenizer, KV cache, quantization, from-scratch
  transformer), also the checkpoint converter for HuggingFace models.
- [nimbus-infer-rs](https://github.com/EAnge1o/nimbus-infer-rs) — the
  std-only Rust rewrite of the same engine: AVX2/FMA kernels, int8/int4,
  checkpoint-compatible with the Python version.
- **nimbus-edge** (this repo) — the server: OpenAI-compatible HTTP, bounded
  scheduling, model registry, one-binary distribution.

## License

MIT — see [LICENSE](LICENSE). The inference core,
[nimbus-infer-rs](https://github.com/EAnge1o/nimbus-infer-rs), is a separate
MIT project.
