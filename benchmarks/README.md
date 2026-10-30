# Benchmarks

Every number published here comes from a measurement recorded in
`results/*.json` by `nimbus-edge bench` (protocol below). Comparisons that
have not been run are marked **pending** — no estimates are presented as
measurements.

## Protocol (what `nimbus-edge bench` does)

- fixed prompt, tokenized with the model's own tokenizer
- greedy decoding (`temperature=0`, fixed xoshiro256++ seed) — reproducible
- 1 warmup run (not measured), then N timed runs (default 3)
- reported per run: prefill time, decode time, time-to-first-token
- aggregated as the **median** over runs, never the best single run
- host facts (CPU model, total RAM, thread count, rustc version, target,
  profile) and process peak RSS are recorded from the machine, or `null`
  when the platform cannot provide them

Reproduce any result file:

```bash
cargo build --release
./target/release/nimbus-edge bench \
  --model <checkpoint-dir> \
  --prompt "$(python benchmarks/make_prompt.py 128)" \
  --max-new-tokens 64 --runs 3 --format json --json-out benchmarks/results/<name>.json
```

## Measured (2026-10-31, real trained checkpoints)

Host: Windows 11, 12th Gen Intel Core i7-12700H (20 threads), 15.7 GB RAM,
rustc 1.99.1, target x86_64-pc-windows-gnu, profile release, engine threads
unpinned. Protocol: 131 prompt tokens → 64 new tokens, greedy, 3 runs,
median. Checkpoints converted from HuggingFace with the Python engine
(`python -m nimbus_infer convert`), fp32 on disk, quantized at load time.

| model | quant | prefill tok/s | decode tok/s | ttft (s) | source |
|---|---|---:|---:|---:|---|
| gpt2-124M | fp32 | 272.9 | 37.3 | 0.480 | [results/gpt2-124M-fp32.json](results/gpt2-124M-fp32.json) |
| gpt2-124M | int8 | 246.7 | 47.0 | 0.531 | [results/gpt2-124M-int8.json](results/gpt2-124M-int8.json) |
| gpt2-124M | int4 | 220.9 | 47.0 | 0.593 | [results/gpt2-124M-int4.json](results/gpt2-124M-int4.json) |
| distilgpt2-82M | fp32 | 552.7 | 60.6 | 0.237 | [results/distilgpt2-82M-fp32.json](results/distilgpt2-82M-fp32.json) |
| distilgpt2-82M | int8 | 515.7 | 91.7 | 0.254 | [results/distilgpt2-82M-int8.json](results/distilgpt2-82M-int8.json) |
| distilgpt2-82M | int4 | 443.6 | 81.6 | 0.295 | [results/distilgpt2-82M-int4.json](results/distilgpt2-82M-int4.json) |
| SmolLM2-135M-Instruct | fp32 | 107.9 | 27.1 | 1.214 | [results/smollm2-135M-fp32.json](results/smollm2-135M-fp32.json) |

Reading the numbers honestly:

- These run the **same engine** as nimbus-infer-rs behind a thin HTTP layer
  (the server adds no per-token cost beyond the channel hop). The same
  machine class in the nimbus-infer-rs README measured gpt2-124M fp32
  decode at 35.1 tok/s, distilgpt2 int8 at 92.5 tok/s, SmolLM2 fp32 at
  35.4 tok/s; the numbers here agree within that project's documented
  ±10% run-to-run variance. Cross-day, cross-process variance on this
  laptop is real — treat single rows as indicative, not as rankings.
- Quantization shows the expected memory-bandwidth decode gain
  (fp32 40.0 → int8 56.5 tok/s) with prefill slightly worse (compute-bound
  dequantize-on-the-fly), matching the core project's own findings.
- SmolLM2 prefill is slower than gpt2's — thin LLaMA-style matrices are the
  core engine's documented weak spot vs BLAS-class GEMM.
- peak RSS includes weights + one full-context KV arena (SmolLM2's 8192
  context costs ~360 MB per sequence) + runtime; `/healthz` reports the
  per-request KV size.

Also measured (fixture model, random weights — API/machine baseline only,
not a capability claim): tiny-gpt2 0.88M params, see
[results/tiny-gpt2-*.json](results/).

## Cross-engine parity (2026-10-31, real checkpoints)

Greedy decoding, identical prompts, Python/NumPy engine vs Rust engine —
**all three real models now agree token-for-token (32/32)**:

- **gpt2-124M: PASS (32/32)**
- **distilgpt2-82M: PASS (32/32)**
- **SmolLM2-135M-Instruct: PASS (32/32)** — and after the fix its replies
  terminate correctly at `<|im_end|>` (`finish_reason: stop` through the
  HTTP API).

The SmolLM2 row used to fail: logits diverged from the Python engine by up
to 12% per forward pass and greedy decoding flipped near-tied decisions.
Root cause (found by dumping the RoPE tables bit-for-bit and slicing the
model to a single layer): the Rust engine computed the RoPE frequency
exponent as `(2i)/half` instead of `i/half`, **doubling every frequency**
and scrambling positional information on every RoPE model. Learned-position
models were unaffected, which is why the gpt2 fixtures never caught it.
Fixed upstream in nimbus-infer-rs with a Python-oracle regression test;
single-layer and 30-layer logits now agree to <1e-6 relative (the usual
BLAS-vs-SIMD accumulation noise).

## Pending (not run in this environment — scripts included, numbers withheld)

| comparison | status | what is needed |
|---|---|---|
| nimbus-edge vs llama.cpp | **pending** | GGUF gpt2 + llama-bench with the same thread count / token counts |

Run and record it when llama.cpp is available:

```bash
python benchmarks/run_bench.py --models-dir checkpoints --out benchmarks/results
```

`benchmarks/run_bench.py` refuses to write a result file for a model whose
checkpoint it could not load, and never fabricates entries.
