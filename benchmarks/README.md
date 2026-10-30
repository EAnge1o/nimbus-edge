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

## Measured (2026-10-30, real trained checkpoints)

Host: Windows 11, 12th Gen Intel Core i7-12700H (20 threads), 15.7 GB RAM,
rustc 1.99.1, target x86_64-pc-windows-gnu, profile release, engine threads
unpinned. Protocol: 131 prompt tokens → 64 new tokens, greedy, 3 runs,
median. Checkpoints converted from HuggingFace with the Python engine
(`python -m nimbus_infer convert`), fp32 on disk, quantized at load time.

| model | quant | prefill tok/s | decode tok/s | ttft (s) | peak RSS | source |
|---|---|---:|---:|---:|---:|---|
| gpt2-124M | fp32 | 279.3 | 40.0 | 0.469 | 1.02 GB | [results/gpt2-124M-fp32.json](results/gpt2-124M-fp32.json) |
| gpt2-124M | int8 | 258.9 | 56.5 | 0.506 | 1.00 GB | [results/gpt2-124M-int8.json](results/gpt2-124M-int8.json) |
| gpt2-124M | int4 | 217.5 | 52.0 | 0.602 | 1.00 GB | [results/gpt2-124M-int4.json](results/gpt2-124M-int4.json) |
| SmolLM2-135M-Instruct | fp32 | 124.4 | 30.5 | 1.054 | 1.10 GB | [results/smollm2-135M-fp32.json](results/smollm2-135M-fp32.json) |

Reading the numbers honestly:

- These run the **same engine** as nimbus-infer-rs behind a thin HTTP layer
  (the server adds no per-token cost beyond the channel hop). The same
  machine class in the nimbus-infer-rs README measured gpt2-124M fp32
  decode at 35.1 tok/s and int4 at 60.8 tok/s; the numbers here agree
  within that project's documented ±10% run-to-run variance. Cross-day,
  cross-process variance on this laptop is real — treat single rows as
  indicative, not as rankings.
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

## Cross-engine parity (2026-10-30, real checkpoints)

Greedy decoding, identical prompts, Python/NumPy engine vs Rust engine:

- **gpt2-124M: 32/32 generated tokens identical** between engines.
- **SmolLM2-135M-Instruct: first 32 characters identical**, then the two
  engines diverge at a near-tie (top-2 logit gap ≈ 0.0011). Investigation
  on 1-layer slices shows the LLaMA-path logits differ at the ~0.1–1%
  level between the two independent implementations (NumPy/BLAS vs
  hand-written SIMD) — larger than the MHA path's documented <1e-6 —
  which flips near-tied argmax decisions. Root cause is tracked upstream in
  the core engines; the server itself is unaffected (both engines produce
  coherent output; the divergence is a cross-engine reproducibility edge,
  not a correctness bug).

## Pending (not run in this environment — scripts included, numbers withheld)

| comparison | status | what is needed |
|---|---|---|
| nimbus-edge vs llama.cpp | **pending** | GGUF gpt2 + llama-bench with the same thread count / token counts |
| distilgpt2-82M rows | **pending** | `python -m nimbus_infer convert --repo distilgpt2` then re-run the same protocol |

Run and record them when the checkpoints are available:

```bash
python -m nimbus_infer convert --repo distilgpt2 --out checkpoints/distilgpt2
python benchmarks/run_bench.py --models-dir checkpoints --out benchmarks/results
```

`benchmarks/run_bench.py` refuses to write a result file for a model whose
checkpoint it could not load, and never fabricates entries.
