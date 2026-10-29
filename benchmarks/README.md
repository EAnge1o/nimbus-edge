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
  --prompt "The quick brown fox jumps over the lazy dog." \
  --max-new-tokens 64 --runs 3 --format json --json-out benchmarks/results/<name>.json
```

## Measured (2026-10-30)

Fixture model `tiny-gpt2` — **random weights, 0.88M params, 3.2 MB, vocab
448, context 256**. It exercises the same code paths as a real model but is
*not* representative of trained-model performance; the numbers are useful as
a machine+binary sanity baseline, not as a capability claim.

Host: Windows 11 (win32 10.0.26200), 12th Gen Intel Core i7-12700H (20
threads), 15.7 GB RAM, rustc 1.99.1, target x86_64-pc-windows-gnu, profile
release. Protocol: 36 prompt tokens → 64 new tokens, greedy, 3 runs, median.

| model    | quant | prefill tok/s | decode tok/s | ttft (s) | source |
|----------|-------|--------------:|-------------:|---------:|--------|
| tiny-gpt2 | fp32 | 654.9         | 2978.3       | 0.055    | [results/tiny-gpt2-fp32.json](results/tiny-gpt2-fp32.json) |
| tiny-gpt2 | int8 | 678.0         | 2647.4       | 0.053    | [results/tiny-gpt2-int8.json](results/tiny-gpt2-int8.json) |
| tiny-gpt2 | int4 | 651.6         | 2695.8       | 0.055    | [results/tiny-gpt2-int4.json](results/tiny-gpt2-int4.json) |

Raw JSON contains the per-run timings and the full host/build record.

## Pending (not run in this environment — scripts included, numbers withheld)

The protocol requires identical model files, quantization, thread counts and
token counts across engines. The conversion + comparison harness is included
below; until it is executed on a machine with the converted checkpoints, the
comparisons stay marked pending and no numbers are published.

| comparison | status | what is needed |
|---|---|---|
| nimbus-edge vs llama.cpp (gpt2-124M, same prompt/threads) | **pending** | a GGUF gpt2 converted for llama.cpp + llama.cpp bench |
| nimbus-edge vs nimbus-infer (Python/NumPy) | **pending** | `python -m nimbus_infer convert` checkpoints (gpt2-124M, distilgpt2-82M, SmolLM2-135M) |
| nimbus-edge on real trained weights (gpt2 / distilgpt2 / SmolLM2-135M) | **pending** | converted checkpoints; the fixture above has random weights |

Run and record them when the checkpoints are available:

```bash
# 1. Convert HuggingFace models with the Python engine
git clone --depth 1 https://github.com/EAnge1o/nimbus-infer.git ../nimbus-infer
python -m nimbus_infer convert --repo gpt2 --out checkpoints/gpt2

# 2. Bench every model/quant into results/
python benchmarks/run_bench.py --models-dir checkpoints --out benchmarks/results

# 3. llama.cpp comparison (pending harness): build llama-bench with the same
#    thread count, same prompt token count and max-new-token count, save the
#    JSON next to results/, and extend this README table with a source link.
```

`benchmarks/run_bench.py` refuses to write a result file for a model whose
checkpoint it could not load, and never fabricates entries.
