#!/usr/bin/env python3
"""Bench every checkpoint in a directory with `nimbus-edge bench` and write
one JSON result file per (model, quant) into --out.

Honesty rules:
- only models whose checkpoint loads are benchmarked;
- each result file carries the full host/build record produced by the binary;
- the script never invents numbers and exits nonzero if any bench failed.

Usage:
  python benchmarks/run_bench.py --models-dir checkpoints --out benchmarks/results \
      [--binary target/release/nimbus-edge] [--quants fp32,int8,int4] \
      [--max-new-tokens 64] [--runs 3]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROMPT = "The quick brown fox jumps over the lazy dog."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--binary", default="target/release/nimbus-edge")
    parser.add_argument("--quants", default="fp32,int8,int4")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None,
                        help="pin engine threads (recommended for comparability)")
    args = parser.parse_args()

    binary = Path(args.binary)
    if not binary.exists():
        print(f"error: binary not found at {binary}; run cargo build --release", file=sys.stderr)
        return 2
    if not args.models_dir.is_dir():
        print(f"error: models dir not found: {args.models_dir}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    failures = 0

    for model_dir in sorted(p for p in args.models_dir.iterdir() if p.is_dir()):
        name = model_dir.name
        for quant in quants:
            out_file = args.out / f"{name}-{quant}.json"
            cmd = [
                str(binary), "bench",
                "--model", str(model_dir),
                "--prompt", PROMPT,
                "--max-new-tokens", str(args.max_new_tokens),
                "--runs", str(args.runs),
                "--quant", quant,
                "--format", "json",
                "--json-out", str(out_file),
            ]
            if args.threads is not None:
                cmd += ["--threads", str(args.threads)]
            print(f"bench {name} [{quant}] ...", flush=True)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  FAILED: {proc.stderr.strip()}", file=sys.stderr)
                if out_file.exists():
                    out_file.unlink()
                failures += 1
                continue
            report = json.loads(out_file.read_text(encoding="utf-8"))
            r = report["results"]
            print(
                f"  prefill {r['prefill_tok_s']} tok/s | decode {r['decode_tok_s']} tok/s "
                f"| ttft {r['ttft_s']} s"
            )
    if failures:
        print(f"{failures} bench run(s) failed; no numbers were fabricated", file=sys.stderr)
        return 1
    print("all benches recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
