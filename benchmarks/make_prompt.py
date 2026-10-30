#!/usr/bin/env python3
"""Print a prompt of ~N tokens (best effort) for benchmark reproduction.

Uses the tokenizer of a checkpoint if available, else falls back to a plain
repeated sentence. Usage: python benchmarks/make_prompt.py [tokens] [tokenizer_dir]
"""
import sys
from pathlib import Path

BASE = "The quick brown fox jumps over the lazy dog. "


def main() -> int:
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    tok_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    text = BASE.strip() + " "
    if tok_dir and tok_dir.joinpath("vocab.json").exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".." / "nimbus-infer"))
        try:
            from nimbus_infer.tokenizer import BPETokenizer  # type: ignore

            tok = BPETokenizer.load(str(tok_dir))
            ids = tok.encode(text)
            while len(ids) < want:
                text += BASE
                ids = tok.encode(text)
        except Exception:
            pass
    # Trim to roughly the requested character budget.
    print(text[: want * 5].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
