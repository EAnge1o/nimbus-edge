#!/usr/bin/env python3
"""Print the exact benchmark prompt used by every real-model row in
benchmarks/README.md (131 tokens with the gpt2 tokenizer). Baked in as
literal text so reproduction needs nothing but this file."""

# The exact prompt captured in every real-model results file (131 gpt2
# tokens): the 13-repetition sentence with its trailing space intact.
PROMPT = ("The quick brown fox jumps over the lazy dog. " * 13)

if __name__ == "__main__":
    print(PROMPT)
