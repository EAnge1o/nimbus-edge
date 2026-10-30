#!/usr/bin/env python3
"""Render the README terminal screenshot from real command output.

Same principle as the sibling projects: the image in the README is rendered
from actual captured output of the commands shown (real model, real HTTP
round-trips). Re-capture by rerunning the serve + curl session and updating
the FLOW lines below, then rerunning this script:

    python docs/make_demo_svg.py
"""
from __future__ import annotations

from pathlib import Path

# --- captured from real runs (SmolLM2-135M-Instruct, i7-12700H) -----------
FLOW = [
    # (kind, text) kind: cmd | out | dim | ok
    ("cmd",  "nimbus-edge models import --path checkpoints/smollm2-135m-instruct \\"),
    ("cont", "    --name smollm2-135m-instruct --chat-template chatml"),
    ("out",  "imported smollm2-135m-instruct (4 files, 514.5MB, template: chatml)"),
    ("cmd",  "nimbus-edge serve --model smollm2-135m-instruct"),
    ("dim",  "loading model smollm2-135m-instruct (fp32) ..."),
    ("dim",  "nimbus-edge v0.2.0 | model smollm2-135m-instruct (llama, fp32,"),
    ("dim",  "    weights 513.0 MB, kv cache 360.0 MB/request) | context 8192 tokens"),
    ("ok",   "listening on http://127.0.0.1:11434 | max_concurrency 1 | auth off"),
    ("cmd",  'curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" \\'),
    ("cont", '     -d \'{"model":"smollm2-135m-instruct","messages":[],"max_tokens":24,"temperature":0}\''),
    ("ok",   '{"choices":[{"message":{"role":"assistant",'),
    ("ok",   '  "content":"\\n\\nThe capital of France is Paris."},'),
    ("ok",   '  "finish_reason":"length"}],"usage":{"prompt_tokens":33,"completion_tokens":24}}'),
    ("dim",  "# OpenAI SDK works unmodified: base_url=http://127.0.0.1:11434/v1"),
]

COLORS = {
    "cmd": "#79c0ff",
    "cont": "#79c0ff",
    "out": "#c9d1d9",
    "dim": "#8b949e",
    "ok": "#7ee787",
}

FONT = "ui-monospace,'Cascadia Code','JetBrains Mono',Menlo,Consolas,monospace"
FS = 13.0
LH = 21.0
PAD_X = 20.0
CHAR_W = 7.9


def esc(text: str) -> str:
    # XML whitespace collapsing would destroy alignment; NBSP has the same
    # advance width in monospace.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "\u00a0")
    )


def main() -> None:
    title_h = 34.0
    pad_top = 14.0
    pad_bottom = 16.0
    width = max(len(t) for _, t in FLOW) * CHAR_W + PAD_X * 2
    height = title_h + pad_top + LH * len(FLOW) + pad_bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        'viewBox="0 0 {:.0f} {:.0f}">'.format(width, height),
        "<rect width=\"100%\" height=\"100%\" rx=\"8\" fill=\"#0d1117\"/>",
        '<rect width="100%" height="34" rx="8" fill="#161b22"/>',
        '<circle cx="18" cy="17" r="6" fill="#ff5f56"/>',
        '<circle cx="38" cy="17" r="6" fill="#ffbd2e"/>',
        '<circle cx="58" cy="17" r="6" fill="#27c93f"/>',
        f'<text x="{width / 2:.0f}" y="22" text-anchor="middle" font-family="{FONT}" '
        f'font-size="12" fill="#8b949e">nimbus-edge — real terminal output</text>',
    ]
    y = title_h + pad_top + FS
    for kind, text in FLOW:
        parts.append(
            f'<text x="{PAD_X}" y="{y:.1f}" font-family="{FONT}" font-size="{FS}" '
            f'fill="{COLORS[kind]}">{esc(text)}</text>'
        )
        y += LH
    parts.append("</svg>")

    out = Path(__file__).resolve().parent / "demo.svg"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out} ({width:.0f}x{height:.0f})")


if __name__ == "__main__":
    main()
