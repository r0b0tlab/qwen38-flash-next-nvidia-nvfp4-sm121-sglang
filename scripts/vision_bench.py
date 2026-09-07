#!/usr/bin/env python3
"""Vision/video benchmark driver (AR baseline vs NEXTN candidate, same model).

Endpoint-side benchmark: requires an explicit --base and an explicit opt-in
environment acknowledgment (QUAL_HARNESS_VISION_ENABLED=1). Ordinary CPU CI
runs only the unit/scaffold tests; with the env unset, vision requests are
never sent and the script exits nonzero with a clear reason (fail closed).

Prompts come exclusively from scripts/make_vision_fixtures.py: they ask only
about the visual property under test and never contain the expected answer or
the fixture filename. Expected answers are never sent to the endpoint.

Every raw response body is persisted BEFORE any assertion/evaluation, so a
failing run preserves full evidence. Token accounting (prompt_tokens with
images/videos) is recorded per row but is explicitly NOT proof of correct
vision: the final text answer is asserted against the prompt property.

Timing: TTFT = request start -> first nonempty generated content/reasoning
fragment; totals = usage.completion_tokens / wall. SSE chunk gaps are never
presented as ITL or token rates.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from http_client import (  # noqa: E402
    MODEL_ID,
    ConcurrencyGate,
    OpenAICompatClient,
    thinking_request_fields,
)

import make_vision_fixtures as mvf  # noqa: E402

ENV_ENABLE = "QUAL_HARNESS_VISION_ENABLED"
THINKING = thinking_request_fields(True, "low")
MAX_TOKENS = 4096
TEMP, TOP_P = 0.0, 1.0

RESIZE_TARGETS = [512, 1024, 1536]
ASPECT_VARIANTS = {"wide_1024x512", "tall_512x1024"}


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def _data_url_image(png_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(png_bytes).decode("ascii")


def _data_url_video(mp4_bytes: bytes) -> str:
    return "data:video/mp4;base64," + base64.b64encode(mp4_bytes).decode("ascii")


def image_message(prompt: str, png_bytes: bytes, mime: str = "image/png") -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": _data_url_image(png_bytes, mime)}},
            {"type": "text", "text": prompt},
        ],
    }


def multi_image_message(prompt: str, png_bytes_list: List[bytes]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _data_url_image(b)}} for b in png_bytes_list
    ]
    content.append({"type": "text", "text": prompt})
    return {"role": "user", "content": content}


def video_message(prompt: str, mp4_bytes: bytes) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": _data_url_video(mp4_bytes)}},
            {"type": "text", "text": prompt},
        ],
    }


def resize_png(png_bytes: bytes, target: int) -> bytes:
    """Square rescale of a stored fixture to target x target (still PNG)."""
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img = img.resize((target, target), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def aspect_png(png_bytes: bytes, variant: str) -> bytes:
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if variant == "wide_1024x512":
        img = img.resize((1024, 512), Image.Resampling.LANCZOS)
    elif variant == "tall_512x1024":
        img = img.resize((512, 1024), Image.Resampling.LANCZOS)
    else:
        raise ValueError(variant)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Benchmark matrix
# ---------------------------------------------------------------------------


def build_cases(fixtures_dir: Path) -> List[Dict[str, Any]]:
    """One request case per (fixture, conditioning) combination."""
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    imgdir = fixtures_dir / "images"
    viddir = fixtures_dir / "videos"
    cases: List[Dict[str, Any]] = []

    for name, entry in manifest["images"].items():
        raw = (imgdir / f"{name}.png").read_bytes()
        prompt = mvf.PROMPTS[name]
        cases.append({"case_id": f"{name}@512", "kind": "image", "prompt": prompt, "png": raw})
        for t in RESIZE_TARGETS:
            if t != 512:
                cases.append({"case_id": f"{name}@{t}", "kind": "image", "prompt": prompt,
                              "png": resize_png(raw, t)})
        for variant in sorted(ASPECT_VARIANTS):
            cases.append({"case_id": f"{name}@{variant}", "kind": "image", "prompt": prompt,
                          "png": aspect_png(raw, variant)})

    # multi-image: both square-counterfactual variants in one request
    raw_a = (imgdir / "red_square_on_blue.png").read_bytes()
    raw_b = (imgdir / "blue_square_on_red.png").read_bytes()
    cases.append({
        "case_id": "multi_square_pair@512",
        "kind": "multi_image",
        "prompt": ("Two images are provided. For the first image state the color of "
                   "the large center square, then do the same for the second image. "
                   "Answer with two color words in order."),
        "pngs": [raw_a, raw_b],
    })

    for name in manifest["videos"]:
        cases.append({
            "case_id": f"{name}@video",
            "kind": "video",
            "prompt": mvf.PROMPTS[name],
            "mp4": (viddir / f"{name}.mp4").read_bytes(),
        })
    return cases


def run_vision_benchmark(
    base: str,
    fixtures_dir: Path,
    out_path: Path,
    *,
    variant: str,
    repeats: int = 1,
    warmup: int = 1,
    timeout_s: float = 300.0,
    no_resize_matrix: bool = False,
) -> Dict[str, Any]:
    if os.environ.get(ENV_ENABLE) != "1":
        raise SystemExit(
            f"{ENV_ENABLE} is not set: vision benchmarking sends real requests and "
            "is disabled by default. Set the env explicitly to enable."
        )
    client = OpenAICompatClient(base)
    client.verify_model()  # exact model identity before any traffic
    cases = build_cases(fixtures_dir)
    if no_resize_matrix:
        cases = [c for c in cases if "@" not in c["case_id"] or c["case_id"].endswith("@512") or c["case_id"].endswith("@video")]
    gate = ConcurrencyGate(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    # warmup: discarded, separately marked (content-free text probe is NOT used;
    # warmup uses the smallest real image case so conditioning is representative)
    first_case = cases[0]
    for w in range(warmup):
        with gate.slot():
            client.chat_stream(
                [image_message(first_case["prompt"], first_case["png"])],
                max_tokens=64, temperature=TEMP, top_p=TOP_P, thinking=THINKING,
                timeout=timeout_s,
            )

    with out_path.open("a", encoding="utf-8") as fout:
        for case in cases:
            for rep in range(repeats):
                if case["kind"] == "image":
                    msg = image_message(case["prompt"], case["png"])
                elif case["kind"] == "multi_image":
                    msg = multi_image_message(case["prompt"], case["pngs"])
                else:
                    msg = video_message(case["prompt"], case["mp4"])
                with gate.slot():  # admission precedes the clock
                    t0 = time.perf_counter()
                    res = client.chat_stream(
                        [msg], max_tokens=MAX_TOKENS, temperature=TEMP, top_p=TOP_P,
                        thinking=THINKING, timeout=timeout_s,
                    )
                row = {
                    "variant": variant,          # "AR" | "NEXTN" (label supplied by caller)
                    "case_id": case["case_id"],
                    "repeat": rep,
                    "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": MODEL_ID,
                    "kind": case["kind"],
                    "ok": res.ok,
                    "error": res.error,
                    "finish_reason": res.finish_reason,
                    "usage": res.usage,
                    "ttft_s": res.ttft_s,
                    "wall_s": res.wall_s,
                    "e2e_output_tok_per_s": res.e2e_output_tok_per_s,
                    "first_fragment_kind": res.first_fragment_kind,
                    "final_text": (res.content or "")[-2000:],
                    "raw_response": res.raw_events,   # preserved BEFORE any assertion
                }
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()  # per-row atomic progress
                if not res.ok:
                    errors.append(f"{case['case_id']}#{rep}: {res.error}: {res.error_detail}")
    return {"rows": len(rows), "errors": errors, "cases": len(cases)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="vision benchmark (AR vs NEXTN), env-gated")
    p.add_argument("--base", required=True)
    p.add_argument("--fixtures", default=str(Path(__file__).resolve().parents[1] / "fixtures"))
    p.add_argument("--output", required=True, help="JSONL output (exclusive, no rerun overwrite)")
    p.add_argument("--variant", required=True, choices=["AR", "NEXTN"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--no-resize-matrix", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.output)
    if out.exists():
        print(f"refusing to overwrite existing output: {out}", file=sys.stderr)
        return 2
    result = run_vision_benchmark(
        args.base, Path(args.fixtures), out,
        variant=args.variant, repeats=args.repeats, warmup=args.warmup,
        timeout_s=args.timeout, no_resize_matrix=args.no_resize_matrix,
    )
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
