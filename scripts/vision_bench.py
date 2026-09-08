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
import hashlib
import re
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from http_client import (  # noqa: E402
    MODEL_ID,
    ConcurrencyGate,
    OpenAICompatClient,
    thinking_request_fields,
    build_chat_payload,
    row_validity,
)

import make_vision_fixtures as mvf  # noqa: E402
from benchmark_evidence import bind_requests, input_hash, load_manifest  # noqa: E402

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


def image_message(
    prompt: str, png_bytes: bytes, mime: str = "image/png"
) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url_image(png_bytes, mime)},
            },
            {"type": "text", "text": prompt},
        ],
    }


def multi_image_message(prompt: str, png_bytes_list: List[bytes]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _data_url_image(b)}}
        for b in png_bytes_list
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
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("visual corpus image changed: " + name)
        prompt = mvf.PROMPTS[name]
        cases.append(
            {"case_id": f"{name}@512", "kind": "image", "prompt": prompt, "png": raw}
        )
        for t in RESIZE_TARGETS:
            if t != 512:
                cases.append(
                    {
                        "case_id": f"{name}@{t}",
                        "kind": "image",
                        "prompt": prompt,
                        "png": resize_png(raw, t),
                    }
                )
        for variant in sorted(ASPECT_VARIANTS):
            cases.append(
                {
                    "case_id": f"{name}@{variant}",
                    "kind": "image",
                    "prompt": prompt,
                    "png": aspect_png(raw, variant),
                }
            )

    # multi-image: both square-counterfactual variants in one request
    raw_a = (imgdir / "red_square_on_blue.png").read_bytes()
    raw_b = (imgdir / "blue_square_on_red.png").read_bytes()
    cases.append(
        {
            "case_id": "multi_square_pair@512",
            "kind": "multi_image",
            "prompt": (
                "Two images are provided. For the first image state the color of "
                "the large center square, then do the same for the second image. "
                "Answer with two color words in order."
            ),
            "pngs": [raw_a, raw_b],
        }
    )

    for name, entry in manifest["videos"].items():
        video_bytes = (viddir / f"{name}.mp4").read_bytes()
        if hashlib.sha256(video_bytes).hexdigest() != entry["sha256"]:
            raise ValueError("visual corpus video changed: " + name)
        cases.append(
            {
                "case_id": f"{name}@video",
                "kind": "video",
                "prompt": mvf.PROMPTS[name],
                "mp4": video_bytes,
            }
        )
    return cases


def case_message(case):
    if case["kind"] == "image":
        return image_message(case["prompt"], case["png"])
    if case["kind"] == "multi_image":
        return multi_image_message(case["prompt"], case["pngs"])
    if case["kind"] == "video":
        return video_message(case["prompt"], case["mp4"])
    raise ValueError("unknown visual case kind")


def case_payload(case, max_tokens=MAX_TOKENS):
    return build_chat_payload(
        [case_message(case)],
        model=MODEL_ID,
        max_tokens=max_tokens,
        temperature=TEMP,
        top_p=TOP_P,
        thinking=THINKING,
    )


def media_hashes(case):
    blobs = (
        [case["png"]]
        if case["kind"] == "image"
        else (case["pngs"] if case["kind"] == "multi_image" else [case["mp4"]])
    )
    return [hashlib.sha256(blob).hexdigest() for blob in blobs]


def semantic_answer(case_id, text):
    """Score only the exact requested visual property, never reasoning text."""
    if not isinstance(text, str) or text.rfind("<think>") > text.rfind("</think>"):
        return False
    final = text.split("</think>")[-1].strip()
    base = case_id.split("@", 1)[0]
    expected = "red blue" if base == "multi_square_pair" else mvf.EXPECTED.get(base)
    if expected is None:
        return False
    words = re.findall(r"[A-Za-z0-9]+", final)
    if base == "ocr_text":
        return words == [expected]
    words = [word.lower() for word in words]
    if expected in ("3", "5"):
        words = [{"three": "3", "five": "5"}.get(word, word) for word in words]
    return words == expected.split()


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
    promotion_manifest=None,
) -> Dict[str, Any]:
    if os.environ.get(ENV_ENABLE) != "1":
        raise SystemExit(f"{ENV_ENABLE} is not set: real vision traffic is disabled")
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative integers")
    raw_path = out_path.with_suffix(out_path.suffix + ".raw.jsonl")
    inputs_path = out_path.with_suffix(out_path.suffix + ".inputs.json")
    if any(path.exists() for path in (out_path, raw_path, inputs_path)):
        raise FileExistsError("vision output and sidecars must all be fresh")
    cases = build_cases(fixtures_dir)
    if no_resize_matrix:
        cases = [case for case in cases if case["case_id"].endswith(("@512", "@video"))]
    if not cases or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("visual case IDs must be nonempty and unique")
    requests = json.loads(
        json.dumps({case["case_id"]: case_payload(case) for case in cases})
    )
    bindings = bind_requests(promotion_manifest, "vision", requests)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with inputs_path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": "qwen38fn.vision-inputs.v1",
                "model_id": MODEL_ID,
                "input_representation": "canonical_request_json",
                "repeats": repeats,
                "warmup_cycles": warmup,
                "inputs": {
                    key: input_hash(payload) for key, payload in requests.items()
                },
                "requests": requests,
                "media_sha256": {case["case_id"]: media_hashes(case) for case in cases},
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
    rows, errors, transport_errors, semantic_errors = [], [], [], []
    gate = ConcurrencyGate(1)
    with (
        out_path.open("x", encoding="utf-8") as output,
        raw_path.open("x", encoding="utf-8") as raw_output,
    ):
        client = OpenAICompatClient(base)
        client.verify_model()

        def request_one(case, repeat, is_warmup):
            limit = 64 if is_warmup else MAX_TOKENS
            payload = json.loads(json.dumps(requests[case["case_id"]]))
            payload["max_tokens"] = limit
            controls = {
                key: payload[key]
                for key in ("chat_template_kwargs", "reasoning_effort")
                if key in payload
            }
            with gate.slot():
                res = client.chat_stream(
                    payload["messages"],
                    max_tokens=limit,
                    temperature=payload["temperature"],
                    top_p=payload["top_p"],
                    thinking=controls,
                    timeout=timeout_s,
                )
            row = {
                "variant": variant,
                "case_id": case["case_id"],
                "repeat": repeat,
                "warmup": is_warmup,
                "model": MODEL_ID,
                "model_reported": res.model_reported,
                "kind": case["kind"],
                "ok": res.ok,
                "error": res.error,
                "finish_reason": res.finish_reason,
                "usage": res.usage,
                "ttft_s": res.ttft_s,
                "wall_s": res.wall_s,
                "e2e_output_tok_per_s": res.e2e_output_tok_per_s,
                "first_fragment_kind": res.first_fragment_kind,
                "final_text": res.content or "",
                "raw_response": res.raw_events,
                "input_sha256": input_hash(payload),
                "media_sha256": media_hashes(case),
                "sampling": {
                    "max_tokens": limit,
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    **controls,
                },
                "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            # Raw evidence is durable before the semantic function can fail.
            raw_output.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
            raw_output.flush()
            valid, reason = row_validity(res)
            if res.model_reported != MODEL_ID:
                valid, reason = False, "wrong_reported_model"
            semantic = (
                semantic_answer(case["case_id"], row["final_text"])
                if not is_warmup
                else None
            )
            if not is_warmup and not semantic:
                valid, reason = False, reason or "semantic_mismatch"
            row.update(
                valid=valid and not is_warmup,
                semantic_ok=semantic,
                reason="warmup_not_measured" if is_warmup else reason,
            )
            if not is_warmup and bindings:
                row["_evidence"] = bindings[case["case_id"]]
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            if res.error or res.model_reported != MODEL_ID:
                message = (
                    f"{case['case_id']}#{repeat}: {res.error or 'wrong_reported_model'}"
                )
                transport_errors.append(message)
                errors.append(message)
                return False
            if not is_warmup and not row["valid"]:
                message = f"{case['case_id']}#{repeat}: {row['reason']}"
                errors.append(message)
                semantic_errors.append(message)
            return True

        # Warm the entire declared envelope, not only the first 512px image.
        healthy = True
        for warm_repeat in range(warmup):
            for case in cases:
                if not request_one(case, warm_repeat, True):
                    healthy = False
                    break
            if not healthy:
                break
        if healthy:
            for case in cases:
                for repeat in range(repeats):
                    if not request_one(case, repeat, False):
                        healthy = False
                        break
                if not healthy:
                    break
    measured = [row for row in rows if not row["warmup"]]
    return {
        "rows": len(rows),
        "cases": len(cases),
        "measured_rows": len(measured),
        "warmup_rows": len(rows) - len(measured),
        "errors": errors,
        "transport_errors": transport_errors,
        "semantic_errors": semantic_errors,
        "promotion_bound": bool(bindings),
        "scope": "VISION_DIAGNOSTIC" if not bindings else "BOUND_VISION_LANE",
        "ok": len(measured) == len(cases) * repeats and not errors,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="vision benchmark (AR vs NEXTN), env-gated")
    p.add_argument("--base", required=True)
    p.add_argument(
        "--fixtures", default=str(Path(__file__).resolve().parents[1] / "fixtures")
    )
    p.add_argument(
        "--output", required=True, help="JSONL output (exclusive, no rerun overwrite)"
    )
    p.add_argument("--variant", required=True, choices=["AR", "NEXTN"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--no-resize-matrix", action="store_true")
    p.add_argument(
        "--promotion-manifest",
        help="frozen common manifest; absent means diagnostic-only",
    )
    args = p.parse_args(argv)

    out = Path(args.output)
    if out.exists():
        print(f"refusing to overwrite existing output: {out}", file=sys.stderr)
        return 2
    result = run_vision_benchmark(
        args.base,
        Path(args.fixtures),
        out,
        variant=args.variant,
        repeats=args.repeats,
        warmup=args.warmup,
        timeout_s=args.timeout,
        no_resize_matrix=args.no_resize_matrix,
        promotion_manifest=load_manifest(args.promotion_manifest),
    )
    print(json.dumps(result, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
