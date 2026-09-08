#!/usr/bin/env python3
"""Produce the common pre-traffic manifest from observed runtime and real inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import bench_real, compare, niah, vision_bench
from scripts.benchmark_evidence import input_hash
from scripts.runtime_context import verify_http_epoch

MODEL_ID = niah.MODEL_ID


def frozen_token_rows(tokenizer, length, seed):
    if type(length) is not int or length <= 0:
        raise ValueError("invalid frozen input length")
    special = set(tokenizer.all_special_ids)
    available = sorted(
        {
            token
            for token in tokenizer.get_vocab().values()
            if type(token) is int and token >= 0 and token not in special
        }
    )
    if len(available) < 8:
        raise ValueError("tokenizer vocabulary is too small")
    rng = random.Random(seed)
    starts = rng.sample(available, 8)
    return [[start] + rng.choices(available, k=length - 1) for start in starts]


def upstream_payload(ids, output_length):
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(x) is not int or x < 0 for x in ids)
    ):
        raise ValueError("token IDs must be literal nonnegative integers")
    if type(output_length) is not int or output_length <= 0:
        raise ValueError("output length must be a positive integer")
    return {
        "model": MODEL_ID,
        "prompt": list(ids),
        "best_of": 1,
        "max_tokens": output_length,
        "stream": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def prepare_requests(tokenizer, fixtures_dir, seed=20260907):
    short = frozen_token_rows(tokenizer, 512, seed)
    medium = frozen_token_rows(tokenizer, 2048, seed + 1)
    prose = bench_real.prose_payloads()
    visual_cases = vision_bench.build_cases(fixtures_dir)
    if len(visual_cases) != 43:
        raise ValueError("full visual envelope requires 43 cases")
    vision = {case["case_id"]: vision_bench.case_payload(case) for case in visual_cases}
    prose_tokens = {}
    for case, payload in prose.items():
        template = tokenizer.apply_chat_template(
            payload["messages"],
            tokenize=False,
            add_generation_prompt=True,
            **payload["chat_template_kwargs"],
        )
        ids = tokenizer.encode(template, add_special_tokens=False)
        if not ids or any(type(token) is not int for token in ids):
            raise ValueError("invalid prose tokenization")
        prose_tokens[case] = ids
    return {
        "requests": {
            "short": [upstream_payload(ids, 256) for ids in short],
            "medium": [upstream_payload(ids, 512) for ids in medium],
            "prose": prose,
            "vision": vision,
        },
        "token_arrays": {"short": short, "medium": medium, "prose": prose_tokens},
    }


def compose_manifest(
    context, runtime_lock, plans, tokenizer_assets, model_sha, lever=None
):
    if (
        context.get("schema") != "qwen38fn.runtime-context.v1"
        or context.get("model_id") != MODEL_ID
        or context.get("model_sha") != model_sha
    ):
        raise ValueError("observed runtime and admitted tokenizer model differ")
    if context["source_tree"] != runtime_lock["sglang"]["tree"]:
        raise ValueError("runtime source tree mismatch")
    if not tokenizer_assets:
        raise ValueError("tokenizer asset hashes missing")
    requests = plans["requests"]
    inputs = {
        "short": {"batch": input_hash(requests["short"])},
        "medium": {"batch": input_hash(requests["medium"])},
        "prose": {
            case: input_hash(payload) for case, payload in requests["prose"].items()
        },
        "vision": {
            case: input_hash(payload) for case, payload in requests["vision"].items()
        },
    }
    profile = context["profile"]
    lever = lever or ("nextn" if profile["mode"] == "nextn" else "none")
    supported = {
        "none",
        "nextn",
        "chunked_prefill_size",
        "max_mamba_cache_size",
        "ple_rss_gib",
        "kv_cache_dtype",
        "mem_fraction_static",
        "vision.backend",
        "vision.cuda_graph",
        "mm_processor_worker_num",
    }
    if lever not in supported:
        raise ValueError("unrecognized single profile lever")
    manifest = {
        "schema": "qwen38fn.promotion.v1",
        **{
            key: context[key]
            for key in (
                "model_id",
                "model_sha",
                "image_id",
                "source_tree",
                "context",
                "total_pool",
                "concurrency",
            )
        },
        "tokenizer": input_hash(tokenizer_assets),
        "tokenizer_assets": tokenizer_assets,
        "input_token_sha": input_hash(plans["token_arrays"]),
        "token_arrays": plans["token_arrays"],
        "inputs": inputs,
        "requests": requests,
        "runtime_context": context,
        "benchmark_module_sha256": runtime_lock["sglang"]["python_files"][
            "benchmark/serving.py"
        ],
        "capture_adapter_sha256": hashlib.sha256(
            (ROOT / "scripts/upstream_capture.py").read_bytes()
        ).hexdigest(),
        "sampling": {
            "text": {"temperature": 0.0, "top_p": 1.0, "enable_thinking": False},
            "vision": {"temperature": 0.0, "top_p": 1.0, **vision_bench.THINKING},
        },
        "lever": lever,
        "lever_kind": "vision"
        if lever.startswith("vision.") or lever == "mm_processor_worker_num"
        else "runtime",
        "vision_metric": "wall_s",
        "repeats": 5,
        "vision_repeats": 1,
        "seed": 20260907,
    }
    manifest["manifest_sha256"] = compare.manifest_fingerprint(manifest)
    compare.validate_manifest(manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--runtime-lock", type=Path, required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lever")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("manifest output must be fresh")
    context = json.loads(args.context.read_text())
    runtime = json.loads(args.runtime_lock.read_text())
    tokenizer = niah.load_real_tokenizer(args.tokenizer_dir)
    plans = prepare_requests(tokenizer, args.fixtures)
    manifest = compose_manifest(
        context,
        runtime,
        plans,
        tokenizer._qwen38_tokenizer_assets,
        tokenizer._qwen38_model_sha,
        args.lever,
    )
    verify_http_epoch(manifest)  # no inference, only fresh identity observations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(manifest, stream, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "BENCHMARK_MANIFEST_FROZEN",
                "manifest_sha256": manifest["manifest_sha256"],
                "inference_requests": 0,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
