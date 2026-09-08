#!/usr/bin/env python3
"""One real upstream SHORT/MED round with observed-wire evidence; one process per round."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmark_evidence import input_hash, load_manifest
from scripts.runtime_context import verify_http_epoch
from scripts.upstream_capture import MODEL_ID, install_capture


def admit_tokenizer(manifest, tokenizer_dir):
    from scripts.niah import load_real_tokenizer

    tokenizer = load_real_tokenizer(tokenizer_dir)
    assets = getattr(tokenizer, "_qwen38_tokenizer_assets", None)
    if (
        not assets
        or assets != manifest.get("tokenizer_assets")
        or input_hash(assets) != manifest.get("tokenizer")
        or getattr(tokenizer, "_qwen38_model_sha", None) != manifest.get("model_sha")
    ):
        raise ValueError("actual client tokenizer differs from frozen model/assets")
    return tokenizer


def upstream_arguments(manifest, lane, repeat, tokenizer_dir, output):
    if (
        lane not in ("short", "medium")
        or type(repeat) is not int
        or not 0 <= repeat < 5
    ):
        raise ValueError("invalid lane/repeat")
    input_len, output_len = (512, 256) if lane == "short" else (2048, 512)
    extra = {
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
        "return_cached_tokens_details": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return [
        "--backend",
        "sglang-oai",
        "--base-url",
        manifest["runtime_context"]["endpoint"],
        "--model",
        MODEL_ID,
        "--tokenizer",
        str(tokenizer_dir),
        "--dataset-name",
        "random-ids",
        "--dataset-path",
        "FROZEN_IN_MANIFEST",
        "--num-prompts",
        "8",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--sharegpt-output-len",
        str(output_len),
        "--random-range-ratio",
        "1",
        "--max-concurrency",
        "1",
        "--request-rate",
        "inf",
        "--warmup-requests",
        "1",
        "--flush-cache",
        "--cache-report",
        "--output-details",
        "--disable-tqdm",
        "--seed",
        str(manifest["seed"]),
        "--extra-request-body",
        json.dumps(extra),
        "--tag",
        f"{manifest['manifest_sha256']}/{lane}/r{repeat}",
        "--output-file",
        str(output),
    ]


def run_round(manifest, lane, repeat, tokenizer_dir, output, *, allow_cold_flush=False):
    if not allow_cold_flush:
        raise ValueError("explicit allow_cold_flush required for this owned runtime")
    raw_path = output.with_suffix(output.suffix + ".upstream.jsonl")
    capture_dir = output.with_suffix(output.suffix + ".capture")
    if output.exists() or raw_path.exists() or capture_dir.exists():
        raise FileExistsError("round outputs must be fresh")
    payloads = manifest["requests"][lane]
    expected_in, expected_out = (512, 256) if lane == "short" else (2048, 512)
    if len(payloads) != 8 or any(
        len(p["prompt"]) != expected_in or p["max_tokens"] != expected_out
        for p in payloads
    ):
        raise ValueError("frozen protocol length mismatch")
    if input_hash(payloads) != manifest["inputs"][lane]["batch"]:
        raise ValueError("frozen batch hash mismatch")
    if (
        hashlib.sha256((ROOT / "scripts/upstream_capture.py").read_bytes()).hexdigest()
        != manifest["capture_adapter_sha256"]
    ):
        raise ValueError("capture adapter identity drift")
    tokenizer = admit_tokenizer(manifest, tokenizer_dir)
    verify_http_epoch(manifest)
    import sglang.benchmark.datasets as datasets
    from sglang.benchmark.datasets.common import BaseDataset, DatasetRow
    from sglang.benchmark import serving as upstream

    class FrozenDataset(BaseDataset):
        @classmethod
        def from_args(cls, args):
            if args.num_prompts != 8:
                raise ValueError("request count drift")
            return cls()

        def load(self, tokenizer, model_id=None):
            if model_id != MODEL_ID:
                raise ValueError("upstream model identity drift")
            return [
                DatasetRow(
                    prompt=list(p["prompt"]),
                    prompt_len=len(p["prompt"]),
                    output_len=p["max_tokens"],
                )
                for p in payloads
            ]

    datasets.DATASET_MAPPING["random-ids"] = FrozenDataset
    output.parent.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(exist_ok=False)
    records, patch_sha = install_capture(
        upstream, payloads, capture_dir, manifest["benchmark_module_sha256"]
    )
    old_argv = sys.argv
    old_get_tokenizer, old_check_template = (
        upstream.get_tokenizer,
        upstream.check_chat_template,
    )

    def bound_tokenizer(path):
        if str(path) != str(tokenizer_dir):
            raise ValueError("upstream tokenizer path drift")
        return tokenizer

    upstream.get_tokenizer = bound_tokenizer
    upstream.check_chat_template = lambda model_id: bool(
        getattr(tokenizer, "chat_template", None)
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        sys.argv = [
            "frozen-upstream",
            *upstream_arguments(manifest, lane, repeat, tokenizer_dir, raw_path),
        ]
        upstream.cli_main()
    finally:
        sys.argv = old_argv
        upstream.get_tokenizer, upstream.check_chat_template = (
            old_get_tokenizer,
            old_check_template,
        )
    measured = sorted(
        [r for r in records if not r["warmup"]], key=lambda r: r["request_index"]
    )
    warmups = [r for r in records if r["warmup"]]
    if (
        len(warmups) != 1
        or not warmups[0]["valid"]
        or [r["request_index"] for r in measured] != list(range(8))
        or not all(r["valid"] for r in measured)
    ):
        raise ValueError("incomplete/invalid observed upstream requests")
    if (
        input_hash([r["request"] for r in measured])
        != manifest["inputs"][lane]["batch"]
    ):
        raise ValueError("actual wire requests differ from the frozen batch")
    lines = raw_path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("upstream result must contain exactly one round")
    row = json.loads(lines[0])
    if row["input_lens"] != [
        r["observed_usage"]["prompt_tokens"] for r in measured
    ] or row["output_lens"] != [
        r["observed_usage"]["completion_tokens"] for r in measured
    ]:
        raise ValueError("native result lengths differ from observed wire usage")
    verify_http_epoch(manifest)
    row.update(
        model=MODEL_ID,
        lane=lane,
        repeat=repeat,
        epoch_verified=True,
        input_sha256=manifest["inputs"][lane]["batch"],
        finish_reasons=[r["finish_reason"] for r in measured],
        observed_usage=[r["observed_usage"] for r in measured],
        request_hashes=[r["request_sha256"] for r in measured],
        native_result_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        capture_patch_sha256=patch_sha,
        capture_files=[r["wire_file"] for r in measured],
        usage_source="observed_oai_sse",
        finish_source="observed_oai_sse",
        _evidence={
            "manifest_sha256": manifest["manifest_sha256"],
            "model_id": MODEL_ID,
            "lane": lane,
            "input_sha256": manifest["inputs"][lane]["batch"],
        },
    )
    with output.open("x") as stream:
        json.dump(row, stream)
        stream.write("\n")
    return {
        "status": "BOUND_UPSTREAM_ROUND_COMPLETE",
        "lane": lane,
        "repeat": repeat,
        "requests": 8,
        "output": str(output),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--lane", choices=("short", "medium"), required=True)
    p.add_argument("--repeat", type=int, required=True)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-cold-flush", action="store_true")
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    print(
        json.dumps(
            run_round(
                manifest,
                args.lane,
                args.repeat,
                args.tokenizer_dir,
                args.output,
                allow_cold_flush=args.allow_cold_flush,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
