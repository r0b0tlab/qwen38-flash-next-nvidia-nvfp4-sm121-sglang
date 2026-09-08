"""Complete synthetic promotion envelopes prove positive and negative gates.

These are reducer unit fixtures, not endpoint or benchmark evidence.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from scripts import compare as c
from scripts.freeze_benchmark import upstream_payload
from scripts.benchmark_evidence import input_hash

spec = importlib.util.spec_from_file_location(
    "promotion_helpers", Path(__file__).with_name("test_compare.py")
)
assert spec and spec.loader
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def full_side(rate, lever):
    requests = {
        "short": [upstream_payload([i + 2] * 512, 256) for i in range(8)],
        "medium": [upstream_payload([i + 2] * 2048, 512) for i in range(8)],
    }
    inputs = {
        "prose": {case: sha(case) for case in h.CASES},
        "short": {"batch": input_hash(requests["short"])},
        "medium": {"batch": input_hash(requests["medium"])},
        "vision": {f"image-{i}": sha(str(i)) for i in range(43)},
    }
    manifest = h._manifest(
        lever,
        schema="qwen38fn.promotion.v1",
        model_id="nvidia/Qwen3.8-Flash-Next-NVFP4",
        image_id="sha256:" + "a" * 64,
        source_tree="b" * 40,
        tokenizer="c" * 64,
        input_token_sha="d" * 64,
        concurrency=1,
        total_pool=262144,
        inputs=inputs,
        requests=requests,
    )
    manifest["manifest_sha256"] = c.manifest_fingerprint(manifest)

    def bind(row, lane, key):
        row["epoch_verified"] = (
            True  # explicit reducer unit fixture, not runtime evidence
        )
        if lane in requests:
            row["tag"] = f"{manifest['manifest_sha256']}/{lane}/r{row['repeat']}"
            row["native_result_sha256"] = sha(
                f"unit-native/{rate}/{lane}/{row['repeat']}"
            )
            row["request_hashes"] = [input_hash(p) for p in requests[lane]]
        row["_evidence"] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "lane": lane,
            "model_id": manifest["model_id"],
            "input_sha256": inputs[lane][key],
        }
        return row

    prose = [bind(row, "prose", row["case"]) for row in h._mk_rows(rate)]
    short = [
        {"row": bind(json.loads(line), "short", "batch")}
        for line in h._upstream(rate).splitlines()
    ]
    medium = copy.deepcopy(short)
    for item in medium:
        row = item["row"]
        row.update(
            random_input_len=2048,
            random_output_len=512,
            input_lens=[2048] * 8,
            output_lens=[512] * 8,
            observed_usage=[
                {"prompt_tokens": 2048, "completion_tokens": 512, "total_tokens": 2560}
                for _ in range(8)
            ],
            total_input_tokens=16384,
            total_output_tokens=4096,
            duration=4096 / rate,
        )
        bind(row, "medium", "batch")
    vision = [
        bind(
            {
                "case_id": name,
                "repeat": 0,
                "input_sha256": value,
                "warmup": False,
                "valid": True,
                "finish_reason": "stop",
                "error": None,
                "wall_s": 20 / rate,
                "ttft_s": 10 / rate,
            },
            "vision",
            name,
        )
        for name, value in inputs["vision"].items()
    ]
    return manifest, prose, short, medium, vision


def pair():
    b = full_side(10, "none")
    a = full_side(11, "nextn")
    return dict(
        base_manifest=b[0],
        base_rows=b[1],
        base_upstream=b[2],
        base_medium=b[3],
        base_vision=b[4],
        cand_manifest=a[0],
        cand_rows=a[1],
        cand_upstream=a[2],
        cand_medium=a[3],
        cand_vision=a[4],
    )


def test_complete_bound_all_lane_comparison_can_pass():
    report = c.compare(**pair())
    assert report["verdict"] == "PASS"
    assert report["gates"]["upstream_medium"]["pass"] is True
    assert report["gates"]["evidence_binding"]["pass"] is True


def test_complete_cli_reaches_same_verdict(tmp_path):
    import os
    import subprocess
    import sys

    data = pair()
    for prefix, mode in (("base", "ar"), ("cand", "nextn")):
        m = data[prefix + "_manifest"]
        profile = {
            "schema": 1,
            "mode": mode,
            "context_length": 32768,
            "max_total_tokens": 32768,
            "vision": {"backend": "triton_attn", "cuda_graph": True},
        }
        if mode == "nextn":
            profile["speculative"] = {"steps": 1}
        m["runtime_context"] = {"profile": profile}
        m["manifest_sha256"] = c.manifest_fingerprint(m)
        for lane in ("rows", "upstream", "medium", "vision"):
            for item in data[prefix + "_" + lane]:
                row = item["row"] if lane in ("upstream", "medium") else item
                row["_evidence"]["manifest_sha256"] = m["manifest_sha256"]
                if lane in ("upstream", "medium"):
                    declared = "short" if lane == "upstream" else "medium"
                    row["tag"] = f"{m['manifest_sha256']}/{declared}/r{row['repeat']}"
    args = []
    for side, prefix in (("baseline", "base"), ("candidate", "cand")):
        manifest = tmp_path / (side + ".manifest.json")
        manifest.write_text(json.dumps(data[prefix + "_manifest"]))
        args += ["--" + side + "-manifest", str(manifest)]
        for suffix, key in (
            ("", "rows"),
            ("-upstream", "upstream"),
            ("-medium", "medium"),
            ("-vision", "vision"),
        ):
            rows = data[prefix + "_" + key]
            if key in ("upstream", "medium"):
                rows = [item["row"] for item in rows]
            path = tmp_path / (side + suffix + ".jsonl")
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args += ["--" + side + suffix, str(path)]
    assert c.main(args) == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(Path(c.__file__).resolve()), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "field",
    [
        "base_upstream",
        "cand_upstream",
        "base_medium",
        "cand_medium",
        "base_vision",
        "cand_vision",
    ],
)
def test_omitting_any_required_lane_refuses_promotion(field):
    args = pair()
    args[field] = None
    try:
        report = c.compare(**args)
    except c.Reject:
        return
    assert report["verdict"] == "NOT_OPTIMIZED"


def test_short_rows_cannot_be_relabelled_as_medium():
    args = pair()
    args["cand_medium"] = args["cand_upstream"]
    with pytest.raises(c.Reject):
        c.compare(**args)


def test_bogus_row_rate_is_recomputed_not_trusted():
    args = pair()
    for item in args["cand_upstream"]:
        item["rate"] = 999999
        item["row"]["duration"] = item["row"]["total_output_tokens"] / 10.0
        item["row"]["output_throughput"] = 10.0
    report = c.compare(**args)
    assert report["verdict"] == "NOT_OPTIMIZED"
    assert "improvement" in report["gates"]["upstream_short"]["rejected"]


def test_manifest_row_binding_cannot_be_omitted():
    args = pair()
    del args["cand_rows"][0]["_evidence"]
    assert c.compare(**args)["verdict"] == "NOT_OPTIMIZED"


def test_selected_vision_lever_requires_observed_improvement():
    args = pair()
    # Change only the declaration and rebind those test-only envelopes.
    m = args["cand_manifest"]
    m.update(lever_kind="vision", vision_metric="wall_s")
    m["manifest_sha256"] = c.manifest_fingerprint(m)
    for row in args["cand_vision"]:
        row["wall_s"] = 2.0
    assert c.compare(**args)["gates"]["vision_lever_gain"]["pass"] is False
