#!/usr/bin/env python3
"""Fail-closed C1 comparison reducer (AR baseline vs NEXTN candidate).

Consumes JSONL row files produced by scripts/bench_real.py for both variants
and decides NOT_OPTIMIZED vs PASS. The reducer fails closed: any invalid row,
missing key, zero/NaN rate, wrong model, mismatched frozen control, missing
required detail, count mismatch, or percentile regression refuses the verdict
(NOT_OPTIMIZED) — never a guessed PASS.

Gates (all must hold):
- parity: manifest model sha, image id, source/tree, tokenizer, all-input
  token SHA, sampling, context, total pool, concurrency must match, EXCEPT
  the explicitly declared lever field (speculative/profile kind);
- each side: exactly 5 measured repeats x 3 prose cases, no missing, no dupes;
- per-row validity already enforced by bench_real (finish=stop, meaningful
  content, usage present); reducer re-checks and re-computes rates from
  usage.completion_tokens / wall_s, refusing zero/NaN/subnormal values;
- improvement: median(e2erate) improvement >= 5% for EACH of short/medium/
  prose (no best-single cherry pick), computed over ALL 5 rounds;
- vision: p95 TTFT must not regress more than 5% (when vision rows provided);
  a selected vision lever must claim a >=5% gain on an OBSERVABLE metric;
  wall-clock TTFT/total only — never a fake encoder timing that was not
  observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

# The public script may be called from an evidence directory, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Dict, List, Optional, Sequence

REQUIRED_UPSTREAM_KEYS = (
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "output_throughput",
)
REQUIRED_MANIFEST_KEYS = (
    "model_sha",
    "image_id",
    "source_tree",
    "tokenizer",
    "input_token_sha",
    "sampling",
    "context",
    "total_pool",
    "concurrency",
)
LEVER_KEY = "lever"  # the ONLY field allowed to differ (explicit declaration)
IMPROVEMENT_THRESHOLD = 0.05
VISION_TTFT_REGRESSION_LIMIT = 0.05
VISION_LEVER_GAIN = 0.05


class Reject(Exception):
    """Verdict refused (fail-closed)."""


# ---------------------------------------------------------------------------
# small math helpers (no silent NaN acceptance)
# ---------------------------------------------------------------------------


def require_finite_positive(x: Any, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise Reject(f"{what} is not a number: {x!r}")
    v = float(x)
    if math.isnan(v) or math.isinf(v) or v <= 0.0:
        raise Reject(f"{what} must be finite and > 0, got {v!r}")
    return v


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def percentile(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


# ---------------------------------------------------------------------------
# row loading / per-row re-validation
# ---------------------------------------------------------------------------


def load_rows(path: Path, variant: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Reject(f"{variant}:{path.name}:{lineno}: unparseable row: {exc}")
            if not isinstance(row, dict):
                raise Reject(f"{variant}:{path.name}:{lineno}: row is not an object")
            rows.append(row)
    if not rows:
        raise Reject(f"{variant}: no rows in {path}")
    return rows


def measured_prose_rows(
    rows: List[Dict[str, Any]], variant: str
) -> List[Dict[str, Any]]:
    """Keep complete measured prose rows; reject missing/invalid ones."""
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        if row.get("warmup") is True or row.get("repeat") == "warmup":
            continue
        missing = [
            k
            for k in (
                "case",
                "repeat",
                "valid",
                "finish_reason",
                "usage",
                "wall_s",
                "ttft_s",
            )
            if k not in row
        ]
        if missing:
            raise Reject(f"{variant}: row {i} missing keys {missing}")
        if row["valid"] is not True:
            raise Reject(
                f"{variant}: invalid measured row {row.get('case')}#{row.get('repeat')}: "
                f"{row.get('reason') or row.get('error')}"
            )
        if (
            row.get("model") != "nvidia/Qwen3.8-Flash-Next-NVFP4"
            or row["finish_reason"] != "stop"
        ):
            raise Reject(f"{variant}: wrong model or incomplete measured prose")
        usage = row["usage"]
        if not isinstance(usage, dict):
            raise Reject(f"{variant}: row {i} usage is not an object")
        if any(
            type(usage.get(k)) is not int
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise Reject(f"{variant}: usage must be exact JSON integers")
        if (
            usage["prompt_tokens"] < 0
            or usage["total_tokens"]
            != usage["prompt_tokens"] + usage["completion_tokens"]
        ):
            raise Reject(f"{variant}: usage total mismatch")
        comp = require_finite_positive(
            usage.get("completion_tokens"), f"{variant} row {i} completion_tokens"
        )
        wall = require_finite_positive(row["wall_s"], f"{variant} row {i} wall_s")
        if require_finite_positive(row["ttft_s"], "prose ttft_s") > wall:
            raise Reject("prose TTFT exceeds wall time")
        if (
            row.get("ok") is not True
            or row.get("error") is not None
            or not row.get("raw_events")
        ):
            raise Reject("prose row lacks successful raw response evidence")
        rate = comp / wall  # recomputed, never trusted from the row
        if row.get("e2erate") is not None:
            stated = float(row["e2erate"])
            if abs(stated - rate) > max(1e-6, 0.01 * rate):
                raise Reject(f"{variant}: row {i} e2erate disagrees with usage/wall")
        row = dict(row)
        row["_rate"] = rate
        out.append(row)
    return out


def check_counts(
    rows: List[Dict[str, Any]], variant: str, repeats: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    expected_cases = {
        "db_index_write_read",
        "city_heat_water",
        "compression_vs_random_access",
    }
    if set(by_case) != expected_cases:
        raise Reject(
            f"{variant}: missing prose cases {sorted(expected_cases - set(by_case))}"
        )
    for case in sorted(expected_cases):
        seen = [r["repeat"] for r in by_case[case]]
        if any(type(value) is not int for value in seen):
            raise Reject("repeat IDs must be exact integers")
        if len(seen) != len(set(seen)):
            raise Reject(f"{variant}: duplicate repeats in {case}")
        if len(seen) != repeats:
            raise Reject(
                f"{variant}: {case} has {len(seen)} measured rows, expected {repeats}"
            )
        if sorted(seen) != list(range(repeats)):
            raise Reject(
                f"{variant}: {case} repeats are {sorted(seen)}, expected 0..{repeats - 1}"
            )
    return by_case


def check_prose_pass_cases(rows: List[Dict[str, Any]], variant: str) -> None:
    """When whole-run JSONL is fed, short/medium/prose all must be present."""
    cases = {r.get("case") for r in rows}
    for required in ("short", "medium"):
        if required not in cases and not (
            cases
            & {"db_index_write_read", "city_heat_water", "compression_vs_random_access"}
        ):
            raise Reject(
                f"{variant}: neither upstream {required} rows nor prose rows present"
            )


# ---------------------------------------------------------------------------
# upstream detail JSON (sglang.benchmark.serving short/medium)
# ---------------------------------------------------------------------------


def _upstream_rate(row, variant, lane):
    if lane not in ("short", "medium") or not isinstance(row, dict):
        raise Reject("unknown upstream lane or malformed row")
    required = set(REQUIRED_UPSTREAM_KEYS) | {
        "backend",
        "max_concurrency",
        "random_input_len",
        "random_output_len",
        "input_lens",
        "output_lens",
        "errors",
        "generated_texts",
        "ttfts",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise Reject(f"{variant}: missing keys {missing}")
    if type(row["completed"]) is not int or row["completed"] != 8:
        raise Reject(f"{variant}: completed={row['completed']}, expected 8")
    inp, out = (512, 256) if lane == "short" else (2048, 512)
    if (
        row.get("usage_source") != "observed_oai_sse"
        or row.get("finish_source") != "observed_oai_sse"
    ):
        raise Reject("upstream rows require observed wire usage and finish evidence")
    if row.get("finish_reasons") != ["length"] * 8:
        raise Reject("upstream fixed-output finish reasons are not all length")
    usage = row.get("observed_usage")
    if not isinstance(usage, list) or len(usage) != 8:
        raise Reject("upstream observed usage is missing")
    for item in usage:
        if not isinstance(item, dict) or any(
            type(item.get(key)) is not int or item[key] != value
            for key, value in (
                ("prompt_tokens", inp),
                ("completion_tokens", out),
                ("total_tokens", inp + out),
            )
        ):
            raise Reject(
                "upstream observed usage differs from the exact input/output protocol"
            )
    cached = row.get("cached_tokens")
    if (
        not isinstance(cached, list)
        or len(cached) != 8
        or any(type(x) is not int or x != 0 for x in cached)
    ):
        raise Reject("upstream cold-cache observations are missing or nonzero")
    hashes = row.get("request_hashes")
    if (
        not isinstance(hashes, list)
        or len(hashes) != 8
        or any(
            not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
            for x in hashes
        )
    ):
        raise Reject("upstream observed request hashes are missing")
    for field, expected in (
        ("random_input_len", inp),
        ("random_output_len", out),
        ("max_concurrency", 1),
        ("total_input_tokens", 8 * inp),
        ("total_output_tokens", 8 * out),
    ):
        if type(row[field]) is not int or row[field] != expected:
            raise Reject(f"{variant}: {lane} protocol mismatch in {field}")
    if row["backend"] != "sglang-oai":
        raise Reject("upstream backend is not the frozen sglang-oai protocol")
    for field, value in (("input_lens", inp), ("output_lens", out)):
        if (
            not isinstance(row[field], list)
            or len(row[field]) != 8
            or any(type(x) is not int or x != value for x in row[field])
        ):
            raise Reject(f"{variant}: per-request {field} mismatch")
    if (
        row["errors"] != [""] * 8
        or not isinstance(row["generated_texts"], list)
        or len(row["generated_texts"]) != 8
        or any(not isinstance(x, str) or not x for x in row["generated_texts"])
    ):
        raise Reject(f"{variant}: incomplete or failed upstream requests")
    if not isinstance(row["ttfts"], list) or len(row["ttfts"]) != 8:
        raise Reject("upstream request timing count mismatch")
    for ttft in row["ttfts"]:
        require_finite_positive(ttft, "upstream ttft")
    rate = row["total_output_tokens"] / require_finite_positive(
        row["duration"], "duration"
    )
    stated = require_finite_positive(row["output_throughput"], "output_throughput")
    if abs(stated - rate) > max(1e-6, 0.02 * rate):
        raise Reject("upstream output_throughput disagrees with tokens/duration")
    return rate


def load_upstream_detail(
    path: Path, variant: str, num_prompts: int = 8, *, lane: str = "short"
) -> List[Dict[str, Any]]:
    """Parse upstream bench_serving JSONL preserving raw details.

    Requires the source keys: duration, completed, total_input_tokens,
    total_output_tokens, output_throughput; requires completed == num_prompts;
    the rate actually used is output/duration recomputed from raw values.
    """
    detail: List[Dict[str, Any]] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Reject(
                    f"{variant}:{path.name}:{lineno}: bad upstream JSON: {exc}"
                )
            missing = [k for k in REQUIRED_UPSTREAM_KEYS if k not in row]
            if missing:
                raise Reject(f"{variant}:{path.name}:{lineno}: missing keys {missing}")
            completed = row["completed"]
            if completed != num_prompts:
                raise Reject(
                    f"{variant}:{path.name}:{lineno}: completed={completed}, expected {num_prompts}"
                )
            duration = require_finite_positive(row["duration"], f"{variant} duration")
            out_tokens = require_finite_positive(
                row["total_output_tokens"], f"{variant} total_output_tokens"
            )
            rate = out_tokens / duration  # output/duration
            stated = row["output_throughput"]
            require_finite_positive(stated, f"{variant} output_throughput")
            if abs(float(stated) - rate) > max(1e-6, 0.02 * rate):
                raise Reject(
                    f"{variant}:{path.name}:{lineno}: output_throughput {stated} "
                    f"disagrees with total_output_tokens/duration {rate}"
                )
            rate = _upstream_rate(row, variant, lane)
            detail.append(
                {
                    "row": row,  # raw details preserved for parent finalize
                    "rate": rate,
                    "completed": completed,
                }
            )
    if not detail:
        raise Reject(f"{variant}: upstream detail file {path} is empty")
    return detail


def upstream_median_rate(
    detail: List[Dict[str, Any]], variant: str, rounds: int = 5, *, lane: str = "short"
) -> float:
    if len(detail) != rounds:
        raise Reject(
            f"{variant}: upstream has {len(detail)} measured rounds, expected {rounds}"
        )
    rows = [item["row"] for item in detail]
    repeats = [row.get("repeat") for row in rows]
    if any(type(value) is not int for value in repeats) or repeats != list(
        range(rounds)
    ):
        raise Reject(
            f"{variant}: {lane} repeat identities must be ordered unique 0..{rounds - 1}"
        )
    identities = [row.get("native_result_sha256") for row in rows]
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in identities
    ):
        raise Reject(
            f"{variant}: {lane} native result identities are missing or malformed"
        )
    if len(set(identities)) != rounds:
        raise Reject(f"{variant}: {lane} reuses the same native result artifact")
    # Identical fixed input hashes and usage counts are intentional across
    # independent rounds; they are not independent-measurement identities.
    return median([_upstream_rate(row, variant, lane) for row in rows])


# ---------------------------------------------------------------------------
# manifest parity
# ---------------------------------------------------------------------------


def check_parity(base_m: Dict[str, Any], cand_m: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in base_m]
    missing += [k for k in REQUIRED_MANIFEST_KEYS if k not in cand_m]
    if missing:
        raise Reject(f"manifests missing required detail: {sorted(set(missing))}")
    mismatch = [
        k
        for k in REQUIRED_MANIFEST_KEYS
        if json.dumps(base_m[k], sort_keys=True)
        != json.dumps(cand_m[k], sort_keys=True)
    ]
    # LEVER_KEY itself is not in REQUIRED_MANIFEST_KEYS; any required-key
    # mismatch is fatal. The lever is declared separately.
    if mismatch:
        raise Reject(
            f"manifest parity mismatch in {mismatch} (only an explicitly "
            f"declared {LEVER_KEY} may differ)"
        )
    lever_b, lever_c = base_m.get(LEVER_KEY), cand_m.get(LEVER_KEY)
    if not lever_c or not isinstance(lever_c, str):
        raise Reject(
            "candidate manifest must declare an explicit 'lever' string "
            "(the single speculative/profile lever being tested)"
        )
    if mixed := _mixed_epoch_guard(lever_b, lever_c):
        raise Reject(mixed)
    contexts = [m.get("runtime_context") for m in (base_m, cand_m)]
    if any(value is not None for value in contexts):
        import dataclasses
        from runtime import profile_from_dict

        profiles = []
        for value in contexts:
            if not isinstance(value, dict) or "profile" not in value:
                raise Reject("runtime profile snapshot missing on one side")
            profiles.append(dataclasses.asdict(profile_from_dict(value["profile"])))
        for profile in profiles:
            profile.pop("raw")
        if lever_c == "nextn":
            if profiles[0]["mode"] != "ar" or profiles[1]["mode"] != "nextn":
                raise Reject("declared NEXTN comparison does not change AR to NEXTN")
            allowed = {"mode", "speculative_steps"}
        else:
            field = {
                "vision.backend": "vision_backend",
                "vision.cuda_graph": "vision_cuda_graph",
            }.get(lever_c, lever_c)
            allowed = {field}
            if field not in profiles[0] or profiles[0][field] == profiles[1][field]:
                raise Reject("declared profile lever is absent or unchanged")
        changed = {key for key in profiles[0] if profiles[0][key] != profiles[1][key]}
        if not changed <= allowed:
            raise Reject(
                "undeclared runtime profile changes: " + str(sorted(changed - allowed))
            )
    for key in ("inputs", "benchmark_module_sha256", "capture_adapter_sha256"):
        if key in base_m or key in cand_m:
            if base_m.get(key) != cand_m.get(key):
                raise Reject("frozen comparison provenance differs: " + key)
    return {"lever": {"baseline": lever_b or "none", "candidate": lever_c}}


def manifest_fingerprint(manifest):
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_manifest(manifest):
    """Validate the shared pre-traffic promotion identity contract."""
    if not isinstance(manifest, dict):
        raise Reject("promotion manifest must be an object")
    if (
        manifest.get("schema") != "qwen38fn.promotion.v1"
        or manifest.get("model_id") != "nvidia/Qwen3.8-Flash-Next-NVFP4"
    ):
        raise Reject("promotion manifest schema/model identity missing")
    fingerprint = manifest_fingerprint(manifest)
    if manifest.get("manifest_sha256") != fingerprint:
        raise Reject("promotion manifest fingerprint mismatch")
    for key, pattern in (
        ("model_sha", r"[0-9a-f]{40}"),
        ("source_tree", r"[0-9a-f]{40}"),
        ("image_id", r"sha256:[0-9a-f]{64}"),
        ("tokenizer", r"[0-9a-f]{64}"),
        ("input_token_sha", r"[0-9a-f]{64}"),
    ):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(
            pattern, manifest[key]
        ):
            raise Reject("malformed promotion identity: " + key)
    if type(manifest.get("concurrency")) is not int or manifest["concurrency"] != 1:
        raise Reject("this promotion protocol requires C1")
    if (
        type(manifest.get("context")) is not int
        or type(manifest.get("total_pool")) is not int
        or not 32768 <= manifest["context"] <= 262144
        or manifest["total_pool"] < manifest["context"]
    ):
        raise Reject("invalid context/pool envelope")
    if not isinstance(manifest.get("inputs"), dict):
        raise Reject("promotion manifest inputs must be an object")
    return fingerprint


def check_evidence_binding(manifest, rows, lane):
    """Promotion requires producer-bound metadata, not two matching documents."""
    fingerprint = validate_manifest(manifest)
    inputs = manifest["inputs"].get(lane)
    if not isinstance(inputs, dict) or not inputs:
        raise Reject("promotion manifest is missing frozen lane inputs")
    measured = [row for row in rows if row.get("warmup") is not True]
    observed_cases = {row.get("case_id", row.get("case", "batch")) for row in measured}
    if set(inputs) != observed_cases or (lane == "vision" and len(inputs) != 43):
        raise Reject("frozen input set is incomplete or differs from measured rows")
    for row in measured:
        if row.get("epoch_verified") is not True:
            raise Reject("row has no verified runtime epoch")
        binding = row.get("_evidence")
        case = row.get("case_id", row.get("case", "batch"))
        if (
            not isinstance(binding, dict)
            or binding.get("manifest_sha256") != fingerprint
            or binding.get("lane") != lane
        ):
            raise Reject("row is not bound to its producer manifest/lane")
        if (
            binding.get("model_id") != manifest["model_id"]
            or row.get("model", manifest["model_id"]) != manifest["model_id"]
        ):
            raise Reject("row model differs from manifest")
        expected = inputs.get(case)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or binding.get("input_sha256") != expected
        ):
            raise Reject("row input differs from frozen manifest")
        if "input_sha256" in row and row["input_sha256"] != expected:
            raise Reject("row input hash disagrees with its bound evidence")
        if lane in ("short", "medium"):
            if row.get("tag") != f"{fingerprint}/{lane}/r{row.get('repeat')}":
                raise Reject("native round tag differs from bound manifest/lane/repeat")
            payloads = manifest.get("requests", {}).get(lane)
            if not isinstance(payloads, list) or len(payloads) != 8:
                raise Reject("upstream promotion requires the frozen request values")
            hashes = [
                hashlib.sha256(
                    json.dumps(
                        p, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest()
                for p in payloads
            ]
            if row.get("request_hashes") != hashes:
                raise Reject(
                    "actual upstream request sequence differs from the frozen corpus"
                )


def _norm_lever(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return None if not s or s.lower() in ("none", "no", "off", "baseline") else s


def _mixed_epoch_guard(lever_b: Any, lever_c: Any) -> Optional[str]:
    """Refuse mixed epochs: the candidate lever must be singular and declared.

    A clean A/B is: unlisted/no-lever baseline (or the same lever on both
    sides) vs exactly one declared candidate lever. Two DIFFERENT declared
    levers means the two sides ran different optimizations — mixed epochs.
    """
    b, c = _norm_lever(lever_b), _norm_lever(lever_c)
    if b and c and b != c:
        return (
            f"baseline lever {lever_b!r} != candidate lever {lever_c!r}: this is a "
            "comparison between different levers (mixed epochs), not a lever A/B"
        )
    return None


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def compare(
    *,
    base_rows: List[Dict[str, Any]],
    cand_rows: List[Dict[str, Any]],
    base_manifest: Dict[str, Any],
    cand_manifest: Dict[str, Any],
    base_upstream: Optional[List[Dict[str, Any]]] = None,
    cand_upstream: Optional[List[Dict[str, Any]]] = None,
    base_medium: Optional[List[Dict[str, Any]]] = None,
    cand_medium: Optional[List[Dict[str, Any]]] = None,
    base_vision: Optional[List[Dict[str, Any]]] = None,
    cand_vision: Optional[List[Dict[str, Any]]] = None,
    repeats: int = 5,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"verdict": "NOT_OPTIMIZED", "gates": {}}

    parity = check_parity(base_manifest, cand_manifest)
    report["gates"]["parity"] = parity
    # Revalidate public-API inputs, not only inputs arriving through main().
    base_rows = measured_prose_rows(base_rows, "baseline")
    cand_rows = measured_prose_rows(cand_rows, "candidate")
    missing = [
        name
        for name, pair in {
            "short": (base_upstream, cand_upstream),
            "medium": (base_medium, cand_medium),
            "vision": (base_vision, cand_vision),
        }.items()
        if any(x is None for x in pair)
    ]
    report["gates"]["complete_lanes"] = {"pass": not missing, "missing": missing}

    # ---- prose (C1): per-case median over ALL repeats, >=5% each ----------
    prose_gates: Dict[str, Any] = {}
    try:
        b_by = check_counts(base_rows, "baseline", repeats)
        c_by = check_counts(cand_rows, "candidate", repeats)
        for case in (
            "db_index_write_read",
            "city_heat_water",
            "compression_vs_random_access",
        ):
            b_rates = [r["_rate"] for r in b_by[case]]
            c_rates = [r["_rate"] for r in c_by[case]]
            b_med, c_med = median(b_rates), median(c_rates)
            if b_med <= 0 or math.isnan(b_med) or math.isnan(c_med):
                raise Reject(f"{case}: zero/NaN median rate")
            gain = (c_med - b_med) / b_med
            prose_gates[case] = {
                "baseline_median": b_med,
                "candidate_median": c_med,
                "improvement": gain,
                "pass": gain >= IMPROVEMENT_THRESHOLD,
            }
            if gain < IMPROVEMENT_THRESHOLD:
                raise Reject(
                    f"{case}: median improvement {gain:.4f} < {IMPROVEMENT_THRESHOLD:.2f}"
                )
    except Reject as exc:
        prose_gates["rejected"] = str(exc)
        report["gates"]["prose_3case"] = prose_gates
        return report
    report["gates"]["prose_3case"] = prose_gates

    # ---- upstream short/medium (when provided) ----------------------------
    if base_upstream is not None or cand_upstream is not None:
        if base_upstream is None or cand_upstream is None:
            raise Reject("upstream short/medium must be provided for BOTH variants")
        try:
            b_short = upstream_median_rate(base_upstream, "baseline")
            c_short = upstream_median_rate(cand_upstream, "candidate")
            gain = (c_short - b_short) / b_short
            if gain < IMPROVEMENT_THRESHOLD:
                raise Reject(f"short: median improvement {gain:.4f} < 0.05")
            report["gates"]["upstream_short"] = {
                "baseline_median": b_short,
                "candidate_median": c_short,
                "improvement": gain,
                "pass": True,
            }
        except Reject as exc:
            report["gates"]["upstream_short"] = {"rejected": str(exc)}
            return report

    if base_medium is not None or cand_medium is not None:
        if base_medium is None or cand_medium is None:
            raise Reject("medium rows must be provided for BOTH variants")
        b_med = upstream_median_rate(base_medium, "baseline medium", lane="medium")
        c_med = upstream_median_rate(cand_medium, "candidate medium", lane="medium")
        gain = (c_med - b_med) / b_med
        report["gates"]["upstream_medium"] = {
            "baseline_median": b_med,
            "candidate_median": c_med,
            "improvement": gain,
            "pass": gain >= IMPROVEMENT_THRESHOLD,
        }
        if gain < IMPROVEMENT_THRESHOLD:
            return report

    # ---- paired vision gate ---------------------------------------------
    if base_vision is not None or cand_vision is not None:
        try:
            gate = vision_ttft_gate(base_vision, cand_vision)
            report["gates"]["vision_p95_ttft"] = gate
            if not gate["pass"]:
                return report
        except Reject as exc:
            report["gates"]["vision_p95_ttft"] = {"rejected": str(exc)}
            return report

    if missing:
        return report
    if cand_manifest.get("lever_kind") == "vision":
        metric = cand_manifest.get("vision_metric")
        if metric not in ("wall_s", "ttft_s"):
            raise Reject("vision lever requires an observable wall_s or ttft_s metric")
        b = median(
            [
                require_finite_positive(r[metric], metric)
                for r in base_vision
                if not r.get("warmup")
            ]
        )
        c = median(
            [
                require_finite_positive(r[metric], metric)
                for r in cand_vision
                if not r.get("warmup")
            ]
        )
        gain = (b - c) / b
        report["gates"]["vision_lever_gain"] = {
            "metric": metric,
            "improvement": gain,
            "pass": gain >= VISION_LEVER_GAIN,
        }
        if gain < VISION_LEVER_GAIN:
            return report
    try:
        for manifest, prose, short, medium, vision in (
            (base_manifest, base_rows, base_upstream, base_medium, base_vision),
            (cand_manifest, cand_rows, cand_upstream, cand_medium, cand_vision),
        ):
            for lane, rows in (
                ("prose", prose),
                ("short", [d["row"] for d in short]),
                ("medium", [d["row"] for d in medium]),
                ("vision", vision),
            ):
                check_evidence_binding(manifest, rows, lane)
    except Reject as exc:
        report["gates"]["evidence_binding"] = {"pass": False, "rejected": str(exc)}
        return report
    report["gates"]["evidence_binding"] = {"pass": True}
    report["verdict"] = "PASS"
    report["label"] = "prose E2E (end-to-end request, usage-based token rate)"
    return report


def vision_ttft_gate(
    base_vision: Optional[List[Dict[str, Any]]],
    cand_vision: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    if base_vision is None or cand_vision is None:
        raise Reject("vision rows must be provided for BOTH variants")

    def ttfts(rows: List[Dict[str, Any]], variant: str) -> List[float]:
        out = []
        for i, r in enumerate(rows):
            if r.get("warmup"):
                continue
            if r.get("error") is not None:
                raise Reject(f"{variant} vision row {i} has error {r['error']}")
            if r.get("valid") is not True or r.get("finish_reason") != "stop":
                raise Reject(
                    f"{variant} vision row {i} lacks successful semantic/finish evidence"
                )
            ttft = r.get("ttft_s")
            if ttft is None:
                raise Reject(f"{variant} vision row {i} missing ttft_s")
            out.append(
                require_finite_positive(ttft, f"{variant} vision row {i} ttft_s")
            )
        if len(out) < 2:
            raise Reject(f"{variant}: too few vision rows ({len(out)}) for p95")
        return out

    b = ttfts(base_vision, "baseline")
    c = ttfts(cand_vision, "candidate")

    def identities(rows):
        selected = [r for r in rows if not r.get("warmup")]
        result = {}
        for row in selected:
            key = (row.get("case_id"), row.get("repeat"))
            if not isinstance(key[0], str) or type(key[1]) is not int or key in result:
                raise Reject(
                    "vision rows have missing/duplicate case/repeat identities"
                )
            value = row.get("input_sha256")
            if not isinstance(value, str) or len(value) != 64:
                raise Reject("vision input byte hash is missing")
            result[key] = value
        return result

    if identities(base_vision) != identities(cand_vision):
        raise Reject("vision case/input pairing mismatch")
    b95, c95 = percentile(b, 0.95), percentile(c, 0.95)
    regression = (c95 - b95) / b95
    return {
        "baseline_p95_ttft": b95,
        "candidate_p95_ttft": c95,
        "regression": regression,
        "pass": regression <= VISION_TTFT_REGRESSION_LIMIT,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_jsonl(name: str) -> List[Dict[str, Any]]:
    p = Path(name)
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="fail-closed C1 comparison reducer")
    p.add_argument("--baseline", required=True, help="baseline JSONL (bench_real rows)")
    p.add_argument(
        "--candidate", required=True, help="candidate JSONL (bench_real rows)"
    )
    p.add_argument("--baseline-manifest", required=True)
    p.add_argument("--candidate-manifest", required=True)
    p.add_argument("--baseline-upstream", help="upstream bench_serving short JSONL")
    p.add_argument("--candidate-upstream", help="upstream bench_serving short JSONL")
    p.add_argument("--baseline-medium", help="upstream bench_serving medium JSONL")
    p.add_argument("--candidate-medium", help="upstream bench_serving medium JSONL")
    p.add_argument("--baseline-vision", help="vision JSONL rows (baseline)")
    p.add_argument("--candidate-vision", help="vision JSONL rows (candidate)")
    p.add_argument("--output", help="optional verdict JSON output path")
    args = p.parse_args(argv)

    try:
        report = compare(
            base_rows=measured_prose_rows(
                load_rows(Path(args.baseline), "baseline"), "baseline"
            ),
            cand_rows=measured_prose_rows(
                load_rows(Path(args.candidate), "candidate"), "candidate"
            ),
            base_manifest=_load_json(args.baseline_manifest),
            cand_manifest=_load_json(args.candidate_manifest),
            base_upstream=load_upstream_detail(Path(args.baseline_upstream), "baseline")
            if args.baseline_upstream
            else None,
            cand_upstream=load_upstream_detail(
                Path(args.candidate_upstream), "candidate"
            )
            if args.candidate_upstream
            else None,
            base_medium=load_upstream_detail(
                Path(args.baseline_medium), "baseline medium", lane="medium"
            )
            if args.baseline_medium
            else None,
            cand_medium=load_upstream_detail(
                Path(args.candidate_medium), "candidate medium", lane="medium"
            )
            if args.candidate_medium
            else None,
            base_vision=_load_jsonl(args.baseline_vision)
            if args.baseline_vision
            else None,
            cand_vision=_load_jsonl(args.candidate_vision)
            if args.candidate_vision
            else None,
        )
    except Reject as exc:
        report = {"verdict": "NOT_OPTIMIZED", "rejected": str(exc)}
    except (OSError, json.JSONDecodeError) as exc:
        report = {"verdict": "NOT_OPTIMIZED", "rejected": f"io/parse: {exc}"}

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
