"""Compile the validated profile + locks into an exact server argv.

The parent owns the image; this module only defines the command contract.
``build_argv`` returns a pure argv array (never a shell string) plus the
exact environment mapping the server should run under.

Status: NOT QUALIFIED. Flag names/values mirror the intended contract and
must still be validated against the actual image CLI by the parent before
any launch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import Profile, ProfileError, load_profile, profile_from_dict

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_PATH = "/model"
SERVE_HOST = "0.0.0.0"
SERVE_PORT = 30000
QUANTIZATION = "modelopt_mixed"
PLE_DIR_ROOT = "/cache/ple"

# env var names the entrypoint owns and sets for the server process
ENV_VARS_OWNED = (
    "SGLANG_VIT_ENABLE_CUDA_GRAPH",
    "SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED",
    "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB",
    "SGLANG_QWEN4_PLE_FILE_PREFETCH",
    "SGLANG_QSA_KV_AMAX_CALIB",
    "SGLANG_QSA_KV_MODEL_REVISION",
    "SGLANG_QSA_KV_SCALE_SIDECAR",
    "SGLANG_QSA_KV_SCALE_SHA256",
    "R0B0TLAB_PLE_PREPARED_PATH",
    "R0B0TLAB_PLE_PREPARED_SHA256",
    "TMPDIR",
)

# image-owned PLE preparation plan; installed by the controller next to the
# entrypoint package. There is deliberately no CLI override for this path.
PLE_PLAN_PATH = str(Path(__file__).resolve().parents[1] / "locks" / "ple.json")
PLE_PLAN_SCHEMA = "qwen38fn.ple-plan.v1"
PLE_PLAN_MAX_BYTES = 1024 * 1024
PLE_PLAN_KEYS = frozenset(
    {
        "schema",
        "model",
        "source",
        "tensor_prefix",
        "dtype",
        "shard_count",
        "rows_per_shard",
        "row_width",
        "table",
        "scale",
        "shards",
    }
)
PLE_DTYPE = "F8_E4M3"
PLE_SOURCE_NAME = "model-fp8-mtp-ple.safetensors"
PLE_TENSOR_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
PLE_SHARD_COUNT = 128
PLE_ROWS_PER_SHARD = 2500012
PLE_ROW_WIDTH = 160
PLE_TABLE_SHAPE = [320001536, PLE_ROW_WIDTH]
PLE_TABLE_SIZE = 51200245760

_SHA256_RE = r"[0-9a-f]{64}"


class PLEPreparationError(ValueError):
    """Raised when verified PLE preparation cannot be bound or trusted."""


def _full_sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ProfileError("sources.model.sha must be a full lowercase Git SHA")
    return value


def _checked_sources(sources: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(sources, dict) or not isinstance(sources.get("model"), dict):
        raise ProfileError("sources.model must be an object")
    if "env" in sources:
        raise ProfileError("runtime environment overrides are not supported")
    model = sources["model"]
    if model.get("id") != MODEL_ID:
        raise ProfileError("unexpected model identity")
    _full_sha(model.get("sha"))
    return model


def _checked_profile(profile: Profile) -> Profile:
    if not isinstance(profile, Profile):
        raise ProfileError("validated Profile required")
    checked = profile_from_dict(profile.raw)
    if any(
        getattr(profile, field.name) != getattr(checked, field.name)
        for field in dataclasses.fields(Profile)
        if field.name != "raw"
    ):
        raise ProfileError("profile fields differ from its validated source")
    return checked


def ple_dir(model_sha: str) -> str:
    return "%s/%s" % (PLE_DIR_ROOT, _full_sha(model_sha))


def build_env(profile: Profile, sources: Dict[str, Any]) -> Dict[str, str]:
    """Writable cache paths, non-overridable safety settings, and cleared
    PLE handoff slots (filled with fresh verified values just before exec)."""
    profile = _checked_profile(profile)
    model = _checked_sources(sources)
    env = {
        "HOME": "/cache/home",
        "TMPDIR": "/cache/tmp",
        "XDG_CACHE_HOME": "/cache/xdg",
        "HF_HOME": "/cache/hf",
        "SGLANG_CACHE_DIR": "/cache/jit",
        "SGLANG_VIT_ENABLE_CUDA_GRAPH": ("1" if profile.vision_cuda_graph else "0"),
        "SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED": "1",
        "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": str(profile.ple_rss_gib),
        "SGLANG_QWEN4_PLE_FILE_PREFETCH": "1",
        "SGLANG_QSA_KV_AMAX_CALIB": "1" if profile.kv_calibration_mode == "collect" else "0",
        "SGLANG_QSA_KV_MODEL_REVISION": model["sha"],
        "SGLANG_QSA_KV_SCALE_SIDECAR": (
            "/opt/r0b0tlab/kv-calibration/%s/%s" % (model["sha"], profile.kv_scale_file)
            if profile.kv_calibration_mode == "frozen" else ""
        ),
        "SGLANG_QSA_KV_SCALE_SHA256": profile.kv_scale_sha256 or "",
        "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK": "0",
        "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH": "0",
        "R0B0TLAB_PLE_PREPARED_PATH": "",
        "R0B0TLAB_PLE_PREPARED_SHA256": "",
        "MAX_JOBS": "1",
        "FLASHINFER_NVCC_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return env


def prepare_jit_temp(path: str) -> None:
    """Create a private compiler scratch directory; refuse link/mode drift."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError(
                "JIT temporary directory must be private and runtime-owned"
            )
    finally:
        os.close(fd)


# ------------------------------------------------------------- PLE plan
#
# The controller installs the derived plan at ``locks/ple.json`` inside the
# packaged image. The wrapper treats it as data: it re-parses it under the
# same strict JSON rules as the source lock, re-checks the frozen geometry
# constants, and independently re-binds model revision and PLE source file
# identity against the checked sources lock before the installed SGLang
# core is imported. Geometry/file integrity itself is owned by the core
# validator; the wrapper never downcasts or reinterprets it.


def _reject_constant(token: str) -> Any:
    raise PLEPreparationError("nonfinite JSON number %r is not allowed" % (token,))


def _finite_plan_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise PLEPreparationError("nonfinite plan number")
    return value


def _plan_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PLEPreparationError("duplicate plan field %r" % (key,))
        result[key] = value
    return result


def _plan_str(plan, key):
    value = plan.get(key)
    if not isinstance(value, str) or not value:
        raise PLEPreparationError("plan.%s must be a non-empty string" % (key,))
    return value


def _plan_sha256(plan, key):
    value = _plan_str(plan, key)
    if not re.fullmatch(_SHA256_RE, value):
        raise PLEPreparationError("plan.%s must be a lowercase 64-hex digest" % (key,))
    return value


def _plan_int(plan, key, minimum=0):
    value = plan.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PLEPreparationError("plan.%s must be an integer" % (key,))
    if value < minimum:
        raise PLEPreparationError("plan.%s below minimum %d" % (key, minimum))
    return value


def _plan_object(plan, key):
    value = plan.get(key)
    if not isinstance(value, dict):
        raise PLEPreparationError("plan.%s must be an object" % (key,))
    return value


def _bind_plan_model(plan):
    model = _plan_object(plan, "model")
    if set(model) != {"id", "revision"}:
        raise PLEPreparationError("plan.model must have exactly id and revision")
    if model["id"] != MODEL_ID or not isinstance(model["id"], str):
        raise PLEPreparationError("plan.model.id is not the frozen model identity")
    revision = model["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PLEPreparationError("plan.model.revision must be a full Git SHA")
    return revision


def _bind_plan_source(plan, revision):
    source = _plan_object(plan, "source")
    if set(source) != {"name", "size", "sha256", "header_sha256", "data_start"}:
        raise PLEPreparationError(
            "plan.source must have exactly name/size/sha256/header_sha256/data_start"
        )
    if source["name"] != PLE_SOURCE_NAME:
        raise PLEPreparationError("plan.source.name is not the frozen PLE shard name")
    size = _plan_int(source, "size", 1)
    sha256 = _plan_sha256(source, "sha256")
    _plan_sha256(source, "header_sha256")
    _plan_int(source, "data_start")
    return {"name": source["name"], "size": size, "sha256": sha256}


def _bind_plan_geometry(plan):
    tensor_prefix = _plan_str(plan, "tensor_prefix")
    if tensor_prefix != PLE_TENSOR_PREFIX:
        raise PLEPreparationError("plan.tensor_prefix is not the frozen PLE prefix")
    if _plan_str(plan, "dtype") != PLE_DTYPE:
        raise PLEPreparationError("plan.dtype is not the frozen PLE dtype")
    if _plan_int(plan, "shard_count") != PLE_SHARD_COUNT:
        raise PLEPreparationError("plan.shard_count is not the frozen PLE shard count")
    if _plan_int(plan, "rows_per_shard") != PLE_ROWS_PER_SHARD:
        raise PLEPreparationError("plan.rows_per_shard is not the frozen row count")
    if _plan_int(plan, "row_width") != PLE_ROW_WIDTH:
        raise PLEPreparationError("plan.row_width is not the frozen row width")
    table = _plan_object(plan, "table")
    if set(table) != {"name", "shape", "size", "sha256"}:
        raise PLEPreparationError("plan.table must have exactly name/shape/size/sha256")
    if not isinstance(table["name"], str) or not table["name"]:
        raise PLEPreparationError("plan.table.name must be a non-empty string")
    shape = table["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(d, bool) or not isinstance(d, int) for d in shape)
    ):
        raise PLEPreparationError("plan.table.shape must be [rows, row_width] integers")
    if shape != PLE_TABLE_SHAPE:
        raise PLEPreparationError("plan.table.shape is not the frozen PLE table shape")
    if _plan_int(table, "size") != PLE_TABLE_SIZE:
        raise PLEPreparationError("plan.table.size is not the frozen PLE table size")
    _plan_sha256(table, "sha256")
    scale = _plan_object(plan, "scale")
    if set(scale) != {"name", "offset", "size", "sha256"}:
        raise PLEPreparationError(
            "plan.scale must have exactly name/offset/size/sha256"
        )
    if not isinstance(scale["name"], str) or not scale["name"]:
        raise PLEPreparationError("plan.scale.name must be a non-empty string")
    _plan_int(scale, "offset")
    _plan_int(scale, "size", 1)
    _plan_sha256(scale, "sha256")


def _bind_plan_inventory(plan, sources):
    shards = plan.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PLEPreparationError("plan.shards must be a non-empty list")
    seen = set()
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {
            "id",
            "name",
            "offset",
            "size",
            "sha256",
        }:
            raise PLEPreparationError(
                "each plan.shards entry must have exactly id/name/offset/size/sha256"
            )
        shard_id = shard["id"]
        if isinstance(shard_id, bool) or not isinstance(shard_id, int) or shard_id < 0:
            raise PLEPreparationError("plan.shards[].id must be a non-negative integer")
        if shard_id in seen:
            raise PLEPreparationError("duplicate plan.shards id %r" % (shard_id,))
        seen.add(shard_id)
        if not isinstance(shard["name"], str) or not shard["name"]:
            raise PLEPreparationError("plan.shards[].name must be a non-empty string")
        _plan_int(shard, "offset")
        _plan_int(shard, "size", 1)
        _plan_sha256(shard, "sha256")
    if seen != set(range(PLE_SHARD_COUNT)):
        raise PLEPreparationError(
            "plan.shards must cover every PLE shard id exactly once"
        )


def bind_ple_plan(plan: Dict[str, Any], sources: Dict[str, Any]) -> Dict[str, Any]:
    """Independently re-check the parsed plan and bind it to the checked
    sources lock. Returns the bound facts used by the exec-path handoff."""
    if not isinstance(plan, dict) or set(plan) != PLE_PLAN_KEYS:
        raise PLEPreparationError("plan must have exactly the versioned key set")
    if plan["schema"] != PLE_PLAN_SCHEMA:
        raise PLEPreparationError("unexpected plan schema %r" % (plan.get("schema"),))
    model = _checked_sources(sources)
    revision = _bind_plan_model(plan)
    if revision != model["sha"]:
        raise PLEPreparationError(
            "plan revision %s does not match frozen model revision %s"
            % (revision, model["sha"])
        )
    bound = _bind_plan_source(plan, revision)
    _bind_plan_geometry(plan)
    _bind_plan_inventory(plan, sources)
    inventory = model.get("files")
    if not isinstance(inventory, list) or any(
        not isinstance(entry, dict) for entry in inventory
    ):
        raise PLEPreparationError("sources.model.files must be a list of objects")
    rows = [entry for entry in inventory if entry.get("path") == bound["name"]]
    if len(rows) != 1:
        raise PLEPreparationError(
            "sources.model.files must contain exactly one %r row" % (bound["name"],)
        )
    row = rows[0]
    if (
        type(row.get("size")) is not int
        or row.get("sha256") != bound["sha256"]
        or row["size"] != bound["size"]
    ):
        raise PLEPreparationError(
            "plan source identity does not match the frozen sources row"
        )
    return {
        "source": bound,
        "table_sha256": plan["table"]["sha256"],
        "table_size": plan["table"]["size"],
    }


def read_ple_plan(path: str) -> Dict[str, Any]:
    """Strictly parse the image-owned plan (bounded size, no duplicate
    fields, no nonfinite numbers)."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(PLE_PLAN_MAX_BYTES + 1)
    except OSError as error:
        raise PLEPreparationError("PLE plan unreadable: %s" % (error,))
    if len(raw) > PLE_PLAN_MAX_BYTES:
        raise PLEPreparationError("PLE plan exceeds 1 MiB")
    try:
        plan = json.loads(
            raw,
            object_pairs_hook=_plan_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_plan_float,
        )
    except PLEPreparationError:
        raise
    except (ValueError, RecursionError) as error:
        raise PLEPreparationError("PLE plan is not valid strict JSON: %s" % (error,))
    if not isinstance(plan, dict):
        raise PLEPreparationError("PLE plan must be a JSON object")
    return plan


def validate_prepared_result(
    result: Any, expected_dir: str, expected_copied_bytes: int
) -> Dict[str, Any]:
    """Check the core prepare_cache receipt shape before it may reach exec."""
    if not isinstance(result, dict) or set(result) != {
        "receipt_path",
        "receipt_sha256",
        "mode",
        "copied_bytes",
        "elapsed_seconds",
    }:
        raise PLEPreparationError(
            "prepared result must have exactly receipt_path/receipt_sha256/mode/"
            "copied_bytes/elapsed_seconds"
        )
    receipt = result["receipt_path"]
    if receipt != expected_dir + "/prepared.json":
        raise PLEPreparationError(
            "prepared receipt path must be %s" % (expected_dir + "/prepared.json",)
        )
    digest = result["receipt_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(_SHA256_RE, digest):
        raise PLEPreparationError("prepared digest must be a lowercase 64-hex digest")
    if result["mode"] not in ("reuse", "materialized"):
        raise PLEPreparationError("prepared mode must be 'reuse' or 'materialized'")
    copied = result["copied_bytes"]
    if isinstance(copied, bool) or not isinstance(copied, int) or copied < 0:
        raise PLEPreparationError(
            "prepared copied_bytes must be a non-negative integer"
        )
    required_bytes = 0 if result["mode"] == "reuse" else expected_copied_bytes
    if copied != required_bytes:
        raise PLEPreparationError(
            "prepared copied_bytes does not match preparation mode"
        )
    elapsed = result["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise PLEPreparationError("prepared elapsed_seconds must be a real number")
    try:
        finite = math.isfinite(elapsed)
    except OverflowError:
        finite = False
    if not finite or elapsed < 0:
        raise PLEPreparationError("prepared elapsed_seconds must be finite and >= 0")
    return result


def prepare_ple_cache(sources: Dict[str, Any]) -> Dict[str, Any]:
    """Read the image-owned plan, bind it to the checked sources lock, run
    the installed core preparation, and validate its receipt. Import of the
    core happens only here, on the exec path."""
    plan = read_ple_plan(PLE_PLAN_PATH)
    bound = bind_ple_plan(plan, sources)
    model_sha = _checked_sources(sources)["sha"]
    directory = ple_dir(model_sha)
    try:
        from sglang.srt.models.qwen4_exp_ple_cache import prepare_cache, validate_plan
    except ImportError as error:
        raise PLEPreparationError(
            "installed PLE preparation core is unavailable: %s" % (error,)
        )
    if not callable(prepare_cache) or not callable(validate_plan):
        raise PLEPreparationError(
            "installed PLE preparation core has no callable prepare_cache"
        )
    try:
        validated = validate_plan(plan)
        if not isinstance(validated, dict) or validated != plan:
            raise PLEPreparationError("core validator changed the image-owned plan")
        result = prepare_cache(MODEL_PATH, directory, validated)
    except PLEPreparationError:
        raise
    except Exception as error:  # core failure must refuse, never exec stale
        raise PLEPreparationError("PLE preparation failed: %s" % (error,))
    return validate_prepared_result(result, directory, bound["table_size"])


def _kv_cache_flag(dtype: str) -> str:
    return dtype


def build_argv(profile: Profile, sources: Dict[str, Any]) -> List[str]:
    """Exact server argv for this profile on exactly one GB10."""
    profile = _checked_profile(profile)
    model = _checked_sources(sources)
    model_sha = model["sha"]

    argv: List[str] = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--served-model-name",
        MODEL_ID,
        "--tp-size",
        "1",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--host",
        SERVE_HOST,
        "--port",
        str(SERVE_PORT),
        "--dtype",
        "bfloat16",
        "--quantization",
        QUANTIZATION,
        "--load-format",
        "safetensors",
        "--ple-offload-embedding",
        "--ple-offload-backend",
        "file",
        "--ple-offload-dir",
        ple_dir(model_sha),
        "--fp4-gemm-backend",
        "flashinfer_cutlass",
        "--moe-runner-backend",
        profile.moe_backend,
        "--attention-backend",
        "triton",
        "--mamba-ssm-dtype",
        "float32",
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
        "--linear-attn-verify-backend",
        "triton",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-bs-decode",
        *[str(n) for n in (1, 2, 4, 8) if n <= profile.max_running_requests],
        "--page-size",
        "64",
        "--enable-metrics",
        "--watchdog-timeout",
        "1800",
        "--context-length",
        str(profile.context_length),
        "--max-total-tokens",
        str(profile.max_total_tokens),
        "--max-running-requests",
        str(profile.max_running_requests),
        "--chunked-prefill-size",
        str(profile.chunked_prefill_size),
        "--max-prefill-tokens",
        str(profile.chunked_prefill_size),
        "--mem-fraction-static",
        str(profile.mem_fraction_static),
        "--kv-cache-dtype",
        _kv_cache_flag(profile.kv_cache_dtype),
        "--max-mamba-cache-size",
        str(profile.max_mamba_cache_size),
        "--mm-attention-backend",
        profile.vision_backend,
        "--mm-processor-worker-num",
        str(profile.mm_processor_worker_num),
    ]
    if profile.mode == "nextn":
        argv += [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            str(profile.speculative_steps),
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(profile.speculative_steps + 1),
            "--speculative-draft-model-quantization",
            QUANTIZATION,
            "--speculative-moe-runner-backend",
            "triton",
            "--speculative-draft-kv-cache-dtype",
            _kv_cache_flag(profile.draft_kv_cache_dtype or profile.kv_cache_dtype),
        ]
    return argv


def build_all(profile: Profile, sources: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "argv": build_argv(profile, sources),
        "env": build_env(profile, sources),
    }


def _source_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError("duplicate source key")
        result[key] = value
    return result


def _source_constant(value):
    raise ProfileError("nonfinite source number")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Single-GB10 SGLang container entrypoint"
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--sources", default="/opt/r0b0tlab/locks/sources.json")
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        with open(args.sources, "rb") as handle:
            raw = handle.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ProfileError("source lock exceeds 4 MiB")
        sources = json.loads(
            raw, object_pairs_hook=_source_pairs, parse_constant=_source_constant
        )
        launch = build_all(profile, sources)
        if args.print_only:
            print(json.dumps(launch, sort_keys=True, allow_nan=False))
            return 0
        # The host launcher provides /cache as a writable local-NVMe bind.
        # Libraries create their own subdirectories; no host path is mutated here.
        environment = dict(os.environ)
        environment.update(launch["env"])
        prepare_jit_temp(environment["TMPDIR"])
        # Verified PLE preparation: refuse before exec rather than hand the
        # server a stale or unverified cache. Freshly bound values overwrite
        # the cleared slots from build_env; the env actually exec'd is the
        # only source of the handoff.
        prepared = prepare_ple_cache(sources)
        environment["R0B0TLAB_PLE_PREPARED_PATH"] = prepared["receipt_path"]
        environment["R0B0TLAB_PLE_PREPARED_SHA256"] = prepared["receipt_sha256"]
        print(
            json.dumps(
                {
                    "model_sha": sources["model"]["sha"],
                    "profile_sha256": profile.digest(),
                    "argv": launch["argv"],
                    "ple": {
                        "mode": prepared["mode"],
                        "receipt_path": prepared["receipt_path"],
                        "receipt_sha256": prepared["receipt_sha256"],
                        "copied_bytes": prepared["copied_bytes"],
                        "elapsed_seconds": prepared["elapsed_seconds"],
                    },
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        os.execvpe(launch["argv"][0], launch["argv"], environment)
    except (OSError, ValueError, TypeError) as error:
        print("ENTRYPOINT REFUSED: " + str(error), file=sys.stderr)
        return 2
    return 2  # An executor that returns did not hand off to the server.


if __name__ == "__main__":
    raise SystemExit(main())
