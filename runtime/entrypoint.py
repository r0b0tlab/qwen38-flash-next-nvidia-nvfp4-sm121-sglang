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
import os
import re
import sys
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
    "TMPDIR",
)


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
    """Writable cache paths and non-overridable runtime safety settings."""
    profile = _checked_profile(profile)
    _checked_sources(sources)
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
        "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK": "0",
        "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH": "0",
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
        "flashinfer_cutlass",
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
            _kv_cache_flag(profile.kv_cache_dtype),
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
        print(
            json.dumps(
                {
                    "model_sha": sources["model"]["sha"],
                    "profile_sha256": profile.digest(),
                    "argv": launch["argv"],
                },
                sort_keys=True,
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
