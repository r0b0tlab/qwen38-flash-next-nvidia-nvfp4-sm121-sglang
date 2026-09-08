"""Entrypoint contract tests: exact argv for the ONE-GB10 launch and the
entry environment; foreign model ids, missing shas and disallowed modes
are rejected; no unquantized draft, no language-only reduction, no
caller-supplied TP."""

import json

import pytest

from runtime import ProfileError, profile_from_dict
from runtime import entrypoint as ep

MODEL_SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
SOURCES = {"model": {"id": ep.MODEL_ID, "sha": MODEL_SHA}}

BASE = {
    "schema": 1,
    "mode": "ar",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}


def _argv(**overrides):
    doc = {**BASE, **overrides}
    profile = profile_from_dict(doc)
    return ep.build_argv(profile, SOURCES), profile


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_argv_is_pure_array_starting_with_sglang_serve():
    argv, _ = _argv()
    assert isinstance(argv, list)
    assert all(isinstance(token, str) for token in argv)
    assert argv[:2] == ["sglang", "serve"]


def test_single_gpu_identity_flags():
    argv, _ = _argv()
    assert _flag(argv, "--tp-size") == "1"
    assert _flag(argv, "--nnodes") == "1"
    assert _flag(argv, "--node-rank") == "0"
    assert _flag(argv, "--model-path") == "/model"
    assert _flag(argv, "--served-model-name") == ep.MODEL_ID
    assert _flag(argv, "--host") == "0.0.0.0"
    assert _flag(argv, "--port") == "30000"


def test_quantization_and_native_expert_flags():
    argv, _ = _argv()
    assert _flag(argv, "--dtype") == "bfloat16"
    assert _flag(argv, "--quantization") == "modelopt_mixed"
    assert _flag(argv, "--load-format") == "safetensors"
    assert _flag(argv, "--fp4-gemm-backend") == "flashinfer_cutlass"
    assert _flag(argv, "--moe-runner-backend") == "flashinfer_cutlass"
    assert "--ple-offload-embedding" in argv
    assert _flag(argv, "--ple-offload-backend") == "file"
    assert _flag(argv, "--ple-offload-dir") == "/cache/ple/" + MODEL_SHA


def test_native_decode_graph_flags():
    argv, _ = _argv()
    assert _flag(argv, "--cuda-graph-backend-decode") == "full"
    assert _flag(argv, "--cuda-graph-backend-prefill") == "disabled"
    assert _flag(argv, "--cuda-graph-bs-decode") == "1"
    assert _flag(argv, "--page-size") == "64"
    assert _flag(argv, "--attention-backend") == "triton"
    assert _flag(argv, "--mamba-ssm-dtype") == "float32"
    assert _flag(argv, "--linear-attn-prefill-backend") == "triton"
    assert _flag(argv, "--linear-attn-decode-backend") == "triton"
    assert _flag(argv, "--linear-attn-verify-backend") == "triton"
    assert _flag(argv, "--reasoning-parser") == "auto"
    assert _flag(argv, "--tool-call-parser") == "auto"
    assert _flag(argv, "--watchdog-timeout") == "1800"
    assert "--enable-metrics" in argv


def test_parameter_fields_flow_from_profile():
    argv, profile = _argv(
        max_running_requests=4,
        chunked_prefill_size=2048,
        mem_fraction_static=0.82,
        max_mamba_cache_size=16,
    )
    assert _flag(argv, "--context-length") == "32768"
    assert _flag(argv, "--max-total-tokens") == "32768"
    assert _flag(argv, "--max-running-requests") == "4"
    assert _flag(argv, "--chunked-prefill-size") == "2048"
    assert _flag(argv, "--max-prefill-tokens") == "2048"
    assert _flag(argv, "--mem-fraction-static") == "0.82"
    assert _flag(argv, "--max-mamba-cache-size") == "16"
    assert _flag(argv, "--kv-cache-dtype") == "bf16"
    assert profile.chunked_prefill_size == 2048


def test_vision_flags_from_profile():
    argv, _ = _argv()
    assert _flag(argv, "--mm-attention-backend") == "triton_attn"
    argv2, _ = _argv(vision={"backend": "flashinfer_cudnn", "cuda_graph": False})
    assert _flag(argv2, "--mm-attention-backend") == "flashinfer_cudnn"


def test_ar_mode_has_no_speculative_flags():
    argv, _ = _argv()
    assert "--speculative-algorithm" not in argv


def test_nextn_flags_exact():
    argv, _ = _argv(
        mode="nextn",
        speculative={"steps": 2},
        kv_cache_dtype="fp8_e4m3",
    )
    assert _flag(argv, "--speculative-algorithm") == "NEXTN"
    assert _flag(argv, "--speculative-num-steps") == "2"
    assert _flag(argv, "--speculative-eagle-topk") == "1"
    assert _flag(argv, "--speculative-num-draft-tokens") == "3"
    assert _flag(argv, "--speculative-draft-model-quantization") == "modelopt_mixed"
    assert _flag(argv, "--speculative-moe-runner-backend") == "triton"
    assert _flag(argv, "--speculative-draft-kv-cache-dtype") == "fp8_e4m3"


def test_no_unquantized_draft_or_language_only_or_tp2():
    for argv, _ in (_argv(), _argv(mode="nextn", speculative={"steps": 3})):
        joined = " ".join(argv)
        assert "--tp-size 2" not in joined
        assert _flag(argv, "--tp-size") == "1"
        assert "language" not in joined
        assert "draft-model-path" not in joined
        assert "--speculative-draft-model-path" not in joined
        # draft quantization must always accompany NEXTN
        if "--speculative-algorithm" in argv:
            assert "--speculative-draft-model-quantization" in argv


def test_env_caches_and_profile_switches():
    profile = profile_from_dict(dict(BASE))
    env = ep.build_env(profile, SOURCES)
    assert env["HOME"] == "/cache/home"
    assert env["XDG_CACHE_HOME"] == "/cache/xdg"
    assert env["HF_HOME"] == "/cache/hf"
    assert env["SGLANG_CACHE_DIR"] == "/cache/jit"
    assert env["SGLANG_VIT_ENABLE_CUDA_GRAPH"] == "1"
    assert env["SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED"] == "1"
    assert env["SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB"] == "8"
    assert env["SGLANG_QWEN4_PLE_FILE_PREFETCH"] == "1"


def test_env_reflects_profile_switches():
    profile = profile_from_dict(
        {**BASE, "vision": {"backend": "fa4", "cuda_graph": False}, "ple_rss_gib": 4}
    )
    env = ep.build_env(profile, SOURCES)
    assert env["SGLANG_VIT_ENABLE_CUDA_GRAPH"] == "0"
    assert env["SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB"] == "4"


def test_env_rejects_non_object_and_non_string():
    profile = profile_from_dict(dict(BASE))
    with pytest.raises(ProfileError):
        ep.build_env(profile, {"env": ["not", "an", "object"]})
    with pytest.raises(ProfileError):
        ep.build_env(profile, {"env": {"KEY": 3}})


def test_foreign_model_id_rejected():
    profile = profile_from_dict(dict(BASE))
    with pytest.raises(ProfileError):
        ep.build_argv(profile, {"model": {"id": "other/model", "sha": MODEL_SHA}})


def test_missing_model_sha_rejected():
    profile = profile_from_dict(dict(BASE))
    with pytest.raises(ProfileError):
        ep.build_argv(profile, {"model": {"id": ep.MODEL_ID}})


def test_ple_dir_keyed_by_full_model_sha():
    assert ep.ple_dir(MODEL_SHA) == "/cache/ple/" + MODEL_SHA
