"""Profile contract tests: accept the common default, reject every
malformed variant (unknown/duplicate fields, nonfinite, bool-as-int,
out-of-range, wrong-shape nested objects)."""

import json

import pytest

from runtime import (
    CONTEXT_LENGTH_REQUIRED,
    PROFILE_SCHEMA,
    ProfileError,
    profile_from_dict,
    profile_from_json,
    validate_payload,
)

BASE = {
    "schema": PROFILE_SCHEMA,
    "mode": "ar",
    "context_length": CONTEXT_LENGTH_REQUIRED,
    "max_total_tokens": CONTEXT_LENGTH_REQUIRED,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}


def _dump(doc):
    return json.dumps(doc)


def test_common_default_profile_accepts():
    profile = profile_from_dict(dict(BASE))
    assert profile.schema == 1
    assert profile.mode == "ar"
    assert profile.max_running_requests == 1
    assert profile.chunked_prefill_size == 1024
    assert profile.max_mamba_cache_size == 8
    assert profile.mem_fraction_static == 0.80
    assert profile.kv_cache_dtype == "bf16"
    assert profile.speculative_steps == 0
    assert profile.ple_rss_gib == 8
    assert profile.mm_processor_worker_num == 1
    assert profile.vision_backend == "triton_attn"
    assert profile.vision_cuda_graph is True


def test_digest_is_canonical_and_stable():
    a = profile_from_dict(dict(BASE))
    b = profile_from_dict(
        json.loads(_dump({**BASE, "max_running_requests": 4}))
    )
    assert a.digest() != b.digest()
    assert a.digest() == profile_from_dict(json.loads(_dump(BASE))).digest()


def test_tunable_values_accepted():
    doc = {
        **BASE,
        "mode": "nextn",
        "speculative": {"steps": 3},
        "max_running_requests": 8,
        "chunked_prefill_size": 4096,
        "max_mamba_cache_size": 64,
        "mem_fraction_static": 0.88,
        "kv_cache_dtype": "fp8_e4m3",
        "vision": {"backend": "fa4", "cuda_graph": False},
        "ple_rss_gib": 12,
        "mm_processor_worker_num": 2,
    }
    profile = profile_from_dict(doc)
    assert profile.speculative_steps == 3
    assert profile.max_running_requests == 8
    assert profile.chunked_prefill_size == 4096
    assert profile.max_mamba_cache_size == 64
    assert profile.mem_fraction_static == 0.88
    assert profile.kv_cache_dtype == "fp8_e4m3"
    assert profile.vision_backend == "fa4"
    assert profile.vision_cuda_graph is False
    assert profile.ple_rss_gib == 12
    assert profile.mm_processor_worker_num == 2


def test_final_context_length_allowed():
    doc = {
        **BASE,
        "context_length": 262144,
        "max_total_tokens": 262144,
    }
    assert profile_from_dict(doc).context_length == 262144


@pytest.mark.parametrize(
    "doc",
    [
        {**BASE, "schema": 2},
        {**BASE, "mode": "hybrid"},
        {**BASE, "context_length": 16384, "max_total_tokens": 16384},
        {**BASE, "max_total_tokens": 262144},
        {**BASE, "max_running_requests": 3},
        {**BASE, "max_running_requests": True},
        {**BASE, "chunked_prefill_size": 512},
        {**BASE, "max_mamba_cache_size": 3},
        {**BASE, "max_mamba_cache_size": 65},
        {**BASE, "mem_fraction_static": 0.69},
        {**BASE, "mem_fraction_static": 0.89},
        {**BASE, "mem_fraction_static": True},
        {**BASE, "kv_cache_dtype": "int8"},
        {**BASE, "speculative": {"steps": 1}},
        {**BASE, "vision": {"backend": "omni", "cuda_graph": True}},
        {**BASE, "vision": {"backend": "fa4"}},
        {**BASE, "vision": {"backend": "fa4", "cuda_graph": True, "x": 1}},
        {**BASE, "vision": {"backend": "fa4", "cuda_graph": 1}},
        {**BASE, "ple_rss_gib": 6},
        {**BASE, "mm_processor_worker_num": 3},
        {**BASE, "tp_size": 1},
        {**BASE, "extra_flags": ["--whatever"]},
        {**BASE, "draft_model_path": "/somewhere"},
    ],
)
def test_rejects_malformed_documents(doc):
    with pytest.raises(ProfileError):
        validate_payload(_dump(doc))


def test_nextn_requires_speculative_and_bounds():
    with pytest.raises(ProfileError):
        validate_payload(_dump({**BASE, "mode": "nextn"}))
    with pytest.raises(ProfileError):
        validate_payload(
            _dump({**BASE, "mode": "nextn", "speculative": {"steps": 0}})
        )
    with pytest.raises(ProfileError):
        validate_payload(
            _dump({**BASE, "mode": "nextn", "speculative": {"steps": 4}})
        )
    profile = profile_from_dict(
        {**BASE, "mode": "nextn", "speculative": {"steps": 2}}
    )
    assert profile.speculative_steps == 2


def test_duplicate_fields_rejected():
    with pytest.raises(ProfileError):
        validate_payload('{"schema":1,"schema":1}')


def test_nonfinite_rejected():
    for token in ("NaN", "Infinity", "-Infinity"):
        payload = (
            '{"schema":1,"mode":"ar","context_length":32768,'
            '"max_total_tokens":32768,'
            '"vision":{"backend":"triton_attn","cuda_graph":true},'
            '"mem_fraction_static":%s}' % (token,)
        )
        with pytest.raises(ProfileError):
            validate_payload(payload)


def test_bool_as_int_rejected():
    bad = dict(BASE)
    bad["context_length"] = True
    with pytest.raises(ProfileError):
        validate_payload(_dump(bad))


def test_non_object_and_trailing_garbage_rejected():
    with pytest.raises(ProfileError):
        validate_payload("[1,2]")
    with pytest.raises(ProfileError):
        validate_payload('{"schema":1} trailing')


def test_missing_file(tmp_path):
    from runtime import load_profile

    with pytest.raises(ProfileError):
        load_profile(str(tmp_path / "absent.json"))


def test_load_profile_roundtrip(tmp_path):
    from runtime import load_profile

    path = tmp_path / "profile.json"
    path.write_text(_dump(BASE), encoding="utf-8")
    profile = load_profile(str(path))
    assert profile.mode == "ar"
    assert profile.digest() == profile_from_dict(dict(BASE)).digest()
