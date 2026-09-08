"""Keep AR/NEXTN admission profiles matched, including hybrid-state headroom.

These are profile-contract tests, not proof of live KV capacity or speed.
"""

import json
from pathlib import Path

import pytest

from runtime import profile_from_dict
from runtime.entrypoint import build_argv

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["ar", "nextn"])
def test_initial_profile_reserves_memory_for_native_draft_and_state(mode):
    raw = json.loads((ROOT / "profiles" / (mode + ".json")).read_text())
    assert raw["mem_fraction_static"] == 0.83
    assert raw["context_length"] == raw["max_total_tokens"] == 32768
    assert raw["max_running_requests"] == 1
    assert raw["max_mamba_cache_size"] == 16
    assert raw["kv_cache_dtype"] == "bf16"
    assert raw["ple_rss_gib"] == 4
    assert raw["vision"] == {"backend": "triton_attn", "cuda_graph": True}
    profile = profile_from_dict(raw)
    sources = json.loads((ROOT / "locks/sources.json").read_text())
    argv = build_argv(profile, sources)
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.83"
    assert "--disable-cuda-graph" not in argv
    assert "--disable-decode-cuda-graph" not in argv


def test_ar_and_nextn_differ_only_in_speculation():
    ar = json.loads((ROOT / "profiles/ar.json").read_text())
    nextn = json.loads((ROOT / "profiles/nextn.json").read_text())
    assert nextn.pop("speculative") == {"steps": 1}
    assert ar.pop("mode") == "ar"
    assert nextn.pop("mode") == "nextn"
    assert ar == nextn
