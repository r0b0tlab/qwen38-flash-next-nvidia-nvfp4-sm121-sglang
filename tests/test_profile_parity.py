"""Source profiles may differ only by the declared optimization lever."""

import copy
import pytest
from scripts import compare as c
from tests.test_compare import _manifest


def pair():
    profile = {
        "schema": 1,
        "mode": "ar",
        "context_length": 32768,
        "max_total_tokens": 32768,
        "vision": {"backend": "triton_attn", "cuda_graph": True},
    }
    other = copy.deepcopy(profile)
    other.update(mode="nextn", speculative={"steps": 1})
    return _manifest("none", runtime_context={"profile": profile}), _manifest(
        "nextn", runtime_context={"profile": other}
    )


def test_extra_profile_lever_is_not_hidden_by_matching_top_level_manifests():
    b, a = pair()
    a["runtime_context"]["profile"]["max_mamba_cache_size"] = 32
    with pytest.raises(c.Reject):
        c.check_parity(b, a)


def test_declared_nextn_requires_actual_mode_change():
    b, a = pair()
    a["runtime_context"]["profile"] = copy.deepcopy(b["runtime_context"]["profile"])
    with pytest.raises(c.Reject):
        c.check_parity(b, a)


def test_one_real_profile_lever_is_allowed():
    b, a = pair()
    assert c.check_parity(b, a)
