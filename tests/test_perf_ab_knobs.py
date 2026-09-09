"""Diagnostic A/B profile knobs: decode-graph backend and draft quantization.

These knobs exist so a measured single-GB10 A/B can isolate a bottleneck
without rebuilding the engine. Production profiles keep the frozen defaults.
"""

import json
from pathlib import Path

import pytest

from runtime import ProfileError, profile_from_dict
from runtime.entrypoint import build_argv

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "locks/sources.json").read_text())
PRODUCTION = json.loads((ROOT / "profiles/nextn-262k-c2-s3.json").read_text())


def test_defaults_keep_frozen_production_argv():
    argv = build_argv(profile_from_dict(PRODUCTION), SOURCES)
    assert argv[argv.index("--cuda-graph-backend-decode") + 1] == "full"
    assert (
        argv[argv.index("--speculative-draft-model-quantization") + 1]
        == "modelopt_mixed"
    )


def test_decode_graph_can_be_disabled_for_diagnostics():
    raw = dict(PRODUCTION)
    raw["decode_graph_backend"] = "disabled"
    argv = build_argv(profile_from_dict(raw), SOURCES)
    assert argv[argv.index("--cuda-graph-backend-decode") + 1] == "disabled"


def test_draft_quantization_unquant_exposed():
    raw = dict(PRODUCTION)
    raw["speculative"] = {"steps": 3, "draft_quantization": "unquant"}
    argv = build_argv(profile_from_dict(raw), SOURCES)
    assert argv[argv.index("--speculative-draft-model-quantization") + 1] == "unquant"


@pytest.mark.parametrize("value", ["off", True, None, "Disabled"])
def test_unknown_decode_graph_backend_rejected(value):
    raw = dict(PRODUCTION)
    raw["decode_graph_backend"] = value
    with pytest.raises(ProfileError):
        profile_from_dict(raw)


@pytest.mark.parametrize("value", ["bf16", "auto", None])
def test_unknown_draft_quantization_rejected(value):
    raw = dict(PRODUCTION)
    raw["speculative"] = {"steps": 3, "draft_quantization": value}
    with pytest.raises(ProfileError):
        profile_from_dict(raw)
