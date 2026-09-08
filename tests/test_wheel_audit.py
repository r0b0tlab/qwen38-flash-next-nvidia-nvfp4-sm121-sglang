"""Installed-wheel inventory differs only by two upstream developer helpers."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OMITTED = (
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py",
    "multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py",
)


def checker():
    spec = importlib.util.spec_from_file_location(
        "image_env", ROOT / "docker/verify_environment.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "installed_source_inventory"), (
        "source-tree list is incorrectly used as installed-wheel inventory"
    )
    return module


def test_installed_inventory_keeps_all_runtime_sources():
    module = checker()
    source = json.loads((ROOT / "locks/runtime.json").read_text())["sglang"]
    assert source["wheel_omissions"] == list(OMITTED)
    expected = module.installed_source_inventory(source)
    assert set(expected) == set(source["python_files"]) - set(OMITTED)
    assert (
        "srt/models/qwen4_exp.py" in expected
        and "srt/models/qwen4_exp_mtp.py" in expected
    )
    assert all(p in source["python_files"] for p in OMITTED)


def test_installed_inventory_cannot_exempt_runtime_code():
    module = checker()
    source = json.loads((ROOT / "locks/runtime.json").read_text())["sglang"]
    source = copy.deepcopy(source)
    source["wheel_omissions"].append("srt/models/qwen4_exp.py")
    with pytest.raises(ValueError):
        module.installed_source_inventory(source)
