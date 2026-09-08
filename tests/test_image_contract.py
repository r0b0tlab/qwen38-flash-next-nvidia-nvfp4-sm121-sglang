"""Image packaging contract; CPU-only, never builds or starts a GPU image."""

import hashlib
import json
from pathlib import Path
import re

from runtime import load_profile
from runtime.entrypoint import build_all

ROOT = Path(__file__).resolve().parents[1]


def test_required_reproducibility_artifacts_exist():
    for name in (
        "Dockerfile",
        ".dockerignore",
        "locks/runtime.json",
        "locks/python-overlay.lock",
        "docker/verify_environment.py",
        "scripts/prepare_build.py",
        "profiles/ar.json",
        "profiles/nextn.json",
    ):
        assert (ROOT / name).is_file(), name


def test_source_and_parent_are_exactly_locked():
    path = ROOT / "locks/runtime.json"
    assert path.exists(), "runtime identity lock not implemented"
    lock = json.loads(path.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", lock["sglang"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", lock["sglang"]["tree"])
    assert lock["sglang"]["main"] == "20ca564bf77518604b89dc9f7fdba640daabd88b"
    assert re.fullmatch(r"lmsysorg/sglang@sha256:[0-9a-f]{64}", lock["base_image"])
    assert lock["runtime_uid"] == lock["runtime_gid"] == 1001
    assert (
        hashlib.sha256((ROOT / "patches/sglang.patch").read_bytes()).hexdigest()
        == lock["sglang"]["patch_sha256"]
    )


def test_entrypoint_and_native_build_runtime_limits():
    path = ROOT / "Dockerfile"
    assert path.exists(), "Dockerfile not implemented"
    text = path.read_text()
    assert "FROM ${BASE_IMAGE}" in text
    assert "SGLANG_BUILD_RUST_EXTS=all" in text
    assert "CARGO_BUILD_JOBS=2" in text and "MAX_JOBS=1" in text
    assert "TORCH_CUDA_ARCH_LIST=12.0" in text
    assert "HF_HUB_OFFLINE=1" in text and "TRANSFORMERS_OFFLINE=1" in text
    assert "USER 1001:1001" in text
    assert 'ENTRYPOINT ["python3", "-m", "runtime.entrypoint"]' in text
    assert '"--profile"' in text and '"--sources"' in text
    assert "pip install --no-deps" in text
    assert "verify_environment.py" in text
    assert "COPY . " not in text


def test_shipped_profiles_have_same_safe_envelope():
    paths = [ROOT / "profiles/ar.json", ROOT / "profiles/nextn.json"]
    assert all(p.is_file() for p in paths), "profiles not implemented"
    source = json.loads((ROOT / "locks/sources.json").read_text())
    ar, nextn = map(lambda p: load_profile(str(p)), paths)
    assert ar.context_length == nextn.context_length == 32768
    assert ar.max_total_tokens == nextn.max_total_tokens == 32768
    assert ar.max_running_requests == nextn.max_running_requests == 1
    a, b = build_all(ar, source), build_all(nextn, source)
    assert "--speculative-algorithm" not in a["argv"]
    assert "NEXTN" in b["argv"] and "--speculative-moe-runner-backend" in b["argv"]
    assert "--language-model-only" not in b["argv"]


def test_driver_is_exact_attributed_first_party_copy():
    path = ROOT / "locks/runtime.json"
    assert path.exists(), "driver provenance lock not implemented"
    lock = json.loads(path.read_text())["q200_driver"]
    assert lock["upstream_commit"] == "7bfacc510506e955a1392f94b87213b798e45e51"
    raw = (ROOT / "docker/q200_sandbox_driver.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == lock["sha256"]
    compile(raw, "q200_sandbox_driver.py", "exec")
