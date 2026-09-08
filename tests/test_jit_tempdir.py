"""Native JIT temp files must be private and executable outside noexec /tmp."""

from pathlib import Path

import pytest
from runtime import entrypoint as ep, profile_from_dict

ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "model": {"id": ep.MODEL_ID, "sha": "fc694b54fb0174e0913e6adf86691ef85a4ead47"}
}
PROFILE = {
    "schema": 1,
    "mode": "ar",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}


def test_runtime_audit_exercises_real_dlopen(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "env_checker", ROOT / "docker/verify_environment.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "verify_jit_temporary_directory")
    assert module.verify_jit_temporary_directory(tmp_path) == "PASS"


def test_jit_temp_is_owned_in_image_and_entrypoint():
    env = ep.build_env(profile_from_dict(PROFILE), SOURCE)
    assert env.get("TMPDIR") == "/cache/tmp"
    assert "TMPDIR=/cache/tmp" in (ROOT / "Dockerfile").read_text()


def test_jit_temp_directory_is_private(tmp_path):
    assert hasattr(ep, "prepare_jit_temp")
    path = tmp_path / "cache" / "tmp"
    ep.prepare_jit_temp(str(path))
    assert path.is_dir() and path.stat().st_mode & 0o777 == 0o700
    ep.prepare_jit_temp(str(path))


def test_jit_temp_refuses_symlink_and_insecure_directory(tmp_path):
    assert hasattr(ep, "prepare_jit_temp")
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        ep.prepare_jit_temp(str(link))
    target.chmod(0o777)
    with pytest.raises((OSError, ValueError)):
        ep.prepare_jit_temp(str(target))
