"""Packaging-only retries preserve the native wheel and cold build recipe."""

import importlib
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_repackage_recipe_uses_verified_wheel_without_native_rebuild():
    assert (ROOT / "scripts/prepare_repackage.py").exists(), (
        "repackage route not implemented"
    )
    module = importlib.import_module("scripts.prepare_repackage")
    original = (ROOT / "Dockerfile").read_text()
    recipe = module.render_recipe(original, "a" * 64)
    assert "COPY --from=builder /wheels/" not in recipe
    assert "COPY build/repackage/ /tmp/sglang-wheels/" in recipe
    assert "a" * 64 in recipe
    assert "SGLANG_BUILD_RUST_EXTS=all" in original  # cold route retained
    assert 'ENTRYPOINT ["python3", "-m", "runtime.entrypoint"]' in recipe


def test_repackage_recipe_refuses_unknown_hash_or_anchor():
    assert (ROOT / "scripts/prepare_repackage.py").exists()
    module = importlib.import_module("scripts.prepare_repackage")
    with pytest.raises(ValueError):
        module.render_recipe("unknown dockerfile", "a" * 64)
    with pytest.raises(ValueError):
        module.render_recipe((ROOT / "Dockerfile").read_text(), "not-a-sha")
