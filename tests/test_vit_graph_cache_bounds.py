"""The shipped sglang patch must bound the ViT graph cache (memory safety)."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_patch_adds_graph_budget():
    text = (ROOT / "patches/sglang.patch").read_text()
    assert "_enforce_graph_budget" in text, "patch must add bounded graph eviction"
    assert "DEFAULT_VIT_MAX_GRAPHS" in text
    assert "_inflight" in text, "patch must guard in-flight replays from eviction"


def test_lock_covers_bounded_runner():
    lock = json.loads((ROOT / "locks/runtime.json").read_text())
    runner = lock["sglang"]["python_files"][
        "srt/multimodal/vit_cuda_graph_runner.py"
    ]
    # The original (defective) runner hash was 9f5ce583d5b7c4e4e780...
    assert runner.startswith("50215cc26c629239"), (
        "runner hash must reflect the bounded-cache implementation"
    )
