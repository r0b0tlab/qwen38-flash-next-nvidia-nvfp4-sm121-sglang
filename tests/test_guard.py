"""Receipt/argv contracts at the current public launch boundary.

Supersedes tests of the removed plan-only launch(), static ownership label,
--rm container, and unwired Journal helpers. Full lifecycle cases are in
 test_guard_final.py; no old passing-helper result is reused as admission.
"""

import json
from pathlib import Path
import pytest
from scripts import guard
from tests.launcher_fixtures import make_env, argv, options, FakeDocker


@pytest.mark.parametrize(
    "mutation",
    [
        "root",
        "model",
        "file_count",
        "total_bytes",
        "digest",
        "missing_file",
        "nonobject",
    ],
)
def test_receipt_semantics_even_with_independent_digest(tmp_path, mutation):
    env = make_env(tmp_path)
    path = Path(env["receipt"])
    doc = json.loads(path.read_text())
    if mutation == "root":
        doc["root"] = str(tmp_path)
    elif mutation == "model":
        doc["model"]["sha"] = "aa" * 20
    elif mutation == "file_count":
        doc["file_count"] = True
    elif mutation == "total_bytes":
        doc["total_bytes"] += 1
    elif mutation == "digest":
        doc["files"][0]["sha256"] = "aa" * 32
    elif mutation == "missing_file":
        doc["files"] = []
    else:
        doc["files"] = [None]
    path.write_text(json.dumps(doc))
    env["receipt-sha256"] = guard.digest(path.read_bytes())  # test-only trusted anchor
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 4
    assert docker.calls == []


@pytest.mark.parametrize(
    "value", ["qwen:latest", "sha256:short", "sha256:" + "gg" * 32]
)
def test_moving_or_malformed_image_ref_rejected(tmp_path, value):
    env = make_env(tmp_path)
    env["image"] = value
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 2
    assert docker.calls == []


@pytest.mark.parametrize(
    "field,value",
    [("Architecture", "amd64"), ("Id", "sha256:" + "ef" * 32), ("Os", "windows")],
)
def test_image_inspection_binding_before_create(tmp_path, field, value):
    env = make_env(tmp_path)
    docker = FakeDocker()
    base = docker.image_inspect
    docker.image_inspect = lambda image: {**base(image), field: value}
    assert guard.run(argv(env), **options(docker)) == 4
    assert "create" not in docker.calls


def test_fresh_nonce_for_each_attempt_and_exact_telemetry_schema(tmp_path):
    env = make_env(tmp_path)
    nonces = []
    for i in range(2):
        env["state-dir"] = str(tmp_path / f"attempt-{i}")
        assert guard.run(argv(env), **options(FakeDocker())) == 0
        state = Path(env["state-dir"])
        nonces.append(json.loads((state / "launch-record.json").read_text())["nonce"])
        for line in (state / "telemetry.jsonl").read_text().splitlines():
            assert set(json.loads(line)) == {"time", "cid", "mem_available_kb"}
    assert nonces[0] != nonces[1]


def test_stat_drift_before_create(tmp_path):
    env = make_env(tmp_path)
    (Path(env["model-root"]) / "config.json").write_text("different bytes")
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 4 and docker.calls == []
