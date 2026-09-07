"""Launcher tests. Everything destructive is mocked or pointed at
tmp_path; no docker invocation, no real GPU, no real model tree."""

import json
import os
import subprocess

import pytest

from scripts import guard, verify_files as vf

MODEL_SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
SOURCES = {"model": {"id": "nvidia/Qwen3.8-Flash-Next-NVFP4", "sha": MODEL_SHA}}


@pytest.fixture
def verified_tree(tmp_path):
    """A tiny verified model tree with a valid receipt."""
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"a":1}')
    data = open(root / "config.json", "rb").read()
    import hashlib

    inventory = {
        "model": SOURCES["model"],
        "files": [
            {
                "path": "config.json",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    inv_path = tmp_path / "model.files.json"
    inv_path.write_text(json.dumps(inventory))
    receipt = vf.verify_files(str(root), str(inv_path))
    receipt_path = tmp_path / "receipt.json"
    vf.write_receipt(receipt, str(receipt_path))
    return str(root), str(receipt_path)


def _launch(tmp_path, verified_tree, **kw):
    root, receipt_path = verified_tree
    profile_path = tmp_path / "profile.json"
    profile_path.write_text('{"schema":1}')
    base = dict(
        image="image:test",
        profile_path=str(profile_path),
        model_root=root,
        cache_dir=str(tmp_path / "cache"),
        sources=SOURCES,
        receipt_path=receipt_path,
        state_dir=str(tmp_path / "state"),
        dry_run=True,
    )
    base.update(kw)
    return guard.launch(**base)


# ------------------------------------------------------------ docker argv

def test_docker_argv_hardening(verified_tree, tmp_path):
    record = _launch(tmp_path, verified_tree)
    argv = record["argv"]
    def flag(name):
        return argv[argv.index(name) + 1]
    assert argv[0] == "docker"
    assert "--rm" in argv
    assert flag("--gpus") == "device=0"
    assert flag("--cpus") == "14"
    assert flag("--memory") == "112g"
    assert flag("--memory-swap") == "112g"
    assert flag("--pids-limit") == "2048"
    assert flag("--shm-size") == "8g"
    assert flag("--cap-drop") == "ALL"
    assert flag("--security-opt") == "no-new-privileges"
    assert "--read-only" in argv
    assert flag("--publish") == "127.0.0.1:30080:30000"
    assert flag("--label") == guard.OWNER_LABEL
    assert "--restart" not in argv
    assert not any("docker.sock" in token for token in argv)
    assert any(
        token.startswith("type=bind,src=%s,dst=/model,ro" % verified_tree[0])
        for token in argv
    )
    assert any(token.startswith("type=bind,src=") and token.endswith(",dst=/cache")
               for token in argv)
    assert any(token.startswith("--tmpfs") or token == "--tmpfs" for token in argv)


def test_docker_argv_binds_profile_read_only(verified_tree, tmp_path):
    record = _launch(tmp_path, verified_tree)
    argv = record["argv"]
    profile_bind = [
        token
        for token in argv
        if token.startswith("type=bind,src=") and "dst=/work/profile.json,ro" in token
    ]
    assert len(profile_bind) == 1


def test_launch_record_persists_identity(verified_tree, tmp_path):
    state_dir = str(tmp_path / "state")
    record = _launch(tmp_path, verified_tree, state_dir=state_dir)
    with open(os.path.join(state_dir, "launch-record.json")) as handle:
        persisted = json.load(handle)
    assert persisted["image_config_id"] == guard.IMAGE_CONFIG_ID
    assert persisted["epoch"] == record["epoch"]
    assert persisted["argv"] == record["argv"]
    assert persisted["model_receipt"]["model"]["sha"] == MODEL_SHA
    assert persisted["image"] == "image:test"
    assert persisted["source"] == "runtime-contracts"
    assert len(persisted["profile_sha256"]) == 64


def test_epoch_increments_under_flock(verified_tree, tmp_path):
    record_a = _launch(tmp_path, verified_tree)
    record_b = _launch(tmp_path, verified_tree)  # same state dir
    assert record_b["epoch"] == record_a["epoch"] + 1


# ---------------------------------------------------------- receipt gate

def test_launch_refuses_model_sha_mismatch(verified_tree, tmp_path):
    root, receipt_path = verified_tree
    with open(receipt_path) as handle:
        receipt = json.load(handle)
    receipt["model"]["sha"] = "dead" * 16
    with open(receipt_path, "w") as handle:
        json.dump(receipt, handle)
    with pytest.raises(guard.LaunchError):
        _launch(tmp_path, verified_tree)


def test_launch_refuses_receipt_root_mismatch(verified_tree, tmp_path):
    with pytest.raises(guard.LaunchError):
        _launch(tmp_path, verified_tree, model_root=str(tmp_path / "elsewhere"))


def test_launch_refuses_stat_drift(verified_tree, tmp_path):
    root, receipt_path = verified_tree
    target = os.path.join(root, "config.json")
    st = os.stat(target)
    os.utime(target, ns=(st.st_mtime_ns + 10, st.st_ctime_ns))
    with pytest.raises(guard.LaunchError):
        _launch(tmp_path, verified_tree)


def test_launch_refuses_missing_receipt(verified_tree, tmp_path):
    with pytest.raises(guard.LaunchError):
        _launch(tmp_path, verified_tree, receipt_path=str(tmp_path / "nope.json"))


def test_launch_refuses_missing_sources_identity(verified_tree, tmp_path):
    with pytest.raises(guard.LaunchError):
        _launch(tmp_path, verified_tree, sources={"model": {"id": "x/y"}})


def test_launch_refuses_failed_preflight(verified_tree, tmp_path):
    with pytest.raises(guard.LaunchError):
        _launch(
            tmp_path,
            verified_tree,
            preflight_fn=lambda: {"status": "FAIL", "checks": []},
        )


def test_launch_accepts_passing_preflight(verified_tree, tmp_path):
    record = _launch(
        tmp_path,
        verified_tree,
        preflight_fn=lambda: {"status": "PASS", "checks": []},
    )
    assert record["epoch"] >= 1


# --------------------------------------------------------------- journal

def test_journal_rejects_unknown_telemetry(tmp_path):
    journal = guard.Journal(str(tmp_path / "j.jsonl"))
    with pytest.raises(guard.LaunchError):
        journal.emit("launch", totally_unknown_field=1)
    journal.close()


def test_journal_records_launch(verified_tree, tmp_path):
    state_dir = str(tmp_path / "state")
    _launch(tmp_path, verified_tree, state_dir=state_dir)
    events = [
        json.loads(line)
        for line in open(os.path.join(state_dir, "journal.jsonl"))
    ]
    assert events[0]["event"] == "launch"
    assert set(events[0]) <= guard.Journal.SCHEMA


# -------------------------------------------------------------- watchdog

def test_watchdog_completes_on_exit(tmp_path):
    journal = guard.Journal(str(tmp_path / "j.jsonl"))
    proc = subprocess.Popen(["true"])
    assert guard.run_watchdog(proc, journal, poll_seconds=0.01) == "completed"
    journal.close()


def test_watchdog_immediate_breach(tmp_path):
    journal = guard.Journal(str(tmp_path / "j.jsonl"))
    proc = subprocess.Popen(["sleep", "5"])
    try:
        result = guard.run_watchdog(
            proc,
            journal,
            poll_seconds=0.01,
            reader=lambda: 3 * 1024 * 1024,  # below 4 GiB immediate floor
        )
        assert result == "foreign"
    finally:
        proc.kill()
        proc.wait()
    journal.close()


def test_watchdog_sustained_breach_requires_five_samples(tmp_path):
    journal = guard.Journal(str(tmp_path / "j.jsonl"))
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        return 7 * 1024 * 1024  # below 8 GiB floor, above 4 GiB immediate

    proc = subprocess.Popen(["sleep", "5"])
    try:
        result = guard.run_watchdog(
            proc, journal, poll_seconds=0.001, reader=reader
        )
        assert result == "foreign"
        assert calls["n"] == guard.WATCHDOG_SUSTAINED_SAMPLES
    finally:
        proc.kill()
        proc.wait()
    journal.close()


def test_watchdog_unknown_samples_do_not_breach(tmp_path):
    journal = guard.Journal(str(tmp_path / "j.jsonl"))
    proc = subprocess.Popen(["sleep", "5"])
    try:
        result = guard.run_watchdog(
            proc,
            journal,
            poll_seconds=0.01,
            reader=lambda: None,
            max_seconds=0.3,
        )
        assert result == "timeout"  # unknown never triggers the kill path
    finally:
        proc.kill()
        proc.wait()
    journal.close()


def test_classify_boundaries():
    floor = 8 * 1024 * 1024
    imm = 4 * 1024 * 1024
    classify = guard._classify
    assert classify([9000000] * 5, floor, imm, 5) == "stopped"
    assert classify([7000000] * 4, floor, imm, 5) == "stopped"
    assert classify([7000000] * 5, floor, imm, 5) == "foreign"
    assert classify([9000000, 3999999], floor, imm, 5) == "foreign"
    assert classify([None, 9000000], floor, imm, 5) == "unknown"
    assert classify([], floor, imm, 5) == "stopped"


# ---------------------------------------------------------------- stop

class FakeContainer:
    def __init__(self, labels, running=True):
        self.labels = labels
        self.running = running
        self.stopped = False


def _docker_inspect_spy(containers):
    """Return an inspect function that never touches real docker."""

    def inspect(cid):
        for container in containers:
            if container["cid"] == cid:
                return container
        raise guard.LaunchError("no such container: %s" % (cid,))

    return inspect


def test_stop_owned_container():
    containers = [
        {"cid": "abc123", "labels": {guard.OWNER_LABEL_KEY: guard.OWNER_LABEL_VALUE}}
    ]
    stopped = []
    guard.stop_owned(
        "abc123",
        docker_inspect=_docker_inspect_spy(containers),
        docker_stop=lambda cid, timeout=None: stopped.append(cid),
    )
    assert stopped == ["abc123"]


def test_stop_refuses_unowned_container():
    containers = [
        {"cid": "xyz789", "labels": {"io.other.owner": "someone"}}
    ]
    with pytest.raises(guard.LaunchError):
        guard.stop_owned(
            "xyz789",
            docker_inspect=_docker_inspect_spy(containers),
            docker_stop=lambda cid, timeout=None: None,
        )


def test_stop_refuses_unknown_cid():
    with pytest.raises(guard.LaunchError):
        guard.stop_owned(
            "ghost",
            docker_inspect=_docker_inspect_spy([]),
            docker_stop=lambda cid, timeout=None: None,
        )
