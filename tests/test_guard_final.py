"""Current lifecycle acceptance matrix. Docker/probes only are faked."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest
from scripts import guard
from tests.launcher_fixtures import IMAGE, CID, FakeDocker, make_env, argv, options


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


def test_actual_script_help_from_unrelated_cwd(tmp_path):
    p = subprocess.run(
        [sys.executable, str(guard.REPO / "scripts/guard.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert p.returncode == 0
    assert "--receipt-sha256" in p.stdout
    assert "Traceback" not in p.stderr


def test_lifecycle_observes_real_transport_exit(env):
    docker = FakeDocker(exit_code=42)
    assert guard.run(argv(env), **options(docker)) == 42
    assert docker.calls.index("create") < docker.calls.index("start")
    assert docker.calls.count("inspect") >= 4
    assert "stop" not in docker.calls  # already exited, never kill after exit
    record = json.loads((Path(env["state-dir"]) / "launch-record.json").read_text())
    assert record["cid"] == CID and record["exit_code"] == 42
    assert "TEST-ONLY" in (Path(env["state-dir"]) / "server.log").read_text()


@pytest.mark.parametrize("flag", ["--print", "--dryrun"])
def test_plan_mode_has_no_host_mutations(env, flag):
    docker = FakeDocker()
    assert (
        guard.run(
            argv(env, flag),
            **options(docker, preflight_fn=lambda: pytest.fail("must not probe")),
        )
        == 0
    )
    assert docker.calls == []
    assert not Path(env["state-dir"]).exists()


def test_preflight_before_docker(env):
    docker = FakeDocker()
    assert (
        guard.run(argv(env), **options(docker, preflight_fn=lambda: {"status": "FAIL"}))
        == 3
    )
    assert docker.calls == []


def test_warm_state_is_never_overwritten(env):
    state = Path(env["state-dir"])
    state.mkdir()
    (state / "sentinel").write_text("preserve")
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) != 0
    assert docker.calls == [] and (state / "sentinel").read_text() == "preserve"


def test_same_cache_lock_refuses_second_owner(env):
    docker = FakeDocker()
    with guard.lifetime_lock(Path(env["cache-dir"])):
        assert guard.run(argv(env), **options(docker)) == 9
    assert docker.calls == []


@pytest.mark.parametrize("value", ["main", "12" * 32, "12" * 19, True])
def test_model_revision_literal_40_hex_contract(env, value):
    path = Path(env["sources"])
    doc = json.loads(path.read_text())
    doc["model"]["sha"] = value
    path.write_text(json.dumps(doc))
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 2
    assert docker.calls == []


@pytest.mark.parametrize("key,value", [("id", "other/model"), ("files", [])])
def test_source_identity_and_inventory_required(env, key, value):
    path = Path(env["sources"])
    doc = json.loads(path.read_text())
    doc["model"][key] = value
    path.write_text(json.dumps(doc))
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 2
    assert docker.calls == []


def test_receipt_cannot_authenticate_itself(env):
    p = Path(env["receipt"])
    doc = json.loads(p.read_text())
    del doc["files"][0]["sha256"]
    p.write_text(json.dumps(doc))
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 4
    assert docker.calls == []


def test_check_before_start_catches_model_drift(env):
    docker = FakeDocker()
    docker.hooks["create"] = lambda d: (
        Path(env["model-root"]) / "config.json"
    ).write_text("changed bytes")
    assert guard.run(argv(env), **options(docker)) == 4
    assert "start" not in docker.calls
    assert docker.doc["State"]["Running"] is False


def test_profile_and_source_mounts_are_frozen_attempt_copies(env):
    original = Path(env["profile"]).read_bytes()
    docker = FakeDocker()
    docker.hooks["create"] = lambda d: Path(env["profile"]).write_text("{}")
    assert guard.run(argv(env), **options(docker)) == 0
    profile_snapshot = Path(env["state-dir"]) / "profile.json"
    assert profile_snapshot.read_bytes() == original
    assert f"src={profile_snapshot},dst=/work/profile.json" in " ".join(docker.argv)


@pytest.mark.parametrize("place", ["model-root", "cache-dir"])
def test_attempt_must_not_overlap_input_tree(env, place):
    env["state-dir"] = str(Path(env[place]) / "attempt")
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 2
    assert not Path(env["state-dir"]).exists() and docker.calls == []


@pytest.mark.parametrize("observation", [None, True, -1, float("nan"), 3 * 1024 * 1024])
def test_memory_unknown_and_hard_floor_fail_closed(env, observation):
    docker = FakeDocker(exits_after=None)
    assert guard.run(argv(env), **options(docker, mem_reader=lambda: observation)) == 8
    assert docker.calls.count("stop") == 1
    assert docker.doc["State"]["Running"] is False


def test_sustained_floor_resets_then_breaches(env):
    docker = FakeDocker(exits_after=None)
    observations = iter([7, 7, 7, 7, 16, 7, 7, 7, 7, 7])
    assert (
        guard.run(
            argv(env),
            **options(docker, mem_reader=lambda: next(observations) * 1024 * 1024),
        )
        == 8
    )
    assert docker.polls >= 10
    assert docker.calls.count("stop") == 1


def test_deadline_is_elapsed_time_not_reset_each_poll(env):
    docker = FakeDocker(exits_after=None)
    counter = iter(range(20))
    assert (
        guard.run(
            argv(env, "--max-watch-seconds", "2"),
            **options(docker, monotonic=lambda: next(counter)),
        )
        == 8
    )
    assert docker.calls.count("stop") == 1


@pytest.mark.parametrize("window", ["create", "start"])
def test_cancel_in_mutation_windows_stops_server_first(env, window):
    docker = FakeDocker(exits_after=None)
    event = threading.Event()
    docker.hooks[window] = lambda d: event.set()
    assert guard.run(argv(env), **options(docker, stop_event=event)) == 143
    assert docker.doc["State"]["Running"] is False
    if window == "create":
        assert "start" not in docker.calls
    else:
        assert docker.calls.index("stop") < docker.calls.index("logs")


def test_real_term_handler_restored_after_shutdown(env):
    docker = FakeDocker(exits_after=None)
    old = signal.getsignal(signal.SIGTERM)
    docker.hooks["start"] = lambda d: os.kill(os.getpid(), signal.SIGTERM)
    assert guard.run(argv(env), **options(docker)) == 143
    assert signal.getsignal(signal.SIGTERM) == old
    assert docker.doc["State"]["Running"] is False


def test_uncertain_create_recovers_cidfile_without_retry(env):
    docker = FakeDocker()

    def failed(d):
        raise guard.TransportError("test-only response lost after create")

    docker.hooks["create"] = failed
    assert guard.run(argv(env), **options(docker)) == 5
    assert docker.calls.count("create") == 1 and "start" not in docker.calls
    assert (Path(env["state-dir"]) / "container.final.json").is_file()


def test_uncertain_start_stops_live_container_without_retry(env):
    docker = FakeDocker(exits_after=None)

    def failed(d):
        raise guard.TransportError("test-only start response lost")

    docker.hooks["start"] = failed
    assert guard.run(argv(env), **options(docker)) == 5
    assert docker.calls.count("start") == 1 and docker.calls.count("stop") == 1
    assert docker.doc["State"]["Running"] is False


@pytest.mark.parametrize(
    "field", [guard.OWNER_LABEL_KEY, guard.LABEL_IMAGE, guard.LABEL_PROFILE]
)
def test_foreign_binding_never_stopped(env, field):
    docker = FakeDocker(exits_after=None)

    def replace(d):
        d.doc["Config"]["Labels"][field] = "foreign"

    docker.hooks["start"] = replace
    assert guard.run(argv(env), **options(docker)) != 0
    assert "stop" not in docker.calls


@pytest.mark.parametrize("original,expected", [(0, 7), (42, 42)])
def test_evidence_failure_preserves_original_exit_code(env, original, expected):
    docker = FakeDocker(exit_code=original)

    def logs_failed(d):
        raise guard.TransportError("test-only log collection failure")

    docker.hooks["logs"] = logs_failed
    assert guard.run(argv(env), **options(docker)) == expected


def test_hardened_argv_and_owned_entrypoint(env):
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 0
    args = docker.argv
    assert args[0] == "create" and args[args.index("--gpus") + 1] == "device=0"
    for flag, value in [
        ("--memory", "112g"),
        ("--memory-swap", "112g"),
        ("--cpus", "14"),
        ("--publish", "127.0.0.1:30080:30000"),
        ("--cap-drop", "ALL"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert args[args.index(IMAGE) + 1 :] == [
        "--profile",
        "/work/profile.json",
        "--sources",
        "/work/sources.json",
    ]
    assert (
        "--privileged" not in args
        and "--entrypoint" not in args
        and "--restart" not in args
    )
    assert "docker.sock" not in " ".join(args)


@pytest.mark.parametrize("body", ["[]", "null", "[1]", "{}", "not-json"])
def test_transport_rejects_non_document_inspection(body, monkeypatch):
    transport = guard.DockerCliTransport()
    monkeypatch.setattr(transport, "command", lambda *a, **k: body)
    with pytest.raises(guard.TransportError):
        transport.inspect(CID)
    with pytest.raises(guard.TransportError):
        transport.image_inspect(IMAGE)


def test_invalid_container_exit_status_is_not_returned_as_success(env):
    docker = FakeDocker(exit_code=-1)
    assert guard.run(argv(env), **options(docker)) == 8


def test_final_record_failure_cannot_return_success(env, monkeypatch):
    original = guard._save

    def fail_final(path, value):
        if path.name == "launch-record.json" and value.get("status") == "STOPPED":
            raise OSError("test-only final persistence failure")
        return original(path, value)

    monkeypatch.setattr(guard, "_save", fail_final)
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 7
    assert docker.doc["State"]["Running"] is False
