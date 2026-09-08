"""Cleanup/lock public contracts; obsolete client-kill helpers are not retained."""

import json
from pathlib import Path
import pytest
from scripts import guard, guard_stop
from tests.launcher_fixtures import make_env, argv, options, FakeDocker, CID


def test_lock_held_through_stop_and_released_afterwards(tmp_path):
    env = make_env(tmp_path)
    docker = FakeDocker(exits_after=None)
    seen = []

    def check(d):
        with pytest.raises(BlockingIOError):
            with guard.lifetime_lock(Path(env["cache-dir"])):
                pytest.fail("released before server stop")
        seen.append(True)

    docker.hooks["stop"] = check
    assert guard.run(argv(env), **options(docker, mem_reader=lambda: None)) == 8
    assert seen == [True]
    with guard.lifetime_lock(Path(env["cache-dir"])):
        pass


def test_failed_stop_retains_watchdog_failure_and_cid(tmp_path):
    env = make_env(tmp_path)
    docker = FakeDocker(exits_after=None)

    def failed(d):
        raise guard.TransportError("test-only failed stop")

    docker.hooks["stop"] = failed
    assert guard.run(argv(env), **options(docker, mem_reader=lambda: None)) == 8
    record = json.loads((Path(env["state-dir"]) / "launch-record.json").read_text())
    assert record["cid"] == CID and record["cleanup_error"]


def test_stop_command_requires_complete_bound_record(tmp_path):
    env = make_env(tmp_path)
    docker = FakeDocker()
    assert guard.run(argv(env), **options(docker)) == 0
    record = Path(env["state-dir"]) / "launch-record.json"
    assert guard_stop.main(["--record", str(record)], transport=docker) == 0
    doc = json.loads(record.read_text())
    doc["nonce"] = "11" * 16
    record.write_text(json.dumps(doc))
    before = docker.calls.count("stop")
    assert guard_stop.main(["--record", str(record)], transport=docker) == 7
    assert docker.calls.count("stop") == before


def test_watchdog_never_samples_after_exit(tmp_path):
    env = make_env(tmp_path)
    docker = FakeDocker(exits_after=1)
    assert (
        guard.run(
            argv(env),
            **options(docker, mem_reader=lambda: pytest.fail("sampled dead container")),
        )
        == 0
    )
    assert "stop" not in docker.calls
