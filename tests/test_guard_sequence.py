"""End-to-end launcher sequencing tests on fake subprocesses: trap
equivalents before spawn, inspect/stop failure windows, exit-code
preservation and cleanup-failure honesty. No docker, no GPU."""

import json
import os

import pytest

from scripts import guard


def _journal(tmp_path):
    return guard.Journal(str(tmp_path / "seq.jsonl"))


def test_supervise_cleanup_failure_does_not_mask_exit_code(tmp_path):
    """If the server exits 3 and cleanup fails, exit must still be 3."""

    def cleanup_fail():
        raise guard.LaunchError("scoped stop failed")

    def failer():
        return 3

    code = guard.supervise(
        proc=failer,
        journal=_journal(tmp_path),
        cleanup=cleanup_fail,
        poll_seconds=0.01,
    )
    assert code == 3
    events = [
        json.loads(line)
        for line in open(tmp_path / "seq.jsonl")
    ]
    assert events[-1]["event"] == "cleanup_failed"


def test_supervise_zero_exit_with_clean_stop(tmp_path):
    stopped = []

    def stop_ok():
        stopped.append(True)

    code = guard.supervise(
        proc=lambda: 0,
        journal=_journal(tmp_path),
        cleanup=stop_ok,
        poll_seconds=0.01,
    )
    assert code == 0
    assert stopped == [True]
    events = [json.loads(line) for line in open(tmp_path / "seq.jsonl")]
    assert events[-1]["event"] == "cleanup_ok"


def test_supervise_watchdog_breach_kills_then_stops(tmp_path):
    killed = []
    stopped = []

    class FakeProc:
        @staticmethod
        def poll():
            return None  # server "still running"

        @staticmethod
        def kill():
            killed.append(True)

    journal = _journal(tmp_path)
    code = guard.supervise(
        proc=FakeProc(),
        journal=journal,
        cleanup=lambda: stopped.append(True),
        reader=lambda: 1024 * 1024,  # 1 GiB: below the 4 GiB immediate floor
        poll_seconds=0.01,
    )
    assert code == guard.EXIT_OOM_FLOOR
    assert killed == [True]
    assert stopped == [True]
    events = [json.loads(line) for line in open(tmp_path / "seq.jsonl")]
    assert [e["event"] for e in events][-2:] == [
        "killing_server", "cleanup_ok",
    ]


def test_supervise_stop_failure_window_preserves_oom_code(tmp_path):
    """Breach -> kill succeeds -> stop fails: exit must stay the OOM code."""

    class FakeProc:
        @staticmethod
        def poll():
            return None

        @staticmethod
        def kill():
            return None

    def cleanup_fail():
        raise guard.LaunchError("docker stop exploded")

    code = guard.supervise(
        proc=FakeProc(),
        journal=_journal(tmp_path),
        cleanup=cleanup_fail,
        reader=lambda: 1024 * 1024,
        poll_seconds=0.01,
    )
    assert code == guard.EXIT_OOM_FLOOR


def test_supervise_stop_failure_window_preserves_generic_code(tmp_path):
    def cleanup_fail():
        raise guard.LaunchError("docker stop exploded")

    code = guard.supervise(
        proc=lambda: 7,
        journal=_journal(tmp_path),
        cleanup=cleanup_fail,
        poll_seconds=0.01,
    )
    assert code == 7


def test_supervise_never_kills_after_exit(tmp_path):
    """Once the process has exited, supervise must not call kill()."""
    killed = []

    class FakeProc:
        calls = {"n": 0}

        @classmethod
        def poll(cls):
            cls.calls["n"] += 1
            return 0  # exited immediately

        @classmethod
        def kill(cls):
            killed.append(True)

    guard.supervise(
        proc=FakeProc(),
        journal=_journal(tmp_path),
        cleanup=lambda: None,
        reader=lambda: 1024 * 1024,  # would breach if sampling continued
        poll_seconds=0.01,
    )
    assert killed == []
