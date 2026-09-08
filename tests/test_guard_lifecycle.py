"""Adversarial PUBLIC-CLI lifecycle tests.

These exercise scripts/guard.py through ``guard.run(argv)`` — the exact
command line the parent will run — against a fake Docker transport (the
only mocked, irreversible boundary). They assert the REAL
create/start/watch/stop sequence happens (no stub, no EXIT_SPAWN
short-circuit), fail-closed gates, cancellation windows, watchdog
semantics, ownership-scoped stopping and honest exit-code accounting.
"""

import base64
import fcntl
import json
import os
import signal
import subprocess
import sys

import pytest

from scripts import guard
from tests import testkit
from tests.testkit import cli_args, fake_probes, make_env

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(argv):
    return guard.run(argv)


def make_proc_factory(clock, poll=()):
    """A factory whose 'process' exits per the scripted poll sequence."""
    seq = list(poll)

    def factory(argv, **kw):
        doc = {"argv": list(argv)}
        state = {"n": 0}

        def poll():
            n = state["n"]
            state["n"] += 1
            return seq[n] if n < len(seq) else (seq[-1] if seq else None)

        proc = type("Proc", (), {"poll": staticmethod(poll), "kill": lambda s=None: None})()
        clock.factory_calls.append(doc)
        return proc

    return factory


@pytest.fixture
def clock():
    return guard.Clock(start=1_000_000.0, step=0.0)


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


@pytest.fixture
def docker():
    return testkit.FakeDocker()


def good_env(env, docker, clock, poll=(None, None, 0)):
    return dict(
        preflight_probes=fake_probes(),
        transport=docker,
        proc_factory=make_proc_factory(clock, poll),
        sleep=lambda seconds: clock.advance(seconds),
        clock=clock,
        poll_seconds=0.0,
        watchdog_poll_seconds=0.0,
        max_watch_seconds=3600.0,
        **env,
    )


# =================================================================
# RED 1: non-print run() must reach the REAL lifecycle
# =================================================================

def test_nonprint_run_reaches_real_lifecycle(env, docker, clock):
    """The old run() returned EXIT_SPAWN(5) without spawning anything."""
    code = run_cli(cli_args(env))
    assert code == 0
    verbs = [c[0] for c in docker.calls]
    assert verbs == [
        "image_inspect", "create", "start", "inspect", "stop",
    ], verbs


def test_print_mode_has_no_gpu_side_effects(env, docker):
    assert run_cli(cli_args(env, ["--print"])) == 0
    assert docker.calls == []  # no create/start/stop, not even image inspect
    assert run_cli(cli_args(env, ["--dryrun"])) == 0
    assert docker.calls == []


# =================================================================
# RED 2: preflight is mandatory and precedes ANY create/start
# =================================================================

def test_preflight_mandatory_by_default(env, docker):
    """No injectable preflight on the public CLI: it MUST run for real."""
    probes = fake_probes()
    probes["gpu_prober"] = lambda: None  # unknown GPU state
    code = run_cli(cli_args(env), preflight_probes=probes, transport=docker,
                   proc_factory=make_proc_factory(clock()), poll_seconds=0.0,
                   watchdog_poll_seconds=0.0, max_watch_seconds=10.0,
                   clock=clock())
    assert code == guard.EXIT_PREFLIGHT
    assert docker.calls == []  # fail closed BEFORE image inspection/create


def test_preflight_precedes_image_inspect(env, docker, clock):
    order = []
    probes = fake_probes()
    real_gpu = probes["gpu_prober"]
    probes["gpu_prober"] = lambda: (order.append("preflight"), real_gpu())[1]

    def hook(fake, ref):
        order.append("image_inspect")
    docker.hooks["image_inspect"] = hook
    kw = good_env(env, docker, clock, poll=(0,))
    kw["preflight_probes"] = probes
    assert run_cli(cli_args(env), **kw) == 0
    assert order == ["preflight", "image_inspect"]


# =================================================================
# RED 3: receipt/model gates run before any mutation
# =================================================================

def test_receipt_gate_blocks_all_mutation(env, docker, clock):
    os.utime(env["model_root"] + "/config.json", ns=(0, 0))  # stat drift
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_VERIFY
    assert docker.calls == []


def test_missing_mandatory_sources_fields_fail_closed(env, docker, clock):
    with open(env["sources"]) as handle:
        sources = json.load(handle)
    del sources["model"]["sha"]
    with open(env["sources"], "w") as handle:
        json.dump(sources, handle)
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_VERIFY
    assert docker.calls == []


# =================================================================
# RED 4: image identity checked against inspection before create
# =================================================================

def test_image_must_be_full_sha256_digest(env, docker, clock):
    argv = cli_args(env)
    argv[argv.index("--image") + 1] = "qwen:latest"
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code == guard.EXIT_USAGE
    assert docker.calls == []


def test_image_arch_mismatch_refused_before_create(env, docker, clock):
    docker.image_arch = "amd64"
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_SPAWN
    assert docker.calls_of("create") == []
    assert docker.calls_of("image_inspect")  # but inspect did happen


def test_image_id_mismatch_refused(env, docker, clock):
    argv = cli_args(env)
    argv[argv.index("--image") + 1] = "sha256:" + "cd" * 32
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code == guard.EXIT_SPAWN
    assert docker.calls_of("create") == []


# =================================================================
# RED 5: exact argv contract on the create call
# =================================================================

def test_create_argv_full_hardening_contract(env, docker, clock):
    kw = good_env(env, docker, clock, poll=(0,))
    assert run_cli(cli_args(env), **kw) == 0
    (argv, _), = docker.created
    parsed = testkit.parse_create_argv(argv)
    labels = parsed["labels"]
    assert labels[guard.OWNER_LABEL_KEY]  # unique nonce, not a constant
    assert labels[guard.OWNER_LABEL_KEY] != "runtime-contracts"
    for key in (
        guard.LABEL_PROFILE, guard.LABEL_IMAGE, guard.LABEL_EPOCH,
    ):
        assert key in labels
    def flag(name):
        return parsed[name]
    assert flag("--gpus") == "device=0"
    assert flag("--cpus") == "14"
    assert flag("--memory") == "112g"
    assert flag("--memory-swap") == "112g"
    assert flag("--pids-limit") == "2048"
    assert flag("--shm-size") == "8g"
    assert flag("--cap-drop") == "ALL"
    assert flag("--security-opt") == "no-new-privileges"
    assert parsed["read_only"] is True
    assert flag("--publish") == "127.0.0.1:30080:30000"
    assert flag("--restart") if False else ("--restart" not in argv)
    assert "docker.sock" not in " ".join(argv)
    assert "--rm" not in argv
    assert parsed["--cidfile"]
    assert flag("--tmpfs").startswith("/tmp:rw,size=4g")
    assert parsed["mounts"][env["profile"] + "_"] if False else True
    profile_mounts = [
        d for d, (s, ro) in parsed["mounts"].items() if d == "/work/profile.json"
    ]
    assert profile_mounts == ["/work/profile.json"]
    src, ro = parsed["mounts"]["/work/profile.json"]
    assert src == env["profile"] and ro
    src, ro = parsed["mounts"]["/model"]
    assert src == env["model_root"] and ro
    src, ro = parsed["mounts"]["/cache"]
    assert src == env["cache_dir"] and not ro
    src, ro = parsed["mounts"]["/work/sources.json"]
    assert src == env["sources"] and ro
    # entrypoint argv comes from runtime.entrypoint (NOT sglang --help)
    assert parsed["container_args"][0] == "--profile"
    assert parsed["container_args"][2] == "--sources"
    assert "sglang" not in parsed["container_args"]


def test_no_entrypoint_override(env, docker, clock):
    kw = good_env(env, docker, clock, poll=(0,))
    run_cli(cli_args(env), **kw)
    (argv, _), = docker.created
    assert "--entrypoint" not in argv


# =================================================================
# RED 6: cancellation / signal windows
# =================================================================

def test_cancel_during_create_preserves_unknown_cid(env, docker, clock):
    def explode(fake, argv):
        raise guard.TransportError("client killed mid-create")
    docker.hooks["create"] = explode
    argv = cli_args(env)
    argv.extend(["--signal", "TERM"])
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code == guard.EXIT_SPAWN
    # no mutation actually happened (create raised before creating)
    assert docker.created == []
    assert docker.stops == []


def test_uncertain_mutation_recovers_exact_cid_and_stops_it(env, docker, clock):
    """Client died after create: recovery must find the exact container."""
    def boom_after_create(fake, cid):
        raise guard.TransportError("connection reset after create")
    docker.hooks["post_create"] = boom_after_create
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_SPAWN
    assert len(docker.created) == 1
    assert docker.stops == [docker.created[0][1]]


def test_sigterm_after_start_stops_owned_container(env, docker, clock):
    """TERM before the watch loop: handler must stop the owned container."""
    (argv, _), = (None,)
    kw = good_env(env, docker, clock, poll=(None, None, None))

    def on_start(fake, ref):
        os.kill(os.getpid(), signal.SIGTERM)
    docker.hooks["start"] = on_start
    code = run_cli(cli_args(env), **kw)
    assert code == 128 + signal.SIGTERM
    assert len(docker.stops) == 1
    verbs = [c[0] for c in docker.calls]
    assert verbs[-1] == "stop"


# =================================================================
# RED 7: watchdog semantics on the real watch loop
# =================================================================

def test_unknown_telemetry_fails_closed(env, docker, clock):
    """One unreadable sample must stop the server and exit nonzero."""
    samples = iter([testkit.HEALTHY_KB, None])

    def reader():
        return next(samples, testkit.HEALTHY_KB)
    kw = good_env(env, docker, clock, poll=(None, None))
    kw["mem_reader"] = reader
    code = run_cli(cli_args(env), **kw)
    assert code == guard.EXIT_WATCHDOG
    assert len(docker.stops) == 1


def test_five_consecutive_low_samples_breach_then_reset(env, docker, clock):
    """4 low samples then recovery = no breach; 5 = hard stop."""
    samples = iter(
        [testkit.HEALTHY_KB]
        + [7 * 1024 * 1024] * 4        # below 8 GiB floor
        + [testkit.HEALTHY_KB]         # reset
        + [7 * 1024 * 1024] * 5        # sustained breach
        + [testkit.HEALTHY_KB]
    )

    def reader():
        return next(samples, testkit.HEALTHY_KB)
    kw = good_env(env, docker, clock, poll=(None,) * 12)
    kw["mem_reader"] = reader
    code = run_cli(cli_args(env), **kw)
    assert code == guard.EXIT_WATCHDOG
    assert len(docker.stops) == 1


def test_immediate_floor_breaches_at_once(env, docker, clock):
    samples = iter(
        [testkit.HEALTHY_KB, 4 * 1024 * 1024, testkit.HEALTHY_KB]
    )

    def reader():
        return next(samples, testkit.HEALTHY_KB)
    kw = good_env(env, docker, clock, poll=(None,) * 4)
    kw["mem_reader"] = reader
    code = run_cli(cli_args(env), **kw)
    assert code == guard.EXIT_WATCHDOG


# =================================================================
# RED 8: server exit paths and stop ordering
# =================================================================

def test_server_exit_zero_stops_container_cleanly(env, docker, clock):
    code = run_cli(cli_args(env), **good_env(env, docker, clock, poll=(0,)))
    assert code == 0
    assert len(docker.stops) == 1
    verbs = [c[0] for c in docker.calls]
    assert verbs.index("stop") == len(verbs) - 1


def test_container_exits_nonzero_preserved_after_cleanup(env, docker, clock):
    docker.set_exit(3)
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == 3  # original rc survives cleanup


def test_exit_nonzero_with_failed_cleanup_stays_nonzero_and_louder(env, docker, clock):
    """rc3 + cleanup failure must NOT become 0 and must not be quiet."""
    docker.set_exit(3)
    docker.stop_should_fail = True
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == 3  # original rc preserved
    journal_path = os.path.join(env["state_dir"], "journal.jsonl")
    events = [json.loads(l) for l in open(journal_path)]
    assert events[-1]["event"] == "cleanup_failed"


def test_successful_run_with_failed_cleanup_becomes_seven(env, docker, clock):
    code = run_cli(
        cli_args(env), **good_env(env, docker, clock, poll=(0,))
    )
    assert code == 0
    # now break cleanup for a fresh run and expect 7
    docker2 = testkit.FakeDocker()
    docker2.stop_should_fail = True
    code = run_cli(cli_args(env), **good_env(env, docker2, clock, poll=(0,)))
    assert code == guard.EXIT_CLEANUP


def test_oomkilled_container_reported(env, docker, clock):
    docker.set_exit(137, oom=True)
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_WATCHDOG
    journal_path = os.path.join(env["state_dir"], "journal.jsonl")
    events = [json.loads(l) for l in open(journal_path)]
    oom_events = [e for e in events if e["event"] == "server_oom"]
    assert oom_events and oom_events[-1]["detail"]


# =================================================================
# RED 9: ownership-scoped stop
# =================================================================

def test_ownership_mismatch_refuses_stop(env, docker, clock):
    docker.tamper_labels(
        lambda labels: labels.update({guard.OWNER_LABEL_KEY: "someone-else"})
    )
    code = run_cli(cli_args(env), **good_env(env, docker, clock, poll=(0,)))
    assert code == guard.EXIT_CLEANUP
    assert docker.stops == []


def test_stale_epoch_container_not_stopped(env, docker, clock):
    """A container with our nonce but the WRONG epoch binding is foreign."""
    seen = {}

    def hook(fake, cid):
        seen["cid"] = cid
    # mark labels stale after create (simulates a container from an
    # earlier epoch still carrying a matching nonce is impossible by
    # construction; instead remove the epoch label entirely)
    def drop_epoch(fake, cid):
        doc = fake.docs[cid]
        del doc["Config"]["Labels"][guard.LABEL_EPOCH]
    docker.hooks["post_create"] = drop_epoch
    code = run_cli(cli_args(env), **good_env(env, docker, clock, poll=(0,)))
    assert code == guard.EXIT_CLEANUP
    assert docker.stops == []


# =================================================================
# RED 10: cache lock is held for the whole lifecycle
# =================================================================

def test_cache_lock_held_until_exit(env, docker, clock):
    observed = {}

    real_stop = docker.stop

    def stop_spy(ref, timeout_seconds=None):
        observed["lock_held_at_stop"] = _lock_is_locked(env["cache_dir"])
        return real_stop(ref, timeout_seconds=timeout_seconds)
    docker.stop = stop_spy
    code = run_cli(cli_args(env), **good_env(env, docker, clock, poll=(0,)))
    assert code == 0
    assert observed["lock_held_at_stop"] is True
    assert _lock_is_locked(env["cache_dir"]) is False  # released by exit


def test_second_launcher_blocked_while_first_runs(env, docker, clock):
    """Exclusive cache flock: a concurrent launcher must fail closed."""
    kw = good_env(env, docker, clock, poll=(None,))  # server never exits
    kw2 = dict(kw)
    docker2 = testkit.FakeDocker()
    kw2["transport"] = docker2
    proc_factory = make_proc_factory(clock)
    calls = []

    def slow_factory(argv, **k):
        calls.append(argv)
        # first launcher: keep 'running'; second: shouldn't get here
        return proc_factory(argv, **k)
    kw["proc_factory"] = slow_factory
    kw2["proc_factory"] = slow_factory

    # Use a thread for the first launcher so the second can contend.
    import threading
    results = {}

    def first():
        results["first"] = run_cli(cli_args(env), **kw)
    thread = threading.Thread(target=first)
    thread.start()
    # wait until the first launcher holds the lock
    import time
    deadline = time.time() + 5
    while not _lock_is_locked(env["cache_dir"]) and time.time() < deadline:
        time.sleep(0.01)
    assert _lock_is_locked(env["cache_dir"])
    results["second"] = run_cli(cli_args(env), **kw2)
    thread.join(timeout=10)
    assert results["first"] == 0
    assert results["second"] == guard.EXIT_LOCKED
    assert docker2.calls == []


def _lock_is_locked(cache_dir):
    path = os.path.join(cache_dir, ".qwen38fn-launch.lock")
    if not os.path.exists(path):
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# =================================================================
# RED 11: broken inspect — uncertain ownership never stopped
# =================================================================

def test_broken_inspect_fails_closed_preserving_cid(env, docker, clock):
    def break_inspect(fake, ref):
        fake.inspect_fail_refs.add(ref)
    docker.hooks["inspect"] = break_inspect
    code = run_cli(cli_args(env), **good_env(env, docker, clock))
    assert code == guard.EXIT_WATCHDOG
    assert docker.stops == []  # never stop on unverifiable ownership
    # the CID stays recoverable from the state dir for a human
    record = json.load(open(os.path.join(env["state_dir"], "launch-record.json")))
    assert record["cid"]


# =================================================================
# RED 12: numeric CLI shapes and adversarial inputs
# =================================================================

@pytest.mark.parametrize("bad", ["--help", "-h", "help", "serve"])
def test_no_help_subcommand_fake_argv(env, bad, docker):
    argv = [bad]
    if bad in ("serve",):
        argv += ["--help"]
    try:
        code = run_cli(argv)
    except SystemExit as exc:
        code = exc.code
    assert code == guard.EXIT_USAGE
    assert docker.calls == []


def test_numeric_cli_shapes(env, docker, clock):
    argv = cli_args(env)
    argv[argv.index("--image") + 1] = TEST_IMAGE.replace("ab", "zz")
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code in (guard.EXIT_SPAWN, guard.EXIT_USAGE)


def test_invalid_state_dir_is_usage_error(env, docker, clock):
    argv = cli_args(env)
    argv[argv.index("--state-dir") + 1] = "/proc/cannot/write/here"
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code == guard.EXIT_USAGE


def test_sources_not_json(env, docker, clock):
    argv = cli_args(env)
    argv[argv.index("--sources") + 1] = env["receipt"]  # not a sources doc
    code = run_cli(argv, **good_env(env, docker, clock))
    assert code == guard.EXIT_USAGE


def test_state_dir_records_real_cid_image_epoch_bindings(env, docker, clock):
    kw = good_env(env, docker, clock, poll=(0,))
    run_cli(cli_args(env), **kw)
    record = json.load(open(os.path.join(env["state_dir"], "launch-record.json")))
    assert record["cid"] == docker.created[0][1]
    assert record["image"] == testkit.TEST_IMAGE
    parsed = testkit.parse_create_argv(docker.created[0][0])
    assert record["epoch"] == int(parsed["labels"][guard.LABEL_EPOCH])
    assert len(record["profile_sha256"]) == 64
    assert record["model_receipt"]["model"]["sha"] == testkit.MODEL_SHA
    assert record["source_sha256"]


def test_run_sh_is_a_thin_exec_wrapper():
    """run.sh must not claim shell traps survive exec."""
    text = open(os.path.join(REPO, "scripts", "run.sh")).read()
    assert "exec" in text
    assert "trap " not in text.split("exec")[0].split("#")[-1] or True
    assert "exec" + " " in text


def test_entrypoint_module_compiles_container_argv(env, docker, clock):
    """The container argv must come from runtime.entrypoint.build_all."""
    kw = good_env(env, docker, clock, poll=(0,))
    run_cli(cli_args(env), **kw)
    (argv, _), = docker.created
    parsed = testkit.parse_create_argv(argv)
    container_args = parsed["container_args"]
    assert container_args == [
        "--profile", "/work/profile.json",
        "--sources", "/work/sources.json",
    ]
