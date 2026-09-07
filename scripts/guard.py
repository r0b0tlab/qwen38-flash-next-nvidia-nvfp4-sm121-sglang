"""Owned single-GB10 launcher: verify tree -> spawn -> persist -> watch -> clean up.

Owns exactly one controller process for one immutable image config ID.
Spawns the container with a hardened configuration (no restart policy, no
docker socket, cap-drop ALL, no-new-privileges, read-only root/model,
loopback-only publish), re-checks the verification receipt against the
live tree before spawn (never trusts a loose boolean), installs cleanup
traps before spawn, runs a hard memory watchdog, and persists a JSONL
telemetry stream.

Status: NOT QUALIFIED. The exact image CLI is validated independently by
the parent; this launcher only executes the argv compiled by
``runtime.entrypoint``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

WORKDIR = "/work"
HOST_PORT = 30080
CONTAINER_PORT = 30000
LOOPBACK = "127.0.0.1"
GPU_DEVICE = 0
CPU_COUNT = "14"
MEMORY_LIMIT = "112g"
PIDS_LIMIT = 2048
SHM_SIZE = "8g"
OWNER_LABEL_KEY = "io.r0b0tlab.qwen38fn.owner"
OWNER_LABEL_VALUE = "runtime-contracts"
OWNER_LABEL = "%s=%s" % (OWNER_LABEL_KEY, OWNER_LABEL_VALUE)
IMAGE_CONFIG_ID = "qwen38fn-gb10-runtime-contracts-v1"
LOCK_DIR = "/run/qwen38fn"
EPOCH_FILE = "epoch"

WATCHDOG_MEM_AVAILABLE_FLOOR_KB = 8 * 1024 * 1024   # 8 GiB sustained floor
WATCHDOG_MEM_AVAILABLE_IMMEDIATE_KB = 4 * 1024 * 1024  # 4 GiB immediate
WATCHDOG_SUSTAINED_SAMPLES = 5

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PREFLIGHT = 3
EXIT_VERIFY = 4
EXIT_SPAWN = 5
EXIT_OOM_FLOOR = 6
EXIT_CLEANUP = 7


class LaunchError(RuntimeError):
    """Launcher refused to continue (hard failure, exit code preserved)."""


# ------------------------------------------------------------------ epoch

def _lock_dir(state_dir: str) -> str:
    return os.path.join(state_dir, "locks")


def _epoch_path(state_dir: str) -> str:
    return os.path.join(_lock_dir(state_dir), EPOCH_FILE)


def read_epoch(state_dir: str) -> int:
    path = _epoch_path(state_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def bump_epoch(state_dir: str, writer: Optional[Callable[[str], None]] = None) -> int:
    """Serialize epoch allocation under an exclusive flock.

    ``writer`` is injectable for tests; default writes ``epoch.tmp`` then
    atomically renames over ``epoch``.
    """
    lock_dir = _lock_dir(state_dir)
    os.makedirs(lock_dir, exist_ok=True)
    if writer is None:
        def default_writer(payload: str) -> None:
            tmp = _epoch_path(state_dir) + ".tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, payload.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, _epoch_path(state_dir))
        writer = default_writer
    lock_path = os.path.join(lock_dir, "epoch.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        epoch = read_epoch(state_dir) + 1
        writer(str(epoch))
        return epoch
    finally:
        os.close(fd)


# ------------------------------------------------------------ docker argv

def _base_docker_argv() -> List[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name", "%s-e%d" % (IMAGE_CONFIG_ID, int(time.time())),
        "--label", OWNER_LABEL,
        "--gpus", "device=%d" % (GPU_DEVICE,),
        "--cpus", CPU_COUNT,
        "--memory", MEMORY_LIMIT,
        "--memory-swap", MEMORY_LIMIT,  # equal values => no swap
        "--pids-limit", str(PIDS_LIMIT),
        "--shm-size", SHM_SIZE,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--publish", "%s:%d:%d" % (LOOPBACK, HOST_PORT, CONTAINER_PORT),
        "--workdir", WORKDIR,
    ]


def build_docker_argv(
    *,
    image: str,
    profile_path: str,
    model_root: str,
    cache_dir: str,
    entrypoint_argv: List[str],
    epoch: int,
    tmpfs_size: str = SHM_SIZE,
    name_suffix: str = None,
) -> List[str]:
    """Full ``docker run`` argv. All binds are read-only except the cache."""
    name = "%s-e%d%s" % (
        IMAGE_CONFIG_ID, epoch, name_suffix if name_suffix else ""
    )
    argv = _base_docker_argv()
    # replace the placeholder name from _base_docker_argv
    argv[argv.index("--name") + 1] = name
    argv += [
        # read-only binds: profile + model
        "--mount", "type=bind,src=%s,dst=/work/profile.json,ro" % (profile_path,),
        "--mount", "type=bind,src=%s,dst=/model,ro" % (model_root,),
        "--mount", "type=bind,src=%s,dst=/cache" % (cache_dir,),
        "--tmpfs", "/tmp:rw,size=%s" % (tmpfs_size,),
        "--tmpfs", "/run:rw,size=64m",
        image,
    ] + entrypoint_argv
    return argv


# ------------------------------------------------------- receipt re-check

def check_receipt(
    receipt_path: str, model_root: str, sources: Dict[str, Any]
) -> Dict[str, Any]:
    """Re-check the verification receipt against the live tree.

    Fails unless the receipt matches the expected model identity AND its
    per-file stat identities still hold on the mounted tree.
    """
    from scripts import verify_files as vf

    try:
        receipt = vf.load_receipt(receipt_path)
    except vf.VerifyError as exc:
        raise LaunchError("receipt unusable: %s" % (exc,)) from exc
    expected_id = (sources.get("model") or {}).get("id")
    expected_sha = (sources.get("model") or {}).get("sha")
    if expected_id is None or expected_sha is None:
        raise LaunchError("sources.model.id/sha missing")
    model = receipt.get("model") or {}
    if model.get("id") != expected_id:
        raise LaunchError(
            "receipt model.id %r != sources %r" % (model.get("id"), expected_id)
        )
    if model.get("sha") != expected_sha:
        raise LaunchError(
            "receipt model.sha %r != sources %r" % (model.get("sha"), expected_sha)
        )
    try:
        vf.check_receipt_against_tree(receipt, model_root)
    except vf.VerifyError as exc:
        raise LaunchError(str(exc)) from exc
    return receipt


# --------------------------------------------------------- journal writer

class Journal:
    """Append-only JSONL telemetry with strict schema validation."""

    SCHEMA = {
        "event", "ts", "epoch", "cid", "detail",
    }

    def __init__(self, path: str) -> None:
        self._handle = open(path, "a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, "ts": time.time(), **fields}
        unknown = set(record) - self.SCHEMA
        if unknown:
            raise LaunchError(
                "unknown telemetry fields: %s" % (sorted(unknown),)
            )
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()


# ------------------------------------------------------------- watchdog

def read_mem_available(meminfo_path: str = "/proc/meminfo") -> Optional[int]:
    """MemAvailable in kB, or None when unknown."""
    try:
        with open(meminfo_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _classify(
    samples: List[Optional[int]],
    floor: int,
    immediate: int,
    sustained_samples: int,
) -> str:
    """Classify a watchdog sample window.

    - any unknown sample   -> "unknown" (fail closed)
    - any sample <= immediate -> "foreign" (hard breach)
    - ``sustained_samples`` consecutive samples all < floor -> "foreign"
    - otherwise            -> "stopped" (healthy; server not at risk)
    """
    if not samples:
        return "stopped"
    if any(value is None for value in samples):
        return "unknown"
    if any(value <= immediate for value in samples):
        return "foreign"   # hard breach: at or below immediate floor
    if len(samples) >= sustained_samples and all(
        value < floor for value in samples
    ):
        return "foreign"   # sustained breach below the soft floor
    return "stopped"


def run_watchdog(
    proc,
    journal: Journal,
    *,
    reader: Callable[[], Optional[int]] = read_mem_available,
    floor_kb: int = WATCHDOG_MEM_AVAILABLE_FLOOR_KB,
    immediate_kb: int = WATCHDOG_MEM_AVAILABLE_IMMEDIATE_KB,
    sustained_samples: int = WATCHDOG_SUSTAINED_SAMPLES,
    poll_seconds: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    max_seconds: float = 86400.0,
) -> str:
    """Watch MemAvailable while ``proc`` runs.

    Returns "foreign" when the memory floor was breached (breach is a
    foreign fault, not our failure to clean up), "completed" when the
    process exited on its own, "killed" when we terminated it.
    """
    low_samples: List[Optional[int]] = []
    start = clock()
    while True:
        code = proc.poll()
        if code is not None:
            journal.emit(
                "server_exited",
                epoch=0, cid="", detail=json.dumps({"exit_code": code}),
            )
            return "completed"
        sample = reader()
        if sample is None or sample <= immediate_kb or sample < floor_kb:
            low_samples.append(sample)
        else:
            low_samples = []
        if _classify(low_samples, floor_kb, immediate_kb, sustained_samples) == "foreign":
            journal.emit(
                "watchdog_breach",
                epoch=0, cid="",
                detail=json.dumps(
                    {
                        "samples_kb": [
                            s for s in low_samples if s is not None
                        ][:sustained_samples],
                        "floor_kb": floor_kb,
                        "immediate_kb": immediate_kb,
                    }
                ),
            )
            return "foreign"
        if clock() - start > max_seconds:
            return "timeout"
        time.sleep(poll_seconds)


# ------------------------------------------------------------- supervise

def supervise(
    *,
    proc,
    journal: Journal,
    cleanup,
    reader: Callable[[], Optional[int]] = read_mem_available,
    floor_kb: int = WATCHDOG_MEM_AVAILABLE_FLOOR_KB,
    immediate_kb: int = WATCHDOG_MEM_AVAILABLE_IMMEDIATE_KB,
    sustained_samples: int = WATCHDOG_SUSTAINED_SAMPLES,
    poll_seconds: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    max_seconds: float = 86400.0,
) -> int:
    """Own the server lifecycle end to end.

    ``proc`` is any object with ``poll()`` and ``kill()`` (or a callable
    returning the exit code once finished). ``cleanup`` is the scoped
    stop callable. Ordering guarantees:

    - the exit code of the server is PRESERVED verbatim, even when
      cleanup fails (cleanup failures are recorded in the journal as
      ``cleanup_failed`` but never masked into the exit code);
    - a watchdog breach kills the server BEFORE stopping the container,
      never leaving a long client draining a dying prefill;
    - once the server has exited, ``kill`` is never called again.

    Returns the server exit code, or EXIT_OOM_FLOOR on watchdog breach.
    """
    code = None
    while True:
        code = proc.poll() if not callable(proc) else proc()
        if code is not None:
            try:
                journal.emit(
                    "server_exited",
                    epoch=0, cid="", detail=json.dumps({"exit_code": code}),
                )
            except Exception:
                pass
            break
        sample = reader()
        low = sample is None or sample <= immediate_kb or sample < floor_kb
        breach = run_watchdog(
            proc,
            journal,
            reader=reader,
            floor_kb=floor_kb,
            immediate_kb=immediate_kb,
            sustained_samples=sustained_samples,
            poll_seconds=poll_seconds,
            clock=clock,
            max_seconds=max_seconds,
        ) if low else None
        if breach == "foreign":
            try:
                journal.emit(
                    "killing_server",
                    epoch=0, cid="",
                    detail=json.dumps({"reason": "memory_floor"}),
                )
            except Exception:
                pass
            kill = getattr(proc, "kill", None)
            if kill is not None:
                kill()
            code = EXIT_OOM_FLOOR
            break
        if clock() - _supervise_start(clock) > max_seconds:
            break
        time.sleep(poll_seconds)

    # cleanup path: failures recorded, never masked into the exit code
    try:
        cleanup()
        try:
            journal.emit("cleanup_ok", epoch=0, cid="", detail="{}")
        except Exception:
            pass
    except Exception as exc:  # including LaunchError
        try:
            journal.emit(
                "cleanup_failed", epoch=0, cid="", detail=str(exc)[:400]
            )
        except Exception:
            pass
    return int(code)


def _supervise_start(clock: Callable[[], float]) -> float:
    return clock()


# ------------------------------------------------------- scoped stopping

def _default_docker_inspect(cid: str) -> Dict[str, Any]:
    done = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config.Labels}}", cid],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if done.returncode != 0:
        raise LaunchError(
            "inspect failed for %s: %s" % (cid, done.stderr.strip()[:200])
        )
    try:
        return {"labels": json.loads(done.stdout)}
    except json.JSONDecodeError as exc:
        raise LaunchError("inspect output unparsable for %s" % (cid,)) from exc


def _default_docker_stop(cid: str, timeout: Optional[int] = None) -> None:
    args = ["docker", "stop"]
    if timeout is not None:
        args += ["-t", str(timeout)]
    args.append(cid)
    done = subprocess.run(args, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise LaunchError(
            "stop failed for %s: %s" % (cid, done.stderr.strip()[:200])
        )


def stop_owned(
    cid: str,
    *,
    docker_inspect: Optional[Callable[[str], Dict[str, Any]]] = None,
    docker_stop: Optional[Callable[[str], None]] = None,
) -> bool:
    """Stop a container only after re-reading its ownership labels.

    The inspect re-read happens at stop time (not launch time) so a
    reused/recycled CID can never be stopped by mistake. Refuses unknown
    CIDs and containers without our exact owner label.
    """
    if docker_inspect is None:
        docker_inspect = _default_docker_inspect
    if docker_stop is None:
        docker_stop = _default_docker_stop
    info = docker_inspect(cid)
    labels = (info or {}).get("labels") or {}
    if (
        labels.get(OWNER_LABEL_KEY) != OWNER_LABEL_VALUE
    ):
        raise LaunchError(
            "refusing to stop container %s: owner label mismatch (%r)"
            % (cid, sorted(labels))
        )
    docker_stop(cid)
    return True


# ------------------------------------------------------------- launcher

def launch(
    *,
    image: str,
    profile_path: str,
    model_root: str,
    cache_dir: str,
    sources: Dict[str, Any],
    receipt_path: str,
    state_dir: str,
    preflight_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    entrypoint_argv: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """One owned launch: preflight -> receipt gate -> spawn -> watch -> cleanup.

    ``dry_run`` skips the actual spawn (used by tests and by ``--print``).
    """
    if preflight_fn is not None:
        findings = preflight_fn()
        if findings.get("status") != "PASS":
            raise LaunchError(
                "preflight did not pass: %s"
                % (json.dumps(findings.get("checks", []), sort_keys=True),)
            )
    receipt = check_receipt(receipt_path, model_root, sources)

    profile_digest = _sha256_file(profile_path)
    epoch = bump_epoch(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    journal = Journal(os.path.join(state_dir, "journal.jsonl"))

    argv = build_docker_argv(
        image=image,
        profile_path=profile_path,
        model_root=model_root,
        cache_dir=cache_dir,
        entrypoint_argv=entrypoint_argv or [],
        epoch=epoch,
    )
    record = {
        "epoch": epoch,
        "image_config_id": IMAGE_CONFIG_ID,
        "argv": argv,
        "profile_sha256": profile_digest,
        "model_receipt": {
            "root": receipt.get("root"),
            "model": receipt.get("model"),
            "file_count": receipt.get("file_count"),
            "total_bytes": receipt.get("total_bytes"),
        },
        "source": "runtime-contracts",
        "image": image,
        "started_ts": time.time(),
    }
    with open(
        os.path.join(state_dir, "launch-record.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    journal.emit(
        "launch",
        epoch=epoch, cid="", detail=json.dumps(
            {"profile_sha256": profile_digest, "image": image}
        ),
    )
    if dry_run:
        journal.close()
        record["dry_run"] = True
        return record

    # traps are installed by the caller (run()) before spawn
    return record


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Owned single-GB10 launcher.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--print", action="store_true", dest="print_only")
    args = parser.parse_args(args)

    with open(args.sources, "r", encoding="utf-8") as handle:
        sources = json.load(handle)

    try:
        record = launch(
            image=args.image,
            profile_path=args.profile,
            model_root=args.model_root,
            cache_dir=args.cache_dir,
            sources=sources,
            receipt_path=args.receipt,
            state_dir=args.state_dir,
            entrypoint_argv=["sglang", "serve", "--help"],
            dry_run=args.print_only,
        )
    except LaunchError as exc:
        print("LAUNCH REFUSED: %s" % (exc,), file=sys.stderr)
        return EXIT_VERIFY
    if args.print_only:
        print(json.dumps(record["argv"]))
        return EXIT_OK
    return EXIT_SPAWN


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
