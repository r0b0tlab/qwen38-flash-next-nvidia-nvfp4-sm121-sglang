#!/usr/bin/env python3
"""Repository-owned bounded driver for untrusted Q200 Python candidates.

The host runner supplies one strict JSON request on stdin.  This process is
inside the admitted image and is the only component allowed to turn candidate
execution into a receipt.  Candidate stdout/stderr are drained incrementally
with bounded tails.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

DRIVER_VERSION = "r0b0tlab.qwen38.q200_sandbox_driver.v1"
REQUEST_SCHEMA = "r0b0tlab.qwen38.q200_sandbox_request.v1"
RECEIPT_SCHEMA = "r0b0tlab.qwen38.q200_sandbox_receipt.v1"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_CANDIDATE_BYTES = 256 * 1024
MAX_TEST_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_OUTPUT_TAIL_BYTES = 4096
PIPE_SETTLE_SECONDS = 0.20
TERMINATE_GRACE_SECONDS = 0.15
ENTRY_POINT_RE = re.compile(r"^[A-Za-z_]\w{0,127}$")
COMPLETION_MARKER = b"Q200-CANDIDATE-COMPLETE\n"

# Keep this manifest literal and identical to the host-side manifest.  It is
# included in every receipt so changing a limit cannot silently reuse a result.
RUNTIME_MANIFEST: dict[str, Any] = {
    "schema": "r0b0tlab.qwen38.q200_sandbox_runtime.v1",
    "cpu_seconds": 9,
    "address_space_bytes": 256 * 1024 * 1024,
    "file_size_bytes": 1024 * 1024,
    "nproc": 32,
    "nofile": 64,
    "output_bytes": MAX_OUTPUT_BYTES,
    "tmpfs_bytes": 64 * 1024 * 1024,
    "network": "none",
    "read_only_root": True,
    "uid_gid": "65532:65532",
}
REQUEST_KEYS = {
    "schema",
    "driver_version",
    "driver_source_sha256",
    "image_id",
    "candidate",
    "reference",
    "timeout_seconds",
    "runtime_manifest",
    "request_sha256",
}


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_load(raw: bytes | str) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical(value))


def _source_hash() -> str:
    return _sha256(Path(__file__).read_bytes())


def _bounded_text(value: Any, limit: int, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) > limit:
        raise ValueError(f"{name} exceeds {limit} bytes")
    return value


def _validate_request(request: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(request, Mapping) or set(request) != REQUEST_KEYS:
        raise ValueError("request schema is incomplete or has unknown fields")
    if request.get("schema") != REQUEST_SCHEMA or request.get("driver_version") != DRIVER_VERSION:
        raise ValueError("request schema or driver version mismatch")
    source_hash = request.get("driver_source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("invalid driver source hash")
    if source_hash != _source_hash():
        raise ValueError("driver source hash mismatch")
    image_id = request.get("image_id")
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("invalid immutable image ID")
    candidate = _bounded_text(request.get("candidate"), MAX_CANDIDATE_BYTES, "candidate")
    reference = request.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("reference must be an object")
    if not set(reference) <= {"entry_point", "canonical", "test"}:
        raise ValueError("reference contains unknown fields")
    if "test" in reference:
        _bounded_text(reference["test"], MAX_TEST_BYTES, "reference.test")
    if "canonical" in reference:
        _bounded_text(reference["canonical"], MAX_CANDIDATE_BYTES, "reference.canonical")
    if "entry_point" in reference:
        entry = reference["entry_point"]
        if not isinstance(entry, str) or not ENTRY_POINT_RE.fullmatch(entry):
            raise ValueError("reference.entry_point is not a safe Python identifier")
    if not any(key in reference for key in ("test", "entry_point")):
        raise ValueError("reference must contain test or entry_point")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or not 0 < float(timeout) <= 30:
        raise ValueError("timeout_seconds is outside the admitted range")
    runtime_manifest = request.get("runtime_manifest")
    if runtime_manifest != RUNTIME_MANIFEST:
        raise ValueError("runtime manifest mismatch")
    request_for_hash = dict(request)
    supplied_hash = request_for_hash.pop("request_sha256")
    if not isinstance(supplied_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied_hash):
        raise ValueError("invalid request hash")
    if _sha256_json(request_for_hash) != supplied_hash:
        raise ValueError("request hash mismatch")
    return dict(request), supplied_hash


def _child_limits() -> None:
    """Apply process limits in the candidate child before Python starts."""
    os.setsid()
    limits = (
        (resource.RLIMIT_CPU, RUNTIME_MANIFEST["cpu_seconds"], RUNTIME_MANIFEST["cpu_seconds"]),
        (resource.RLIMIT_AS, RUNTIME_MANIFEST["address_space_bytes"], RUNTIME_MANIFEST["address_space_bytes"]),
        (resource.RLIMIT_FSIZE, RUNTIME_MANIFEST["file_size_bytes"], RUNTIME_MANIFEST["file_size_bytes"]),
        (resource.RLIMIT_NPROC, RUNTIME_MANIFEST["nproc"], RUNTIME_MANIFEST["nproc"]),
        (resource.RLIMIT_NOFILE, RUNTIME_MANIFEST["nofile"], RUNTIME_MANIFEST["nofile"]),
        (resource.RLIMIT_CORE, 0, 0),
    )
    for kind, soft, hard in limits:
        try:
            resource.setrlimit(kind, (int(soft), int(hard)))
        except (OSError, ValueError):
            # A missing optional limit is not allowed to weaken a required
            # one.  Linux images used for Q200 provide all of these limits.
            if kind in {resource.RLIMIT_CPU, resource.RLIMIT_AS, resource.RLIMIT_FSIZE, resource.RLIMIT_NPROC}:
                raise
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        # Container cleanup remains the outer backstop if prctl is unavailable.
        pass


def _enable_subreaper() -> None:
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(36, 1)  # PR_SET_CHILD_SUBREAPER
    except Exception:
        pass


def _children(pid: int) -> set[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text(encoding="ascii")
    except (FileNotFoundError, OSError, UnicodeError):
        return set()
    result: set[int] = set()
    for token in raw.split():
        try:
            child = int(token)
        except ValueError:
            continue
        if child > 1:
            result.add(child)
            result.update(_children(child))
    return result


def _alive(pids: set[int]) -> set[int]:
    result: set[int] = set()
    for pid in pids:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue
        except OSError:
            continue
        result.add(pid)
    return result


def _terminate(proc: subprocess.Popen[bytes], owned: set[int]) -> None:
    targets = set(owned)
    targets.add(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None and not _alive(targets):
            break
        time.sleep(0.01)
    for pid in targets | _alive(targets):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


def _tail_append(tail: bytearray, chunk: bytes) -> None:
    tail.extend(chunk)
    if len(tail) > MAX_OUTPUT_TAIL_BYTES:
        del tail[: len(tail) - MAX_OUTPUT_TAIL_BYTES]


def _run_candidate(candidate: str, reference: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    test = reference.get("test")
    entry = reference.get("entry_point")
    harness = candidate
    if isinstance(test, str) and test.strip():
        harness += "\n" + test + "\n"
        if isinstance(entry, str) and entry and not re.search(rf"\bcheck\s*\(\s*{re.escape(entry)}\s*\)", test):
            harness += f"check({entry})\n"
    elif isinstance(entry, str) and entry:
        harness += f"\nassert callable({entry})\n"

    completion_read, completion_write = os.pipe()
    os.set_inheritable(completion_write, True)
    # A completion marker travels over a private descriptor, not candidate
    # stdout.  os._exit(), signal death, and test aborts therefore cannot look
    # successful merely by returning code zero.
    harness += (
        "\nimport os as __q200_os\n"
        f"__q200_os.write({completion_write}, {COMPLETION_MARKER!r})\n"
        f"__q200_os.close({completion_write})\n"
    )
    candidate_path = Path("/workspace/candidate.py")
    candidate_path.write_text(harness, encoding="utf-8")
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    proc: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    owned: set[int] = set()
    stdout_tail = bytearray()
    stderr_tail = bytearray()
    completion = bytearray()
    stdout_count = stderr_count = 0
    status = "ok"
    failure_reason: str | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", str(candidate_path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/workspace",
            env=env,
            close_fds=True,
            pass_fds=(completion_write,),
            preexec_fn=_child_limits,
        )
        os.close(completion_write)
        completion_write = -1
        assert proc.stdout is not None and proc.stderr is not None
        os.set_blocking(proc.stdout.fileno(), False)
        os.set_blocking(proc.stderr.fileno(), False)
        os.set_blocking(completion_read, False)
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        selector.register(completion_read, selectors.EVENT_READ, "completion")
        deadline = time.monotonic() + timeout
        pipe_deadline: float | None = None
        while True:
            now = time.monotonic()
            owned.update(_children(proc.pid))
            if now >= deadline and proc.poll() is None:
                status, failure_reason = "timeout", "candidate wall-time limit exceeded"
                _terminate(proc, owned)
                break
            if proc.poll() is not None and pipe_deadline is None:
                pipe_deadline = now + PIPE_SETTLE_SECONDS
            if pipe_deadline is not None and now >= pipe_deadline:
                live = _alive(owned)
                if live or any(key in {"stdout", "stderr"} for key in (key.data for key in selector.get_map().values())):
                    status, failure_reason = "failed", "retained candidate process or pipe"
                    _terminate(proc, owned)
                break
            events = selector.select(max(0.01, min(0.05, (deadline - now) if pipe_deadline is None else (pipe_deadline - now))))
            for key, _mask in events:
                label = key.data
                fd = key.fileobj if isinstance(key.fileobj, int) else key.fileobj.fileno()
                try:
                    chunk = os.read(fd, 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(key.fileobj)
                    except Exception:
                        pass
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    continue
                if label == "stdout":
                    stdout_count += len(chunk)
                    _tail_append(stdout_tail, chunk)
                    if stdout_count > MAX_OUTPUT_BYTES and status == "ok":
                        status, failure_reason = "output_limit", "candidate stdout exceeded live bound"
                        _terminate(proc, owned)
                elif label == "stderr":
                    stderr_count += len(chunk)
                    _tail_append(stderr_tail, chunk)
                    if stderr_count > MAX_OUTPUT_BYTES and status == "ok":
                        status, failure_reason = "output_limit", "candidate stderr exceeded live bound"
                        _terminate(proc, owned)
                else:
                    completion.extend(chunk)
            if status in {"output_limit", "timeout"}:
                break
            if proc.poll() is not None and not selector.get_map():
                break
        if proc.poll() is None:
            try:
                proc.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _terminate(proc, owned)
        returncode = proc.returncode
        if status == "ok":
            if returncode != 0:
                status = "oom" if returncode in {-signal.SIGKILL, 137} else "failed"
                failure_reason = "candidate exited unsuccessfully"
            elif bytes(completion) != COMPLETION_MARKER:
                status, failure_reason = "failed", "candidate did not reach the trusted completion marker"
    except (OSError, subprocess.SubprocessError) as exc:
        status, failure_reason = "grader_error", f"candidate supervisor error: {type(exc).__name__}"
        if proc is not None:
            _terminate(proc, owned)
        returncode = proc.returncode if proc is not None else None
    finally:
        if completion_write >= 0:
            try:
                os.close(completion_write)
            except OSError:
                pass
        try:
            selector.close()
        except Exception:
            pass
        try:
            os.close(completion_read)
        except OSError:
            pass
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    failure_class = "MODEL_REJECT" if status != "grader_error" and status != "ok" else (None if status == "ok" else "HARNESS_BLOCK")
    return {
        "status": status,
        "passed": status == "ok",
        "failure_class": failure_class,
        "returncode": returncode,
        "stdout_bytes": stdout_count,
        "stderr_bytes": stderr_count,
        "stdout_tail": stdout_tail.decode("utf-8", errors="replace"),
        "stderr_tail": stderr_tail.decode("utf-8", errors="replace"),
        "reason": failure_reason or "",
        "stdin_isolated": True,
    }


def _error_receipt(reason: str, *, image_id: str | None = None, source_hash: str | None = None) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "driver_version": DRIVER_VERSION,
        "driver_source_sha256": source_hash or "",
        "image_id": image_id or "",
        "request_sha256": "",
        "candidate_sha256": "",
        "reference_sha256": "",
        "runtime_manifest_sha256": _sha256_json(RUNTIME_MANIFEST),
        "runtime_manifest": RUNTIME_MANIFEST,
        "status": "grader_error",
        "passed": None,
        "failure_class": "HARNESS_BLOCK",
        "returncode": None,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_tail": "",
        "stderr_tail": "",
        "reason": reason[:256],
        "stdin_isolated": True,
    }


def main() -> int:
    _enable_subreaper()
    receipt: dict[str, Any]
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds byte bound")
        request = _strict_load(raw)
        request, request_hash = _validate_request(request)
        candidate = str(request["candidate"])
        reference = request["reference"]
        result = _run_candidate(candidate, reference, float(request["timeout_seconds"]))
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "driver_version": DRIVER_VERSION,
            "driver_source_sha256": request["driver_source_sha256"],
            "image_id": request["image_id"],
            "request_sha256": request_hash,
            "candidate_sha256": _sha256(candidate.encode("utf-8")),
            "reference_sha256": _sha256_json(reference),
            "runtime_manifest_sha256": _sha256_json(RUNTIME_MANIFEST),
            "runtime_manifest": RUNTIME_MANIFEST,
            **result,
        }
    except Exception as exc:
        receipt = _error_receipt(f"driver validation failed: {type(exc).__name__}")
    output = _canonical(receipt) + b"\n"
    if len(output) > 16 * 1024:
        # This is a driver defect, not a candidate result.  Keep the one-line
        # protocol bounded even if a future receipt field grows unexpectedly.
        receipt = _error_receipt("receipt exceeded bounded protocol size")
        output = _canonical(receipt) + b"\n"
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
