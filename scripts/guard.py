"""One owned Docker lifecycle, with immutable inputs and no GPU fallback.

The optional transport/probes are test seams. The CLI uses the real Docker
transport and the same create/start/watch/stop control flow exercised by tests.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time

# Script mode and module mode resolve this repository, not a site-packages tests/scripts package.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime import profile_from_json
from runtime.entrypoint import build_all
from scripts import preflight

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
OWNER_LABEL_KEY = "io.r0b0tlab.qwen38fn.owner"
LABEL_PROFILE = "io.r0b0tlab.qwen38fn.profile-sha256"
LABEL_IMAGE = "io.r0b0tlab.qwen38fn.image-id"
CACHE_LOCK_NAME = ".qwen38fn-launch.lock"
GIB = 1 << 30
PLE_BYTES = 51200245760
EXIT_OK, EXIT_USAGE, EXIT_PREFLIGHT, EXIT_VERIFY = 0, 2, 3, 4
EXIT_SPAWN, EXIT_CLEANUP, EXIT_WATCHDOG, EXIT_LOCKED = 5, 7, 8, 9


class LaunchError(ValueError):
    pass


class TransportError(RuntimeError):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LaunchError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value):
    raise LaunchError("nonfinite JSON value")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise LaunchError("nonfinite JSON numeric value")
    return parsed


def read_json(path, limit=16 * 1024 * 1024):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise LaunchError("JSON artifact exceeds its byte limit")
    return json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=_constant,
        parse_float=_finite_float,
    ), raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require_hex(value, length, field):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{%d}" % length, value):
        raise LaunchError(field + " has an invalid digest")
    return value


def safe_path(value, *, directory=False):
    path = Path(value)
    if not path.is_absolute() or any(c in str(path) for c in ("\x00", "\n", "\r", ",")):
        raise LaunchError(
            "paths must be absolute and contain no control/comma characters"
        )
    if path.is_symlink() or path.resolve() != path:
        raise LaunchError("symlink/noncanonical path rejected")
    if directory and path in (Path("/"), Path.home()):
        raise LaunchError("root/home cannot be a runtime state/cache directory")
    return path


def source_inventory(sources):
    model = sources.get("model") if isinstance(sources, dict) else None
    if not isinstance(model, dict) or model.get("id") != MODEL_ID:
        raise LaunchError("wrong/missing model identity")
    require_hex(model.get("sha"), 40, "model revision")
    rows = model.get("files")
    if not isinstance(rows, list) or not rows:
        raise LaunchError("sources.model.files must be a nonempty pinned inventory")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise LaunchError("invalid inventory entry")
        name = row.get("path")
        rel = Path(name) if isinstance(name, str) else Path("/")
        if (
            not name
            or rel.is_absolute()
            or ".." in rel.parts
            or "\\" in name
            or str(rel) != name
            or any(ord(c) < 32 for c in name)
            or name in result
        ):
            raise LaunchError("unsafe/duplicate inventory path")
        if type(row.get("size")) is not int or row["size"] < 0:
            raise LaunchError("invalid inventory byte count")
        if row.get("sha256") is not None:
            require_hex(row["sha256"], 64, "file SHA256")
        else:
            require_hex(row.get("git_blob"), 40, "Git blob")
        result[name] = row
    return result


def check_receipt(path, expected_hash, model_root, sources):
    """The receipt hash is an independent verifier output, never self-derived here."""
    require_hex(expected_hash, 64, "verified receipt SHA256")
    receipt, raw = read_json(path)
    if digest(raw) != expected_hash:
        raise LaunchError("receipt differs from the independently verified digest")
    inventory = source_inventory(sources)
    if not isinstance(receipt, dict):
        raise LaunchError("receipt must be an object")
    if receipt.get("kind") != "CHECKPOINT_VERIFIED" or receipt.get("root") != str(
        model_root
    ):
        raise LaunchError("receipt kind/root mismatch")
    if receipt.get("model") != {k: sources["model"][k] for k in ("id", "sha")}:
        raise LaunchError("receipt model identity mismatch")
    rows = receipt.get("files")
    if (
        not isinstance(rows, list)
        or type(receipt.get("file_count")) is not int
        or receipt["file_count"] != len(inventory)
    ):
        raise LaunchError("receipt file count mismatch")
    if any(not isinstance(row, dict) for row in rows):
        raise LaunchError("receipt entries must be objects")
    if len(rows) != len(inventory) or {r.get("path") for r in rows} != set(inventory):
        raise LaunchError("receipt file set mismatch")
    total = 0
    for row in rows:
        expected = inventory[row["path"]]
        path = model_root / row["path"]
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(model_root)
            or not path.is_file()
        ):
            raise LaunchError("checkpoint file escapes root or is not regular")
        stat = path.stat()
        fields = {
            "dev": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
        if any(
            type(row.get(k)) is not int or row[k] != value
            for k, value in fields.items()
        ):
            raise LaunchError("checkpoint stat changed after full verification")
        if row["size"] != expected["size"]:
            raise LaunchError("checkpoint size differs from source lock")
        require_hex(row.get("sha256"), 64, "observed file SHA256")
        if expected.get("sha256") is not None and row["sha256"] != expected["sha256"]:
            raise LaunchError("receipt digest differs from source lock")
        if (
            expected.get("sha256") is None
            and row.get("git_blob") != expected["git_blob"]
        ):
            raise LaunchError("receipt Git blob differs from source lock")
        total += row["size"]
    if type(receipt.get("total_bytes")) is not int or receipt["total_bytes"] != total:
        raise LaunchError("receipt byte total mismatch")
    return receipt


class DockerCliTransport:
    def command(self, args, timeout=30, *, merge_stderr=False):
        try:
            result = subprocess.run(
                ["docker", *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransportError(str(error)) from error
        if result.returncode:
            raise TransportError(
                (result.stderr or result.stdout or "Docker command failed").strip()[
                    :1000
                ]
            )
        return result.stdout.strip()

    def _one_document(self, args):
        try:
            docs = json.loads(
                self.command(args),
                object_pairs_hook=_pairs,
                parse_constant=_constant,
                parse_float=_finite_float,
            )
        except ValueError as error:
            raise TransportError("invalid Docker inspection JSON") from error
        if (
            not isinstance(docs, list)
            or len(docs) != 1
            or not isinstance(docs[0], dict)
        ):
            raise TransportError("Docker inspection must return exactly one object")
        return docs[0]

    def image_inspect(self, image):
        return self._one_document(["image", "inspect", image])

    def create(self, argv, cidfile):
        return self.command(argv, timeout=120)

    def start(self, cid):
        self.command(["start", cid], timeout=120)

    def inspect(self, cid):
        return self._one_document(["container", "inspect", cid])

    def stop(self, cid):
        self.command(["stop", "--time", "120", cid], timeout=150)

    def logs(self, cid):
        # Docker's per-container rotating log limit bounds this final capture.
        return self.command(["logs", cid], timeout=30, merge_stderr=True)


def verify_ownership(doc, expected):
    if doc.get("Id") != expected["cid"] or doc.get("Image") != expected["image"]:
        raise LaunchError("container ID/image ownership mismatch")
    labels = doc.get("Config", {}).get("Labels") or {}
    required = {
        OWNER_LABEL_KEY: expected["nonce"],
        LABEL_PROFILE: expected["profile_sha256"],
        LABEL_IMAGE: expected["image"],
    }
    if any(labels.get(key) != value for key, value in required.items()):
        raise LaunchError("container epoch/profile ownership mismatch")
    return doc


def stop_owned(cid, *, expect, transport=None):
    transport = transport or DockerCliTransport()
    doc = verify_ownership(transport.inspect(cid), {**expect, "cid": cid})
    if doc.get("State", {}).get("Running"):
        transport.stop(cid)
    final = verify_ownership(transport.inspect(cid), {**expect, "cid": cid})
    if final.get("State", {}).get("Running") is not False:
        raise LaunchError("container stop could not be verified")
    return final


@contextmanager
def lifetime_lock(cache):
    path = cache / CACHE_LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _save(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    temp = path.with_name(path.name + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def build_docker_argv(image, profile, sources, model, cache, state, expect):
    return [
        "create",
        "--cidfile",
        str(state / "container.cid"),
        "--name",
        "qwen38fn-" + expect["nonce"],
        "--label",
        OWNER_LABEL_KEY + "=" + expect["nonce"],
        "--label",
        LABEL_PROFILE + "=" + expect["profile_sha256"],
        "--label",
        LABEL_IMAGE + "=" + image,
        "--gpus",
        "device=0",
        "--cpus",
        "14",
        "--memory",
        "112g",
        "--memory-swap",
        "112g",
        "--pids-limit",
        "2048",
        "--shm-size",
        "8g",
        "--cap-drop",
        "ALL",
        "--log-driver",
        "local",
        "--log-opt",
        "max-size=8m",
        "--log-opt",
        "max-file=3",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--publish",
        "127.0.0.1:30080:30000",
        "--mount",
        f"type=bind,src={profile},dst=/work/profile.json,readonly",
        "--mount",
        f"type=bind,src={sources},dst=/work/sources.json,readonly",
        "--mount",
        f"type=bind,src={model},dst=/model,readonly",
        "--mount",
        f"type=bind,src={cache},dst=/cache",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g,mode=1777",
        image,
        "--profile",
        "/work/profile.json",
        "--sources",
        "/work/sources.json",
    ]


def _mem_floor_kb(floor_gib: float) -> int:
    """Watchdog MemAvailable floor in kB from a GiB value."""
    return int(floor_gib * GIB) // 1024


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Owned single-GB10 runtime lifecycle")
    for name in (
        "image",
        "profile",
        "model-root",
        "cache-dir",
        "sources",
        "receipt",
        "receipt-sha256",
        "state-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--print", "--dryrun", action="store_true", dest="print_only")
    parser.add_argument("--max-watch-seconds", type=float, default=0)
    parser.add_argument(
        "--mem-available-floor-gib",
        type=float,
        default=8.0,
        help=(
            "watchdog MemAvailable floor in GiB (5 consecutive samples below it, "
            "or any sample below floor/2, stops the owned container). Release "
            "default 8.0; an explicit override is recorded in launch-record.json."
        ),
    )
    args = parser.parse_args(argv)
    if not 4.0 <= args.mem_available_floor_gib <= 16.0:
        parser.error(
            "--mem-available-floor-gib must be within [4.0, 16.0] GiB "
            "(below 4 GiB the host is genuinely out of memory)"
        )
    return args


def run(
    argv=None,
    *,
    transport=None,
    preflight_fn=None,
    mem_reader=None,
    stop_event=None,
    sleep=time.sleep,
    monotonic=time.monotonic,
):
    args = _parse_args(argv)
    rc, cid, record = EXIT_USAGE, None, {}
    transport = transport or DockerCliTransport()
    event = stop_event or threading.Event()
    old_signals = {}
    interrupted = {"signal": None}
    state = None
    expect = None
    create_attempted = False
    try:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
            raise LaunchError("image must be an exact sha256 config ID")
        if not math.isfinite(args.max_watch_seconds) or args.max_watch_seconds < 0:
            raise LaunchError("invalid watch deadline")
        profile_path = safe_path(args.profile)
        source_path = safe_path(args.sources)
        receipt_path = safe_path(args.receipt)
        model = safe_path(args.model_root)
        cache = safe_path(args.cache_dir, directory=True)
        state = safe_path(args.state_dir, directory=True)
        roots = (state, model, cache)
        if any(
            a.is_relative_to(b) or b.is_relative_to(a)
            for i, a in enumerate(roots)
            for b in roots[i + 1 :]
        ):
            raise LaunchError("state, model and cache must be disjoint")
        sources, source_raw = read_json(source_path)
        inventory = source_inventory(sources)
        _, profile_raw = read_json(profile_path)
        profile = profile_from_json(profile_raw.decode("utf-8"))
        built = build_all(profile, sources)
        expected_hash = require_hex(args.receipt_sha256, 64, "verified receipt SHA256")
        if args.print_only:
            print(
                json.dumps(
                    {"status": "PLAN_ONLY", "server": built, "image": args.image},
                    sort_keys=True,
                )
            )
            return 0
        if not cache.is_dir() or not model.is_dir():
            raise LaunchError("existing cache/model directories required")
        # Fresh per-attempt state; never overwrite a prior epoch's evidence.
        state.mkdir(mode=0o700, parents=True, exist_ok=False)
        with lifetime_lock(cache):
            rc = EXIT_PREFLIGHT
            findings = (
                preflight_fn
                or (
                    lambda: preflight.run_preflight(
                        phase="serve",
                        model_bytes=sum(row["size"] for row in inventory.values()),
                        ple_bytes=PLE_BYTES,
                        model_root=str(model),
                        ple_dir=str(cache / "ple" / sources["model"]["sha"]),
                    )
                )
            )()
            _save(state / "preflight.json", findings)
            if findings.get("status") != "PASS":
                raise LaunchError("host preflight rejected")
            rc = EXIT_VERIFY
            check_receipt(receipt_path, expected_hash, model, sources)
            image = transport.image_inspect(args.image)
            if (image.get("Id"), image.get("Architecture"), image.get("Os")) != (
                args.image,
                "arm64",
                "linux",
            ):
                raise LaunchError("image identity/platform mismatch")
            # Mount validated bytes, not the caller's subsequently mutable paths.
            for name, raw in (
                ("profile.json", profile_raw),
                ("sources.json", source_raw),
            ):
                fd = os.open(state / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
            expect = {
                "nonce": secrets.token_hex(16),
                "image": args.image,
                "profile_sha256": digest(profile_raw),
            }
            command = build_docker_argv(
                args.image,
                state / "profile.json",
                state / "sources.json",
                model,
                cache,
                state,
                expect,
            )
            record = {
                **expect,
                "source_sha256": digest(source_raw),
                "receipt_sha256": expected_hash,
                "model": sources["model"]["id"],
                "model_sha": sources["model"]["sha"],
                "mem_available_floor_gib": args.mem_available_floor_gib,
                "argv": ["docker", *command],
                "status": "STARTING",
            }
            _save(state / "launch-record.json", record)
            if threading.current_thread() is threading.main_thread():

                def on_signal(number, frame):
                    interrupted["signal"] = number
                    event.set()

                for number in (signal.SIGTERM, signal.SIGINT):
                    old_signals[number] = signal.signal(number, on_signal)
            rc = EXIT_SPAWN
            try:
                if event.is_set():
                    rc = 128 + (interrupted["signal"] or signal.SIGTERM)
                else:
                    create_attempted = True
                    cid = transport.create(command, str(state / "container.cid"))
                    require_hex(cid, 64, "container ID")
                    record["cid"] = cid
                    _save(state / "launch-record.json", record)
                    verify_ownership(transport.inspect(cid), {**expect, "cid": cid})
                    if not event.is_set():
                        rc = EXIT_VERIFY
                        check_receipt(receipt_path, expected_hash, model, sources)
                        rc = EXIT_SPAWN
                        transport.start(cid)
                        rc = EXIT_WATCHDOG
                        start = monotonic()
                        low = 0
                        reader = mem_reader or (
                            lambda: (preflight.read_meminfo() or {}).get("MemAvailable")
                        )
                        with (state / "telemetry.jsonl").open(
                            "x", encoding="utf-8", buffering=1
                        ) as log:
                            while True:
                                doc = verify_ownership(
                                    transport.inspect(cid), {**expect, "cid": cid}
                                )
                                current = doc.get("State") or {}
                                if current.get("Running") is False:
                                    value = current.get("ExitCode")
                                    if type(value) is not int or not 0 <= value <= 255:
                                        raise LaunchError("invalid container exit code")
                                    rc = (
                                        value
                                        if value or not current.get("OOMKilled")
                                        else EXIT_WATCHDOG
                                    )
                                    break
                                if current.get("Running") is not True:
                                    raise LaunchError("container running state unknown")
                                if event.is_set():
                                    rc = 128 + (interrupted["signal"] or signal.SIGTERM)
                                    break
                                available = reader()
                                known = type(available) is int and available >= 0
                                log.write(
                                    json.dumps(
                                        {
                                            "time": time.time(),
                                            "cid": cid,
                                            "mem_available_kb": available
                                            if known
                                            else None,
                                        },
                                        allow_nan=False,
                                    )
                                    + "\n"
                                )
                                if not known:
                                    rc = EXIT_WATCHDOG
                                    break
                                low = (
                                    low + 1
                                    if available < _mem_floor_kb(
                                        args.mem_available_floor_gib
                                    )
                                    else 0
                                )
                                if (
                                    available
                                    < _mem_floor_kb(args.mem_available_floor_gib) // 2
                                    or low >= 5
                                ):
                                    rc = EXIT_WATCHDOG
                                    break
                                if (
                                    args.max_watch_seconds
                                    and monotonic() - start >= args.max_watch_seconds
                                ):
                                    rc = EXIT_WATCHDOG
                                    break
                                sleep(1)
                    else:
                        rc = 128 + (interrupted["signal"] or signal.SIGTERM)
            except (OSError, ValueError, TransportError) as error:
                record["error"] = str(error)[:1500]
            finally:
                # Resolve an uncertain create from its exact cidfile/name before
                # cleanup. Never kill the client and abandon an owned GPU request.
                if cid is None and create_attempted:
                    cidfile = state / "container.cid"
                    if cidfile.is_file():
                        cid = cidfile.read_text().strip()
                    else:
                        try:
                            doc = transport.inspect("qwen38fn-" + expect["nonce"])
                            candidate = doc.get("Id")
                            require_hex(candidate, 64, "recovered container ID")
                            verify_ownership(doc, {**expect, "cid": candidate})
                            cid = candidate
                        except (LaunchError, TransportError):
                            record["create_outcome"] = "UNKNOWN"
                if cid:
                    try:
                        require_hex(cid, 64, "cleanup container ID")
                        final = stop_owned(cid, expect=expect, transport=transport)
                        _save(state / "container.final.json", final)
                        (state / "server.log").write_text(transport.logs(cid))
                    except (OSError, ValueError, TransportError) as error:
                        record["cleanup_error"] = str(error)[:1500]
                        rc = rc or EXIT_CLEANUP
                record.update(
                    status="STOPPED" if rc == 0 else "FAILED", exit_code=rc, cid=cid
                )
                _save(state / "launch-record.json", record)
    except BlockingIOError:
        rc = EXIT_LOCKED
    except (OSError, ValueError, TransportError) as error:
        rc = rc or EXIT_CLEANUP
        print("LAUNCH REFUSED/FAILED: " + str(error), file=sys.stderr)
    finally:
        for number, handler in old_signals.items():
            signal.signal(number, handler)
    return rc


if __name__ == "__main__":
    raise SystemExit(run())
