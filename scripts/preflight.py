"""Prelaunch host admission for the single-GB10 serve.

Checks, failing closed on anything unknown:

- exactly one GB10 GPU, idle (graphics/Xorg processes are fine, compute
  apps are not), aarch64 host, prelaunch available RAM >= 104 GiB;
- per underlying filesystem: required bytes = missing model bytes +
  missing file-backed PLE bytes + build allowance (build phase only)
  + a 32 GiB reserve — grouped per filesystem so shared filesystems are
  never double charged;
- PLE/model roots must live on local NVMe: FUSE/NFS/network filesystems
  are rejected for the PLE target.

Phases: ``build`` charges image/build scratch allowance; ``serve`` does
not, so a warm restart with the image and caches already present is not
rejected for allocations it no longer needs. Nothing is ever deleted by
this module and no sysctl/driver changes are attempted.

All probes are injectable module functions so tests can stub the host
without touching real paths.

Status: NOT QUALIFIED.
"""

from __future__ import annotations

import os
import math
import re
import platform
from typing import Any, Callable, Dict, List, Optional

PREFLIGHT_PASS = "PASS"
PREFLIGHT_FAIL = "FAIL"

PHASE_BUILD = "build"
PHASE_SERVE = "serve"

RAM_AVAILABLE_MIN_GIB = 104.0
DISK_RESERVE_GIB = 32.0
BUILD_ALLOWANCE_GIB_DEFAULT = 0.0
GIB = 1024 ** 3

#: filesystem types never acceptable for the PLE/model roots
REJECT_FILESYSTEMS = frozenset(
    {"fuse", "fuseblk", "nfs", "nfs4", "cifs", "smbfs", "sshfs", "nfsd", "9p"}
)

ARCH_EXPECTED = "aarch64"

#: GPU may hold this much memory and still count as idle (ECC/display)
GPU_IDLE_MEMORY_MIB_MAX = 256.0

MeminfoReader = Callable[[], Optional[Dict[str, int]]]
GpuProber = Callable[[], Optional[Dict[str, Any]]]
Statvfser = Callable[[str], Any]
MountinfoReader = Callable[[], Optional[List[Dict[str, str]]]]


class PreflightError(ValueError):
    """Preflight could not be completed (unknowns fail closed)."""


# ---------------------------------------------------------------- probes

def read_meminfo(path: str = "/proc/meminfo") -> Optional[Dict[str, int]]:
    """kB-valued meminfo fields, or None when unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    fields: Dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "kB":
            try:
                fields[parts[0].rstrip(":")] = int(parts[1])
            except ValueError:
                return None
    return fields or None


def read_gpu_state() -> Optional[Dict[str, Any]]:
    """GPU inventory via nvidia-smi; None means unknown (fail closed)."""
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    try:
        done = subprocess.run(
            [
                binary,
                "--query-gpu=index,name,uuid,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        apps = subprocess.run(
            [binary, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or apps.returncode != 0:
        return None
    gpus: List[Dict[str, Any]] = []
    for line in done.stdout.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 5:
            return None
        try:
            gpus.append(
                {
                    "index": int(cells[0]),
                    "name": cells[1],
                    "uuid": cells[2],
                    "utilization_percent": float(cells[3]),
                    "memory_used_mib": None if cells[4] in ("[N/A]", "N/A") else float(cells[4]),
                }
            )
        except ValueError:
            return None
    compute_apps: List[Dict[str, str]] = []
    for line in apps.stdout.strip().splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 3:
            return None
        compute_apps.append(
            {"pid": cells[0], "process_name": cells[1], "used_memory": cells[2]}
        )
    return {"gpus": gpus, "compute_apps": compute_apps}


def read_mountinfo(path: str = "/proc/self/mountinfo") -> Optional[List[Dict[str, str]]]:
    """Parsed mount entries (mount point, fstype, source) or None."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    out: List[Dict[str, str]] = []
    for line in lines:
        left, separator, right = line.partition(" - ")
        fields, fs = left.split(), right.split()
        if not separator or len(fields) < 6 or len(fs) < 3:
            return None
        unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
        out.append({"mount_point": unescape(fields[4]), "fstype": fs[0], "source": unescape(fs[1])})
    return out


def statvfs(path: str) -> Any:
    return os.statvfs(path)


def _mount_for(
    target: str, mounts: List[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Deepest mount point at or above ``target`` (root "/" included)."""
    best: Optional[Dict[str, str]] = None
    target = os.path.abspath(target)
    for entry in mounts:
        mount_point = os.path.abspath(entry["mount_point"])
        prefix = mount_point.rstrip("/") + "/"
        if target == mount_point or target.startswith(prefix):
            if best is None or len(os.path.abspath(best["mount_point"])) < len(
                mount_point
            ):
                best = entry
    return best


def _fs_identity(
    target: str, mounts: List[Dict[str, str]], statvfs_fn: Statvfser
) -> Dict[str, Any]:
    entry = _mount_for(target, mounts)
    if entry is None:
        return {"fstype": None, "source": None, "known": False}
    identity = {
        "fstype": entry["fstype"],
        "source": entry["source"],
        "known": True,
    }
    try:
        probe = os.path.abspath(target)
        while True:
            try:
                st = statvfs_fn(probe)
                break
            except FileNotFoundError:
                parent = os.path.dirname(probe)
                if parent == probe:
                    raise
                probe = parent
        identity["free_bytes"] = st.f_bavail * st.f_frsize
    except OSError:
        identity["free_bytes"] = None
        identity["known"] = False
    return identity


def _present_bytes(root: Optional[str], *, allocated: bool = False) -> Optional[int]:
    """Total bytes of regular files under ``root`` (None: unreadable)."""
    if not root:
        return 0
    total = 0
    seen = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError):
            continue  # target tree simply not present yet: nothing to charge
        except OSError:
            return None
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    st = entry.stat(follow_symlinks=False)
                    identity = (st.st_dev, st.st_ino)
                    if identity not in seen:
                        total += st.st_blocks * 512 if allocated else st.st_size
                        seen.add(identity)
            except OSError:
                return None
    return total


# ---------------------------------------------------------------- engine

def _add_check(checks: List[Dict[str, Any]], name: str, ok: bool, detail: str, known: bool = True) -> None:
    checks.append(
        {
            "name": name,
            "status": ("pass" if ok else "fail") if known else "unknown",
            "detail": detail,
        }
    )


def run_preflight(
    *,
    phase: str,
    model_bytes: int,
    ple_bytes: int,
    model_root: Optional[str],
    ple_dir: Optional[str],
    build_root: Optional[str] = None,
    build_allowance_gib: float = BUILD_ALLOWANCE_GIB_DEFAULT,
    meminfo_reader: MeminfoReader = read_meminfo,
    gpu_prober: GpuProber = read_gpu_state,
    statvfs_fn: Statvfser = statvfs,
    mountinfo_reader: MountinfoReader = read_mountinfo,
    machine: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all checks and return a findings document.

    ``status`` is PASS only when every check passed. Any unknown counter
    fails the preflight (fail closed).
    """
    if phase not in (PHASE_BUILD, PHASE_SERVE):
        raise PreflightError("unknown phase %r" % (phase,))
    checks: List[Dict[str, Any]] = []

    # ---- CPU architecture
    arch = machine if machine is not None else platform.machine()
    _add_check(
        checks,
        "arch",
        arch == ARCH_EXPECTED,
        "machine=%r expected %r" % (arch, ARCH_EXPECTED),
        known=bool(arch),
    )

    # ---- RAM available
    meminfo = meminfo_reader()
    if meminfo is None or "MemAvailable" not in meminfo:
        _add_check(checks, "ram_available", False, "MemAvailable unknown — failing closed", known=False)
    else:
        available_gib = meminfo["MemAvailable"] * 1024 / GIB
        _add_check(
            checks,
            "ram_available",
            available_gib >= RAM_AVAILABLE_MIN_GIB,
            "MemAvailable=%.1f GiB required>=%.1f GiB"
            % (available_gib, RAM_AVAILABLE_MIN_GIB),
        )

    # ---- exactly one GB10, idle, no compute apps
    gpu_state = gpu_prober()
    if gpu_state is None:
        _add_check(checks, "gpu_inventory", False, "GPU inventory unknown — failing closed", known=False)
        _add_check(checks, "gpu_idle", False, "GPU state unknown — failing closed", known=False)
        _add_check(checks, "gpu_compute_apps", False, "compute apps unknown — failing closed", known=False)
    else:
        gpus = gpu_state["gpus"]
        _add_check(
            checks,
            "gpu_inventory",
            len(gpus) == 1 and gpus[0].get("index") == 0 and gpus[0].get("name") in ("GB10", "NVIDIA GB10"),
            "found %d GPU(s), need exactly 1: %s"
            % (len(gpus), [g.get("name") for g in gpus]),
        )
        if gpus:
            target = gpus[0]
            used = target.get("memory_used_mib")
            idle = (
                target["utilization_percent"] == 0.0
                and (used is None or (math.isfinite(used) and used <= GPU_IDLE_MEMORY_MIB_MAX))
            )
            _add_check(
                checks,
                "gpu_idle",
                idle,
                "utilization=%s%% memory_used_mib=%s (UMA RAM checked separately)"
                % (target["utilization_percent"], used),
            )
        else:
            _add_check(checks, "gpu_idle", False, "no GPU present")
        _add_check(
            checks,
            "gpu_compute_apps",
            len(gpu_state["compute_apps"]) == 0,
            "compute apps: %s"
            % ([a["process_name"] for a in gpu_state["compute_apps"]] or "none")
            + " (graphics/Xorg allowed)",
        )

    # ---- per-filesystem disk admission
    mounts = mountinfo_reader()
    if mounts is None:
        _add_check(
            checks, "filesystems", False, "mountinfo unreadable — failing closed", known=False
        )
        disk_rows: List[Dict[str, Any]] = []
    else:
        model_present = _present_bytes(model_root)
        ple_present = _present_bytes(ple_dir, allocated=True)
        if model_present is None or ple_present is None:
            _add_check(
                checks, "filesystems", False,
                "cannot measure existing trees — failing closed", known=False,
            )
            disk_rows = []
        else:
            missing_model = max(0, model_bytes - model_present)
            missing_ple = max(0, ple_bytes - ple_present)
            allowances: Dict[str, Dict[str, Any]] = {}

            def _charge(target: Optional[str], label: str, gib: float) -> None:
                if not target or gib <= 0:
                    return
                identity = _fs_identity(target, mounts, statvfs_fn)
                key = (identity.get("source"), identity.get("fstype"))
                row = allowances.setdefault(
                    key,
                    {
                        "target": target,
                        "fstype": identity.get("fstype"),
                        "source": identity.get("source"),
                        "known": identity["known"],
                        "free_bytes": identity.get("free_bytes"),
                        "required_bytes": 0.0,
                        "charges": [],
                    },
                )
                row["required_bytes"] += gib * GIB
                row["charges"].append({"label": label, "gib": round(gib, 3)})

            if phase == PHASE_BUILD:
                _charge(build_root, "model_missing", missing_model / GIB)
                _charge(build_root, "ple_missing", missing_ple / GIB)
                _charge(
                    build_root, "build_allowance",
                    build_allowance_gib if build_allowance_gib > 0 else 0.0,
                )
                _charge(build_root, "reserve", DISK_RESERVE_GIB)
            else:
                # serve phase: model is bind-mounted read-only; only the
                # file-backed PLE target and the reserve are charged, so a
                # warm restart is not rejected for already-present bytes.
                _charge(ple_dir, "ple_missing", missing_ple / GIB)
                _charge(ple_dir, "reserve", DISK_RESERVE_GIB)

            disk_rows = sorted(allowances.values(), key=lambda r: str(r["source"]))
            for row in disk_rows:
                ok = (
                    row["known"]
                    and row["free_bytes"] is not None
                    and row["free_bytes"] >= row["required_bytes"]
                )
                _add_check(
                    checks,
                    "disk[%s %s]" % (row["fstype"], row["source"]),
                    ok,
                    "free=%.1f GiB required=%.1f GiB (%s) known=%s"
                    % (
                        (row["free_bytes"] or 0) / GIB,
                        row["required_bytes"] / GIB,
                        ",".join(c["label"] for c in row["charges"]),
                        row["known"],
                    ),
                    known=row["known"],
                )
            for row in disk_rows:
                if row["fstype"] in REJECT_FILESYSTEMS or str(row["fstype"]).startswith("fuse."):
                    _add_check(
                        checks,
                        "local_nvme[%s]" % (row["source"],),
                        False,
                        "filesystem %r is not local NVMe (FUSE/NFS rejected)"
                        % (row["fstype"],),
                    )

    status = PREFLIGHT_PASS if all(c["status"] == "pass" for c in checks) else PREFLIGHT_FAIL
    return {
        "status": status,
        "phase": phase,
        "checks": checks,
        "disk": disk_rows if mounts is not None else [],
        "policy": {
            "delete_on_failure": False,
            "sysctl_or_driver_changes": False,
        },
    }


def main(argv: List[str] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prelaunch host admission (fail closed on unknowns)."
    )
    parser.add_argument("--phase", choices=(PHASE_BUILD, PHASE_SERVE), required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    parser.add_argument("--ple-bytes", type=int, required=True)
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--ple-dir", default=None)
    parser.add_argument("--build-root", default=None)
    parser.add_argument("--build-allowance-gib", type=float, default=BUILD_ALLOWANCE_GIB_DEFAULT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    findings = run_preflight(
        phase=args.phase,
        model_bytes=args.model_bytes,
        ple_bytes=args.ple_bytes,
        model_root=args.model_root,
        ple_dir=args.ple_dir,
        build_root=args.build_root,
        build_allowance_gib=args.build_allowance_gib,
    )
    if args.json:
        import json

        print(json.dumps(findings, indent=2, sort_keys=True))
    else:
        for check in findings["checks"]:
            print("%-8s %s: %s" % (check["status"], check["name"], check["detail"]))
        print("PREFLIGHT %s" % (findings["status"],))
    return 0 if findings["status"] == PREFLIGHT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
