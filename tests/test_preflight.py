"""Preflight tests: entirely fixture-driven; no real GPU, disk or
/proc dependence. Unknown counters must fail closed."""

import pytest

from scripts import preflight as pf

MEMINFO_OK = {"MemAvailable": 110 * 1024 * 1024}  # kB -> 110 GiB
MEMINFO_LOW = {"MemAvailable": 90 * 1024 * 1024}

GPU_ONE_IDLE = {
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA GB10",
            "uuid": "GPU-0",
            "utilization_percent": 0.0,
            "memory_used_mib": 2.0,
        }
    ],
    "compute_apps": [],
}

GPU_WITH_XORG = {
    "gpus": [dict(GPU_ONE_IDLE["gpus"][0])],
    "compute_apps": [],  # graphics/Xorg simply do not appear as compute apps
}

GPU_BUSY = {
    "gpus": [
        {
            "index": 0,
            "name": "NVIDIA GB10",
            "uuid": "GPU-0",
            "utilization_percent": 87.0,
            "memory_used_mib": 40000.0,
        }
    ],
    "compute_apps": [{"pid": "99", "process_name": "python", "used_memory": "39000"}],
}

GPU_TWO = {
    "gpus": [GPU_ONE_IDLE["gpus"][0], dict(GPU_ONE_IDLE["gpus"][0], index=1)],
    "compute_apps": [],
}

MOUNTS = [
    {"mount_point": "/", "fstype": "ext4", "source": "/dev/nvme0n1p2"},
    {"mount_point": "/data", "fstype": "ext4", "source": "/dev/nvme1n1"},
    {"mount_point": "/mnt/net", "fstype": "nfs4", "source": "server:/export"},
    {"mount_point": "/mnt/fuse", "fstype": "fuseblk", "source": "persist:/x"},
]


def _runner(**kw):
    """Build a run_preflight partial with fixture overrides."""
    mounts = kw.pop("mounts", MOUNTS)
    meminfo = kw.pop("meminfo", MEMINFO_OK)
    gpu = kw.pop("gpu", GPU_ONE_IDLE)
    free = kw.pop("free", {})
    machine = kw.pop("machine", "aarch64")

    def statvfs(path):
        # resolve which source this path belongs to via mounts
        best = None
        for entry in mounts:
            mp = entry["mount_point"]
            if path.startswith(mp) and (best is None or len(mp) > len(best)):
                best = mp
        source = next(
            (e["source"] for e in mounts if e["mount_point"] == best), "?"
        )
        free_bytes = free.get(source, 0)

        class S:
            pass

        s = S()
        s.f_frsize = 4096
        s.f_bavail = free_bytes // 4096
        return s

    base = dict(
        phase="serve",
        model_bytes=0,
        ple_bytes=0,
        model_root=None,
        ple_dir=None,
        meminfo_reader=lambda: meminfo,
        gpu_prober=lambda: gpu,
        statvfs_fn=statvfs,
        mountinfo_reader=lambda: mounts,
        machine=machine,
    )
    base.update(kw)
    return lambda: pf.run_preflight(**base)


def _statuses(result):
    return {check["name"]: check["status"] for check in result["checks"]}


def test_warm_serve_passes():
    result = _runner(
        ple_bytes=52 << 30, ple_dir="/data/ple", free={"/dev/nvme1n1": 300 << 30}
    )()
    assert result["status"] == pf.PREFLIGHT_PASS, _statuses(result)


def test_serve_phase_does_not_charge_model_bytes():
    result = _runner(
        model_bytes=200 << 30,
        ple_bytes=52 << 30,
        ple_dir="/data/ple",
        free={"/dev/nvme1n1": 100 << 30},  # far less than model+ple+reserve
    )()
    assert result["status"] == pf.PREFLIGHT_PASS, _statuses(result)
    charges = {
        charge["label"]: charge["gib"] for charge in result["disk"][0]["charges"]
    }
    assert "model_missing" not in charges
    assert charges["reserve"] == 32.0


def test_serve_phase_credits_existing_ple_bytes():
    import os

    result = _runner(
        ple_bytes=52 << 30,
        ple_dir="/data/ple",
        model_root="/data/present",
        free={"/dev/nvme1n1": 100 << 30},
    )()
    assert result["status"] == pf.PREFLIGHT_PASS
    charges = {
        charge["label"]: charge["gib"] for charge in result["disk"][0]["charges"]
    }
    assert charges["ple_missing"] == 52.0  # nothing present under /data/ple


def test_ram_below_floor_fails():
    result = _runner(meminfo=MEMINFO_LOW)()
    assert result["status"] == pf.PREFLIGHT_FAIL
    assert _statuses(result)["ram_available"] == "fail"


def test_two_gpus_fail():
    result = _runner(gpu=GPU_TWO)()
    assert _statuses(result)["gpu_inventory"] == "fail"


def test_busy_gpu_fails():
    result = _runner(gpu=GPU_BUSY)()
    statuses = _statuses(result)
    assert statuses["gpu_idle"] == "fail"
    assert statuses["gpu_compute_apps"] == "fail"


def test_xorg_only_gpu_passes():
    result = _runner(gpu=GPU_WITH_XORG)()
    assert _statuses(result)["gpu_compute_apps"] == "pass"


def test_unknown_gpu_fails_closed():
    result = _runner(gpu=None)()
    statuses = _statuses(result)
    assert statuses["gpu_inventory"] == "unknown"
    assert statuses["gpu_idle"] == "unknown"
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_unknown_meminfo_fails_closed():
    result = _runner(meminfo=None)()
    assert _statuses(result)["ram_available"] == "unknown"
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_unknown_mountinfo_fails_closed():
    result = _runner(mounts=None)()
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_unknown_free_space_fails_closed():
    # statvfs resolves no source -> free unknown -> check unknown/fail
    result = _runner(ple_dir="/nowhere/ple", ple_bytes=1 << 30)()
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_nfs_ple_target_rejected():
    result = _runner(
        ple_dir="/mnt/net/ple", free={"server:/export": 1 << 40}
    )()
    statuses = _statuses(result)
    assert any(name.startswith("local_nvme[") for name in statuses)
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_fuse_ple_target_rejected():
    result = _runner(
        ple_dir="/mnt/fuse/ple", free={"persist:/x": 1 << 40}
    )()
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_small_disk_fails():
    result = _runner(
        ple_bytes=52 << 30, ple_dir="/data/ple", free={"/dev/nvme1n1": 40 << 30}
    )()
    assert result["status"] == pf.PREFLIGHT_FAIL


def test_build_phase_charges_missing_model_and_allowance():
    common = dict(
        phase="build",
        model_bytes=200 << 30,
        ple_bytes=52 << 30,
        model_root="/data/model",
        ple_dir="/data/ple",
        build_root="/data/build",
        build_allowance_gib=10.0,
        free={"/dev/nvme1n1": 400 << 30},
    )
    ok = _runner(**common)()
    assert ok["status"] == pf.PREFLIGHT_PASS, _statuses(ok)
    charges = {
        charge["label"]: charge["gib"] for charge in ok["disk"][0]["charges"]
    }
    assert set(charges) == {
        "model_missing", "ple_missing", "build_allowance", "reserve",
    }
    assert charges["model_missing"] == pytest.approx(200.0)
    assert charges["build_allowance"] == 10.0
    assert charges["reserve"] == 32.0

    tight = _runner(**{**common, "free": {"/dev/nvme1n1": 150 << 30}})()
    assert tight["status"] == pf.PREFLIGHT_FAIL


def test_shared_filesystem_not_double_charged():
    # model, ple and build roots on the same fs -> exactly one disk row
    result = _runner(
        phase="build",
        model_bytes=200 << 30,
        ple_bytes=52 << 30,
        model_root="/data/model",
        ple_dir="/data/ple",
        build_root="/data/build",
        free={"/dev/nvme1n1": 600 << 30},
    )()
    assert len(result["disk"]) == 1
    assert result["status"] == pf.PREFLIGHT_PASS


def test_non_aarch64_rejected():
    result = _runner(machine="x86_64")()
    assert _statuses(result)["arch"] == "fail"


def test_unknown_phase_rejected():
    with pytest.raises(pf.PreflightError):
        pf.run_preflight(
            phase="nope",
            model_bytes=0,
            ple_bytes=0,
            model_root=None,
            ple_dir=None,
        )


def test_policy_no_deletion():
    result = _runner()()
    assert result["policy"]["delete_on_failure"] is False
    assert result["policy"]["sysctl_or_driver_changes"] is False
