"""Real probe formats; no GPU workloads, downloads, or host mutation."""
from types import SimpleNamespace
import subprocess
import shutil

from scripts import preflight as pf
from tests.test_preflight import _runner, GPU_ONE_IDLE


def test_meminfo_kernel_format(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal: 125829120 kB\nMemAvailable: 115343360 kB\nHugePages_Total: 0\n")
    assert pf.read_meminfo(str(p)) == {"MemTotal": 125829120, "MemAvailable": 115343360}


def test_mountinfo_separator_and_escaped_path(tmp_path):
    p = tmp_path / "mountinfo"
    p.write_text("31 1 259:2 / / rw - ext4 /dev/nvme0n1p2 rw\n"
                 "33 31 259:3 / /data\\040disk rw shared:3 master:1 - xfs /dev/nvme1n1 rw\n")
    assert pf.read_mountinfo(str(p)) == [
        {"mount_point": "/", "fstype": "ext4", "source": "/dev/nvme0n1p2"},
        {"mount_point": "/data disk", "fstype": "xfs", "source": "/dev/nvme1n1"},
    ]


def test_smi_uma_na_is_explicit_not_zero(monkeypatch):
    seen = []
    def run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="0, NVIDIA GB10, GPU-fixture, 0, [N/A]\n" if len(seen) == 1 else "")
    monkeypatch.setattr(shutil, "which", lambda _: "/test-only/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", run)
    got = pf.read_gpu_state()
    assert got is not None
    assert got["gpus"][0]["memory_used_mib"] is None
    assert "--format=csv,noheader,nounits" in seen[1]
    assert _runner(gpu=got)()["status"] == "PASS"


def test_wrong_gpu_model_rejected():
    gpu = {"gpus": [{**GPU_ONE_IDLE["gpus"][0], "name": "NVIDIA RTX 5090"}], "compute_apps": []}
    assert _runner(gpu=gpu)()["status"] == "FAIL"


def test_sparse_ple_credit_is_allocated_not_logical(tmp_path):
    p = tmp_path / "ple.bin"
    with p.open("wb") as f:
        f.truncate(32 << 20)
    assert pf._present_bytes(str(tmp_path), allocated=True) == p.stat().st_blocks * 512
    assert pf._present_bytes(str(tmp_path), allocated=True) < p.stat().st_size


def test_uncreated_nested_cache_uses_existing_filesystem(tmp_path):
    entry = {"mount_point": "/", "fstype": "ext4", "source": "/dev/nvme0n1p2"}
    got = pf._fs_identity(str(tmp_path / "not-yet" / "ple" / "revision"), [entry], pf.statvfs)
    assert got["known"] is True
    assert got["free_bytes"] > 0


def test_named_fuse_subtype_rejected():
    assert _runner(ple_dir="/mnt/ple", mounts=[{"mount_point": "/", "fstype": "fuse.sshfs", "source": "test-only"}], free={"test-only": 1 << 40})()["status"] == "FAIL"
