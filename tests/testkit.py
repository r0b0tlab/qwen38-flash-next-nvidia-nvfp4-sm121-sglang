"""Shared CPU-test kit for the launcher lifecycle tests.

The ONLY mocked boundary here is the Docker transport (an irreversible
external system). The launcher lifecycle itself — validation, gates,
locking, watch loop, stop ordering, exit-code accounting — runs for real
in-process. No docker binary, no GPU, no real model tree, no network.
"""

import copy
import hashlib
import json
import re
from types import SimpleNamespace

from scripts import guard, verify_files as vf

TEST_IMAGE = "sha256:" + "ab" * 32
MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MODEL_SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
BASE_PROFILE = {
    "schema": 1,
    "mode": "ar",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}

HEALTHY_KB = 16 * 1024 * 1024  # 16 GiB: above the 8 GiB floor


# ------------------------------------------------------------- fixtures

def make_env(tmp_path):
    """Verified model tree + receipt, profile, trusted sources, cache dir."""
    root = tmp_path / "model"
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"a":1}')
    data = (root / "config.json").read_bytes()
    model = {"id": MODEL_ID, "sha": MODEL_SHA}
    files = [
        {
            "path": "config.json",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    ]
    inventory = {"model": model, "files": files}
    inv = tmp_path / "model.files.json"
    inv.write_text(json.dumps(inventory))
    receipt = vf.verify_files(str(root), str(inv))
    receipt_path = tmp_path / "receipt.json"
    vf.write_receipt(receipt, str(receipt_path))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(BASE_PROFILE))
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps({"model": model, "files": files}))
    cache = tmp_path / "cache"
    cache.mkdir()
    return {
        "model_root": str(root),
        "receipt": str(receipt_path),
        "profile": str(profile),
        "sources": str(sources_path),
        "cache_dir": str(cache),
        "state_dir": str(tmp_path / "state"),
        "sources_dict": {"model": model, "files": files},
    }


def cli_args(env, extra=()):
    return [
        "--image", TEST_IMAGE,
        "--profile", env["profile"],
        "--model-root", env["model_root"],
        "--cache-dir", env["cache_dir"],
        "--sources", env["sources"],
        "--receipt", env["receipt"],
        "--state-dir", env["state_dir"],
        *extra,
    ]


def fake_probes():
    """Host probes that satisfy preflight on any CPU machine."""
    return {
        # 120 GiB available (>= 104 GiB required)
        "meminfo_reader": lambda: {"MemAvailable": 120 * 1024 * 1024},
        "gpu_prober": lambda: {
            "gpus": [
                {
                    "index": 0,
                    "name": "GB10",
                    "uuid": "GPU-fake",
                    "utilization_percent": 0.0,
                    "memory_used_mib": 2.0,
                }
            ],
            "compute_apps": [],
        },
        # 64 GiB free (>= 32 GiB serve reserve)
        "statvfs_fn": lambda path: SimpleNamespace(
            f_bavail=16 * 1024 * 1024, f_frsize=4096
        ),
        "mountinfo_reader": lambda: [
            {"mount_point": "/", "fstype": "ext4", "source": "/dev/nvme0n1p2"}
        ],
        "machine": "aarch64",
    }


# ------------------------------------------------- docker argv parsing

_SIZE_RE = re.compile(r"^(\d+)([kmg]?)$")


def _bytes(token):
    match = _SIZE_RE.match(token)
    mult = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}
    return int(match.group(1)) * mult[match.group(2)]


def _port_bindings(publish):
    host_ip, host_port, container_port = publish.split(":")
    return {
        "%s/tcp" % container_port: [
            {"HostIp": host_ip, "HostPort": host_port}
        ]
    }


def parse_create_argv(argv):
    """Parse a `docker create` argv into a structured summary."""
    assert argv[:2] == ["docker", "create"], argv[:2]
    out = {
        "labels": {},
        "mounts": {},  # dst -> (src, readonly)
        "security_opt": [],
        "cap_drop": [],
        "tmpfs": [],
        "read_only": False,
    }
    valued = {
        "--name", "--cidfile", "--gpus", "--cpus", "--memory",
        "--memory-swap", "--pids-limit", "--shm-size", "--cap-drop",
        "--security-opt", "--publish", "--workdir", "--mount", "--tmpfs",
    }
    repeated = {"--security-opt", "--cap-drop", "--mount", "--tmpfs"}
    rest = argv[2:]
    positionals = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--label":
            key, value = rest[i + 1].split("=", 1)
            out["labels"][key] = value
            i += 2
        elif tok == "--read-only":
            out["read_only"] = True
            i += 1
        elif tok in valued:
            value = rest[i + 1]
            if tok in repeated:
                out.setdefault(tok, []).append(value)
            else:
                out[tok] = value
            i += 2
        else:
            positionals.append(tok)
            i += 1
    out["image"] = positionals[0]
    out["container_args"] = positionals[1:]
    for mount in out.get("--mount", []):
        kv = {}
        for part in mount.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                kv[key] = value
            else:
                kv[part] = ""
        out["mounts"][kv["dst"]] = (kv["src"], "ro" in kv)
    return out


def build_container_doc(cid, parsed):
    """A plausible `docker inspect` document derived from the create argv."""
    mounts = [
        {
            "Type": "bind",
            "Source": src,
            "Destination": dst,
            "RW": not ro,
            "Propagation": "rprivate",
        }
        for dst, (src, ro) in sorted(parsed["mounts"].items())
    ]
    tmpfs = dict(t.split(":", 1) for t in parsed.get("--tmpfs", []))
    gpus = parsed.get("--gpus", "device=")
    return {
        "Id": cid,
        "Name": "/" + parsed["--name"],
        "Created": "2026-01-01T00:00:00Z",
        "Config": {
            "Image": parsed["image"],
            "Labels": dict(parsed["labels"]),
            "Cmd": list(parsed["container_args"]),
            "WorkingDir": parsed.get("--workdir"),
        },
        "State": {
            "Running": False,
            "ExitCode": 0,
            "OOMKilled": False,
            "Error": "",
            "StartedAt": "0001-01-01T00:00:00Z",
            "FinishedAt": "0001-01-01T00:00:00Z",
        },
        "HostConfig": {
            "Memory": _bytes(parsed["--memory"]),
            "MemorySwap": _bytes(parsed["--memory-swap"]),
            "NanoCpus": int(float(parsed["--cpus"]) * 1e9),
            "PidsLimit": int(parsed["--pids-limit"]),
            "ShmSize": _bytes(parsed["--shm-size"]),
            "Privileged": False,
            "ReadonlyRootfs": parsed["read_only"],
            "CapDrop": list(parsed.get("--cap-drop", [])),
            "SecurityOpt": list(parsed.get("--security-opt", [])),
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "PortBindings": _port_bindings(parsed["--publish"]),
            "Tmpfs": tmpfs,
            "DeviceRequests": [
                {"DeviceIDs": [gpus.split("=", 1)[1]], "Count": 0}
            ],
        },
        "Mounts": mounts,
        "_stop_exit_code": 0,
    }


# ------------------------------------------------------ fake transport

class FakeDocker:
    """In-memory stand-in for the docker CLI transport.

    Hooks (stage -> callable(fake, *args)) let tests inject failures and
    cancellations at exact lifecycle windows:
    - image_inspect / create / post_create / start / inspect / stop
    - a `create` hook raising models "client failed before mutation";
      a `post_create` hook raising models "container created but the
      client reported failure" (uncertain mutation).
    """

    def __init__(self, image_id=TEST_IMAGE, image_arch="arm64",
                 image_os="linux", hooks=None):
        self.image_doc = {
            "Id": image_id,
            "Architecture": image_arch,
            "Os": image_os,
        }
        self.hooks = dict(hooks or {})
        self.calls = []
        self.stops = []
        self.created = []
        self.docs = {}  # cid or name -> shared doc
        self.cid_counter = 0
        self.inspect_fail_refs = set()
        self.stop_should_fail = False
        self.image_inspect_fails = False
        self.last_parsed = None

    # -- helpers for tests ------------------------------------------

    def _unique_docs(self):
        seen = {}
        for doc in self.docs.values():
            seen[id(doc)] = doc
        return seen.values()

    def set_exit(self, code, oom=False):
        """Make every container appear exited (as docker inspect would)."""
        for doc in self._unique_docs():
            doc["State"]["Running"] = False
            doc["State"]["ExitCode"] = code
            doc["State"]["OOMKilled"] = oom

    def tamper_labels(self, mutate):
        for doc in self._unique_docs():
            mutate(doc["Config"]["Labels"])

    def calls_of(self, verb):
        return [c for c in self.calls if c[0] == verb]

    # -- transport API ----------------------------------------------

    def _hook(self, stage, *args):
        hook = self.hooks.get(stage)
        if hook is not None:
            hook(self, *args)

    def _doc(self, ref):
        doc = self.docs.get(ref)
        if doc is None:
            raise guard.TransportError("no such container: %s" % (ref,))
        return doc

    def image_inspect(self, ref):
        self.calls.append(("image_inspect", ref))
        self._hook("image_inspect", ref)
        if self.image_inspect_fails:
            raise guard.TransportError("image inspect exploded")
        if ref != self.image_doc["Id"]:
            raise guard.TransportError("no such image: %s" % (ref,))
        return dict(self.image_doc)

    def create(self, argv, cidfile_path):
        self.calls.append(("create", list(argv)))
        self._hook("create", argv)  # may raise BEFORE any mutation
        self.cid_counter += 1
        cid = "fakecid%012d" % self.cid_counter
        parsed = parse_create_argv(argv)
        self.last_parsed = parsed
        doc = build_container_doc(cid, parsed)
        self.docs[cid] = doc
        self.docs[parsed["--name"]] = doc
        with open(cidfile_path, "w", encoding="utf-8") as handle:
            handle.write(cid + "\n")
        self.created.append((list(argv), cid))
        self._hook("post_create", cid)  # may raise AFTER mutation
        return cid

    def start(self, ref):
        self.calls.append(("start", ref))
        self._hook("start", ref)
        doc = self._doc(ref)
        doc["State"]["Running"] = True

    def inspect(self, ref):
        self.calls.append(("inspect", ref))
        self._hook("inspect", ref)
        if ref in self.inspect_fail_refs:
            raise guard.TransportError("inspect broken for %s" % (ref,))
        return copy.deepcopy(self._doc(ref))

    def stop(self, ref, timeout_seconds=None):
        self.calls.append(("stop", ref, timeout_seconds))
        self._hook("stop", ref)
        if self.stop_should_fail:
            raise guard.TransportError("stop exploded")
        doc = self._doc(ref)
        doc["State"]["Running"] = False
        doc["State"]["ExitCode"] = doc["_stop_exit_code"]
        cid = doc["Id"] if doc["Id"] == ref else ref
        self.stops.append(cid)
        return cid
