"""Temporary byte-verified checkpoint and Docker-only fake for launcher tests."""

import copy
import hashlib
import json
from pathlib import Path

from scripts import guard, verify_files

IMAGE = "sha256:" + "ab" * 32
CID = "cd" * 32
PROFILE = {
    "schema": 1,
    "mode": "ar",
    "context_length": 32768,
    "max_total_tokens": 32768,
    "vision": {"backend": "triton_attn", "cuda_graph": True},
}


def make_env(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    payload = b"test-only checkpoint bytes\n"
    (model / "config.json").write_bytes(payload)
    identity = {"id": guard.MODEL_ID, "sha": "ef" * 20}
    rows = [
        {
            "path": "config.json",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"model": identity, "files": rows}))
    receipt = verify_files.verify_files(str(model), str(inventory_path))
    receipt_path = tmp_path / "receipt.json"
    verify_files.write_receipt(receipt, str(receipt_path))
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"model": {**identity, "files": rows}}))
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(PROFILE))
    cache = tmp_path / "cache"
    cache.mkdir()
    return {
        "image": IMAGE,
        "model-root": str(model),
        "profile": str(profile),
        "sources": str(sources),
        "receipt": str(receipt_path),
        "receipt-sha256": guard.digest(receipt_path.read_bytes()),
        "cache-dir": str(cache),
        "state-dir": str(tmp_path / "attempt"),
    }


def argv(env, *extra):
    return [v for key, value in env.items() for v in ("--" + key, value)] + list(extra)


class FakeDocker:
    """All outcomes are transport fixtures, never inference evidence."""

    def __init__(self, exits_after: int | None = 2, exit_code: int = 0):
        self.calls = []
        self.doc = {}
        self.exits_after = exits_after
        self.polls = 0
        self.exit_code = exit_code
        self.hooks = {}

    def hook(self, action):
        self.calls.append(action)
        if action in self.hooks:
            self.hooks[action](self)

    def image_inspect(self, image):
        self.hook("image_inspect")
        return {"Id": image, "Architecture": "arm64", "Os": "linux"}

    def create(self, args, cidfile):
        labels = {
            args[i + 1].split("=", 1)[0]: args[i + 1].split("=", 1)[1]
            for i, value in enumerate(args)
            if value == "--label"
        }
        self.doc = {
            "Id": CID,
            "Image": IMAGE,
            "Config": {"Labels": labels},
            "State": {"Running": False, "ExitCode": 0, "OOMKilled": False},
        }
        self.argv = args
        Path(cidfile).write_text(CID)
        self.hook("create")
        return CID

    def start(self, cid):
        self.doc["State"]["Running"] = True
        self.hook("start")

    def inspect(self, cid):
        self.hook("inspect")
        if not self.doc:
            raise guard.TransportError("test-only: no container")
        if self.doc["State"]["Running"]:
            self.polls += 1
            if self.exits_after is not None and self.polls >= self.exits_after:
                self.doc["State"].update(Running=False, ExitCode=self.exit_code)
        return copy.deepcopy(self.doc)

    def stop(self, cid):
        assert cid == CID
        self.hook("stop")
        self.doc["State"].update(Running=False, ExitCode=143)

    def logs(self, cid):
        self.hook("logs")
        return "TEST-ONLY TRANSPORT LOG, NOT INFERENCE EVIDENCE\n"


def options(docker, **extra):
    return {
        "transport": docker,
        "preflight_fn": lambda: {"status": "PASS"},
        "mem_reader": lambda: 16 * 1024 * 1024,
        "sleep": lambda _: None,
        **extra,
    }
