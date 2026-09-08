"""Capture and re-check the exact owned runtime used by a benchmark."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import load_profile
from scripts import guard

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def http_json(base, path):
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=15) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("runtime observation exceeds byte limit")
    return json.loads(raw)


def observed_context(record, profile, lock, doc, server, model, base):
    """Pure validation seam; live reads happen in capture()."""
    guard.verify_ownership(doc, record)
    if doc.get("State", {}).get("Running") is not True:
        raise ValueError("owned runtime is not running")
    if (
        not isinstance(doc["State"].get("StartedAt"), str)
        or not doc["State"]["StartedAt"]
    ):
        raise ValueError("container startup identity missing")
    ports = doc.get("HostConfig", {}).get("PortBindings", {})
    if ports.get("30000/tcp") != [{"HostIp": "127.0.0.1", "HostPort": "30080"}]:
        raise ValueError("container is not bound to the owned loopback endpoint")
    labels = doc["Config"]["Labels"]
    if (
        labels.get("io.r0b0tlab.sglang.tree") != lock["sglang"]["tree"]
        or labels.get("io.r0b0tlab.model.sha") != record["model_sha"]
    ):
        raise ValueError("live image labels differ from runtime/model lock")
    if (
        record["model"] != MODEL_ID
        or model.get("served_model_name") != MODEL_ID
        or model.get("model_path") != "/model"
    ):
        raise ValueError("live model identity/path mismatch")
    if model.get("has_image_understanding") is not True or model.get(
        "architectures"
    ) != ["Qwen4ExpForConditionalGeneration"]:
        raise ValueError("the full vision architecture is not active")
    if (
        server.get("status") != "ready"
        or type(server.get("max_total_num_tokens")) is not int
    ):
        raise ValueError("scheduler capacity is not observed")
    if (
        server["max_total_num_tokens"] < profile.max_total_tokens
        or type(server.get("max_req_input_len")) is not int
        or server["max_req_input_len"] < profile.context_length - 1
    ):
        raise ValueError("effective runtime capacity is below the requested envelope")
    if profile.max_running_requests != 1:
        raise ValueError("this comparison contract is C1")
    startup = server.get("startup_time")
    command = server.get("launch_command")
    if startup is None or not command:
        raise ValueError("server startup identity is missing")
    return {
        "schema": "qwen38fn.runtime-context.v1",
        "model_id": MODEL_ID,
        "model_sha": record["model_sha"],
        "image_id": record["image"],
        "source_tree": lock["sglang"]["tree"],
        "context": profile.context_length,
        "total_pool": server["max_total_num_tokens"],
        "concurrency": 1,
        "profile": profile.raw,
        "endpoint": base.rstrip("/"),
        "epoch": {
            "cid": record["cid"],
            "nonce": record["nonce"],
            "profile_sha256": record["profile_sha256"],
            "container_started_at": doc["State"].get("StartedAt"),
            "server_startup_time": startup,
            "launch_command_sha256": canonical_hash(command),
            "model_info_sha256": canonical_hash(model),
        },
    }


def capture(record_path, runtime_lock, base, *, transport=None, reader=http_json):
    parsed = urllib.parse.urlsplit(base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.port != 30080
        or parsed.path not in ("", "/")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only the owned loopback port 30080 may be benchmarked")
    record, _ = guard.read_json(record_path)
    guard.require_hex(record.get("cid"), 64, "container ID")
    guard.require_hex(record.get("nonce"), 32, "owner nonce")
    guard.require_hex(record.get("model_sha"), 40, "model revision")
    lock, _ = guard.read_json(runtime_lock)
    path = Path(record_path).parent / "profile.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["profile_sha256"]:
        raise ValueError("launch profile snapshot changed")
    profile = load_profile(str(path))
    transport = transport or guard.DockerCliTransport()
    first = transport.inspect(record["cid"])
    guard.verify_ownership(first, record)
    if first.get("State", {}).get("Running") is not True:
        raise ValueError("owned container is not running")
    server = reader(base, "/server_info")
    model = reader(base, "/model_info")
    result = observed_context(record, profile, lock, first, server, model, base)
    last = transport.inspect(record["cid"])
    guard.verify_ownership(last, record)
    if last.get("State", {}).get("Running") is not True or last["State"].get(
        "StartedAt"
    ) != first["State"].get("StartedAt"):
        raise ValueError("runtime epoch changed during observation")
    return result


def verify_http_epoch(manifest, *, reader=http_json):
    context = manifest["runtime_context"]
    base = context["endpoint"]
    server, model = reader(base, "/server_info"), reader(base, "/model_info")
    expected = context["epoch"]
    if (
        server.get("startup_time") != expected["server_startup_time"]
        or canonical_hash(server.get("launch_command"))
        != expected["launch_command_sha256"]
        or canonical_hash(model) != expected["model_info_sha256"]
    ):
        raise ValueError("runtime HTTP epoch/model identity changed")
    if server.get("max_total_num_tokens") != context["total_pool"]:
        raise ValueError("runtime token pool changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("runtime context output must be fresh")
    result = capture(args.record, args.runtime_lock, args.base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "RUNTIME_CONTEXT_CAPTURED",
                "epoch": result["epoch"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
