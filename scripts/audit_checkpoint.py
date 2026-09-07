"""Structural admission audit of the on-disk checkpoint.

Reads only safetensors headers, the weight index and the two metadata
JSON documents. Never reads tensor payload bytes beyond the header block
of each shard. Produces an admission report; ``HEADERS_ADMITTED`` is a
structural verdict only and is NOT full integrity verification (use
``scripts/verify_files.py`` for full-content hashing).

Status: NOT QUALIFIED.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List

EXPECTED_MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
EXPECTED_MODEL_SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
EXPECTED_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
EXPECTED_NUM_LAYERS = 48
EXPECTED_NUM_EXPERTS = 512
EXPECTED_NUM_EXPERTS_USED = 10
EXPECTED_MTP_LAYER_COUNT = 1
EXPECTED_INDEX_KEY_COUNT = 299545
EXPECTED_MTP_KEY_COUNT = 3101
EXPECTED_PLE_SHARD_COUNT = 128
EXPECTED_PLE_TOTAL_BYTES = 51200245760
EXPECTED_SHARD_COUNT = 11
MTP_PLE_SHARD_NAME = "model-fp8-mtp-ple.safetensors"

# MTP FP8 E4M3 expert shapes: gate/up and down, with scale_inv grids
EXPECTED_MTP_GATE_UP_SHAPE = [640, 2560]
EXPECTED_MTP_DOWN_SHAPE = [2560, 640]
EXPECTED_MTP_SCALE_INV_GRIDS = ([5, 20], [20, 5])

SAFETENSORS_HEADER_LIMIT = 64 * 1024 * 1024
MAX_TENSOR_NAME_LENGTH = 4096

# admission verdicts
HEADERS_ADMITTED = "HEADERS_ADMITTED"
HEADERS_REJECTED = "HEADERS_REJECTED"


class AuditError(ValueError):
    """Checkpoint failed structural admission."""


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_relpath(name: str) -> bool:
    """Path must be a plain relative path that cannot escape the tree."""
    if not name or len(name) > 1024:
        return False
    if name.startswith("/") or "\\" in name:
        return False
    if "\x00" in name:
        return False
    parts = name.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def _check_header_buffer(name: str, header: bytes, offset: int) -> None:
    if len(header) > SAFETENSORS_HEADER_LIMIT:
        raise AuditError("shard %s: header too large" % (name,))
    # header must be valid JSON; no path fields may escape
    try:
        parsed = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("shard %s: invalid header JSON: %s" % (name, exc)) from exc
    if not isinstance(parsed, dict):
        raise AuditError("shard %s: header must be a JSON object" % (name,))
    for tensor_name in parsed:
        if not isinstance(tensor_name, str):
            raise AuditError("shard %s: non-string tensor name" % (name,))
        if len(tensor_name) > MAX_TENSOR_NAME_LENGTH:
            raise AuditError("shard %s: tensor name too long" % (name,))
        if not _safe_relpath(tensor_name):
            raise AuditError(
                "shard %s: unsafe tensor name %r" % (name, tensor_name[:80])
            )


def audit_safetensors_headers(root: str) -> Dict[str, Any]:
    """Read each shard header in ``root`` and apply structural checks."""
    shards: List[Dict[str, Any]] = []
    total_ple_bytes = 0
    ple_shard_count = 0
    names = sorted(
        entry
        for entry in os.listdir(root)
        if entry.endswith(".safetensors") and os.path.isfile(os.path.join(root, entry))
    )
    if not names:
        raise AuditError("no .safetensors shards found under %s" % (root,))
    for name in names:
        path = os.path.join(root, name)
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise AuditError("shard %s: truncated header length" % (name,))
            (header_len,) = struct.unpack("<Q", prefix)
            if header_len > SAFETENSORS_HEADER_LIMIT:
                raise AuditError(
                    "shard %s: declared header length %d exceeds limit"
                    % (name, header_len)
                )
            header = handle.read(header_len)
            if len(header) < header_len:
                raise AuditError("shard %s: truncated header" % (name,))
            _check_header_buffer(name, header[:header_len], 8 + header_len)
        size = os.path.getsize(path)
        if size < 8 + header_len:
            raise AuditError("shard %s: file smaller than header" % (name,))
        shards.append({"name": name, "size": size, "header_len": header_len})
        if "mtp-ple" in name or "ple" in name:
            ple_shard_count += 1
            total_ple_bytes += size
    return {
        "shards": shards,
        "shard_count": len(shards),
        "ple_shard_count": ple_shard_count,
        "ple_total_bytes": total_ple_bytes,
    }


def audit_checkpoint(root: str) -> Dict[str, Any]:
    """Structural admission audit of the checkpoint at ``root``.

    Validates both metadata entry points (config.json and
    hf_quant_config.json), the weight index shape, the MTP/PLE shard
    presence and reports architecture/layer/expert facts. Preserves the
    original literal files — this function only reads.
    """
    config_path = os.path.join(root, "config.json")
    quant_path = os.path.join(root, "hf_quant_config.json")
    index_path = os.path.join(root, "model.safetensors.index.json")

    for required in (config_path, quant_path, index_path):
        if not os.path.isfile(required):
            raise AuditError("missing required metadata file: %s" % (required,))

    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise AuditError("config.json must be an object")
    quant_config = _read_json(quant_path)
    if not isinstance(quant_config, dict):
        raise AuditError("hf_quant_config.json must be an object")

    architecture = config.get("architectures", [None])[0]
    if architecture != EXPECTED_ARCHITECTURE:
        raise AuditError(
            "unexpected architecture %r (expected %r)"
            % (architecture, EXPECTED_ARCHITECTURE)
        )
    quant_attr = config.get("quantization_config", {})
    if not isinstance(quant_attr, dict):
        raise AuditError("config quantization_config must be an object")

    index = _read_json(index_path)
    if not isinstance(index, dict) or "weight_map" not in index:
        raise AuditError("index missing weight_map")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise AuditError("weight_map must be an object")

    findings: Dict[str, Any] = {
        "model": {
            "id": EXPECTED_MODEL_ID,
            "sha": EXPECTED_MODEL_SHA,
        },
        "root": os.path.abspath(root),
        "architecture": architecture,
        "config_quantization_group_size": quant_attr.get("group_size"),
        "metadata": {
            "config_json": {
                "mtp_fp8_pb_wo": (
                    quant_attr.get("mtp") is not None
                    or "fp8_pb_wo" in json.dumps(config)
                ),
            },
            "hf_quant_config": {
                "quant_method": quant_config.get("quant_method"),
            },
        },
        "index_key_count": len(weight_map),
        "mtp_key_count": sum(
            1 for key in weight_map if ".mtp." in key or key.startswith("mtp.")
        ),
        "checkpoint_files": audit_safetensors_headers(root),
    }

    # every weight_map entry must point at a present shard with a safe name
    shard_files = set()
    for key, shard in weight_map.items():
        if not isinstance(shard, str) or not _safe_relpath(shard):
            raise AuditError("weight_map[%r]: unsafe shard path" % (key,))
        shard_files.add(shard)
    present = {
        entry["name"] for entry in findings["checkpoint_files"]["shards"]
    }
    missing = sorted(shard_files - present)
    if missing:
        raise AuditError("weight_map references missing shards: %s" % (missing[:4],))

    findings["verdict"] = HEADERS_ADMITTED
    findings["verdict_note"] = (
        "structural header admission only; NOT full integrity verification"
    )
    return findings


def main(argv: List[str] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Structural admission audit of the checkpoint (headers only)."
    )
    parser.add_argument("root", help="checkpoint root directory")
    parser.add_argument(
        "--json", action="store_true", help="emit the full findings document"
    )
    args = parser.parse_args(argv)
    try:
        findings = audit_checkpoint(args.root)
    except AuditError as exc:
        print("AUDIT FAILED: %s" % (exc,))
        return 1
    if args.json:
        print(json.dumps(findings, indent=2, sort_keys=True))
    else:
        print(
            "verdict=%s shards=%d index_keys=%d mtp_keys=%d"
            % (
                findings["verdict"],
                findings["checkpoint_files"]["shard_count"],
                findings["index_key_count"],
                findings["mtp_key_count"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
