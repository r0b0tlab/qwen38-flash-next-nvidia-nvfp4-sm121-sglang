"""Structural admission audit of the on-disk checkpoint.

Reads only safetensors headers, the weight index and the two metadata
JSON documents. Never reads tensor payload bytes beyond the header block
of each shard. Produces an admission report; ``HEADERS_ADMITTED`` is a
structural verdict only and is NOT full integrity verification (use
``scripts/verify_files.py`` for full-content hashing).

The audit derives tensor facts from the actual safetensors headers and
cross-checks them against the weight index entry by entry. Declared
constants in this module describe the pinned checkpoint contract; they
are used only to *validate* what is read from disk — facts reported by
the audit are independently derived from the parsed headers and index.
Every referenced tensor name must exist in exactly its named shard with
the contracted dtype/shape/data-offsets; scale tensors are validated
per-projection; PLE payload bytes are derived from header spans, not
from file names or counts.

Safety rejections: duplicate JSON keys, non-finite JSON numbers,
LFS-pointer shards, truncated or oversized headers, escaping/symlinked
paths, bool-as-int in tensor schemas, unknown dtypes, offsets outside
the file, and quantization alias mixing between the two metadata
entry points.

Status: NOT QUALIFIED.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from typing import Any, Dict, List, Optional, Set, Tuple

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
MAX_JSON_DEPTH = 64
METADATA_JSON_LIMIT = 64 * 1024 * 1024
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "BOOL": 1,
}

KNOWN_DTYPES = frozenset(
    {
        "F64",
        "F32",
        "F16",
        "BF16",
        "I64",
        "I32",
        "I16",
        "I8",
        "U64",
        "U32",
        "U16",
        "U8",
        "F8_E4M3",
        "F8_E5M2",
        "BOOL",
    }
)

# admission verdicts
HEADERS_ADMITTED = "HEADERS_ADMITTED"
HEADERS_REJECTED = "HEADERS_REJECTED"

_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"

# Tensor-name grammar of the quantized families (exact, anchored).
_MAIN_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\."
    r"(weight|weight_scale|weight_scale_2|input_scale)$"
)
_MTP_EXPERT_KEY_RE = re.compile(
    r"^mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_PLE_SHARD_KEY_RE = re.compile(
    r"^" + re.escape(_PLE_PREFIX) + r"\.shard_(\d+)\.weight$"
)


class AuditError(ValueError):
    """Checkpoint failed structural admission."""


# --------------------------------------------------------- the contract


def _build_trusted_contract() -> Dict[str, Any]:
    """The trusted contract for the pinned NVIDIA revision.

    Derived from real metadata captured at lock freeze (bounded header and
    metadata reads only; see tests/metadata_fixtures/). Numeric members
    mirror the EXPECTED_* constants above; tensor facts were verified
    entry-by-entry against the real shard headers.
    """
    gs = 128  # MTP FP8 block scale group size (per metadata entry points)
    mtp_projections = {
        "gate_proj": list(EXPECTED_MTP_GATE_UP_SHAPE),
        "up_proj": list(EXPECTED_MTP_GATE_UP_SHAPE),
        "down_proj": list(EXPECTED_MTP_DOWN_SHAPE),
    }
    mtp_scale_grids = {
        "gate_proj": list(EXPECTED_MTP_SCALE_INV_GRIDS[0]),
        "up_proj": list(EXPECTED_MTP_SCALE_INV_GRIDS[0]),
        "down_proj": list(EXPECTED_MTP_SCALE_INV_GRIDS[1]),
    }
    nvfp4_weight_shapes = {
        "gate_proj": [640, 1280],  # 640x2560 packed 2x/U8 (NVFP4 group 16)
        "up_proj": [640, 1280],
        "down_proj": [2560, 320],
    }
    nvfp4_scale_shapes = {
        "gate_proj": [640, 160],  # F8_E4M3 scale, 16-wide groups
        "up_proj": [640, 160],
        "down_proj": [2560, 40],
    }
    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_sha": EXPECTED_MODEL_SHA,
        "architecture": EXPECTED_ARCHITECTURE,
        "num_hidden_layers": EXPECTED_NUM_LAYERS,
        "num_experts": EXPECTED_NUM_EXPERTS,
        "num_experts_per_tok": EXPECTED_NUM_EXPERTS_USED,
        "native_context": 262144,
        "quant_algo": "MIXED_PRECISION",
        # Both metadata entry points describe the same MTP quantization
        # semantics under per-file literal aliases.
        "mtp_alias_by_entrypoint": {
            "config.json": "FP8_PB_WO",
            "hf_quant_config.json": "FP8_BLOCK_SCALES",
        },
        "mtp_prefix": "mtp.",
        "mtp": {
            "experts": EXPECTED_NUM_EXPERTS,
            "group_size": gs,
            "weight_dtype": "F8_E4M3",
            "scale_name": "weight_scale_inv",
            # BF16 in the current revision; FP32 opt-in if the contract
            # later moves scales to FP32 (validated per dtype).
            "scale_dtypes": ("BF16", "F32"),
            "projections": mtp_projections,
            "scale_grids": mtp_scale_grids,
        },
        "ple": {
            "shard_count": EXPECTED_PLE_SHARD_COUNT,
            "shard_name": _PLE_PREFIX + ".shard_{}.weight",
            "shard_dtype": "F8_E4M3",
            "shard_shape": [2500012, 160],
            "scale_name": _PLE_PREFIX + ".weight_scale",
            "scale_dtype": "BF16",
            "scale_shape": [1],
        },
        "nvfp4": {
            "experts": EXPECTED_NUM_EXPERTS,
            "group_size": 16,
            "weight_dtype": "U8",
            "scale_dtype": "F8_E4M3",
            "scale_2_dtype": "F32",
            "input_scale_dtype": "F32",
            "projections": ("gate_proj", "up_proj", "down_proj"),
            "weight_shapes": nvfp4_weight_shapes,
            "scale_shapes": nvfp4_scale_shapes,
        },
        # Non-expert payloads stay BF16 (indexer/embedding meta tensors
        # are I64); nothing outside the expert families may carry
        # packed/quantized dtypes.
        "nonexpert_dtypes": ("BF16", "I64"),
        "mtp_shard": MTP_PLE_SHARD_NAME,
        "index_key_count": EXPECTED_INDEX_KEY_COUNT,
        "mtp_key_count": EXPECTED_MTP_KEY_COUNT,
        "shard_count": EXPECTED_SHARD_COUNT,
        "ple_payload_bytes": EXPECTED_PLE_TOTAL_BYTES,
    }


TRUSTED_MODEL_CONTRACT = _build_trusted_contract()


# ------------------------------------------------------- strict JSON


def _reject_constant(_unused: str) -> float:
    raise AuditError("non-finite JSON number (NaN/Infinity) rejected")


def _no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AuditError("duplicate JSON key %r" % (key,))
        seen[key] = value
    return seen


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AuditError("nonfinite JSON numeric value")
    return parsed


def strict_json_loads(text: str, label: str) -> Any:
    """Parse JSON rejecting duplicate keys and NaN/Infinity literals."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except AuditError:
        raise
    except (ValueError, RecursionError) as exc:
        raise AuditError("%s: invalid JSON: %s" % (label, exc)) from exc


def _read_json(path: str, label: Optional[str] = None) -> Any:
    if os.path.islink(path):
        raise AuditError("%s is a symlink" % (label or path,))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read(METADATA_JSON_LIMIT + 1)
            if len(raw) > METADATA_JSON_LIMIT:
                raise AuditError("metadata JSON exceeds byte limit")
            return strict_json_loads(raw, label or os.path.basename(path))
    except OSError as exc:
        raise AuditError("cannot read %s: %s" % (path, exc)) from exc


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


# ------------------------------------------------- tensor entry schema


def _is_int(value: Any) -> bool:
    # bool is an int subclass in Python; tensor schemas must not use it
    return isinstance(value, int) and not isinstance(value, bool)


def _check_tensor_entry(shard: str, name: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise AuditError("shard %s: tensor %r entry must be an object" % (shard, name))
    dtype = entry.get("dtype")
    if not isinstance(dtype, str) or dtype not in KNOWN_DTYPES:
        raise AuditError(
            "shard %s: tensor %r has unknown dtype %r" % (shard, name, dtype)
        )
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(
        _is_int(dim) and dim >= 0 for dim in shape
    ):
        raise AuditError(
            "shard %s: tensor %r shape dimensions must be nonnegative ints"
            % (shard, name)
        )
    offsets = entry.get("data_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(_is_int(v) and v >= 0 for v in offsets)
        or offsets[0] > offsets[1]
    ):
        raise AuditError(
            "shard %s: tensor %r data_offsets must be [start<=end] ints" % (shard, name)
        )

    if offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]:
        raise AuditError(
            "shard %s: tensor %r shape/dtype/span disagree" % (shard, name)
        )


def validate_tensor_entries(shard: str, header: Dict[str, Any]) -> None:
    """Schema-validate every tensor entry of a parsed shard header."""
    for name, entry in header.items():
        if not isinstance(name, str):
            raise AuditError("shard %s: non-string tensor name" % (shard,))
        if name == "__metadata__":
            continue
        if len(name) > MAX_TENSOR_NAME_LENGTH:
            raise AuditError("shard %s: tensor name too long" % (shard,))
        if not _safe_relpath(name):
            raise AuditError("shard %s: unsafe tensor name %r" % (shard, name[:80]))
        _check_tensor_entry(shard, name, entry)


# ------------------------------------------------- index <-> headers


def validate_index_header_agreement(
    headers_by_file: Dict[str, Dict[str, Any]],
    weight_map: Dict[str, str],
) -> None:
    """Every index claim must resolve to its named shard header and every
    header tensor must be claimed — both directions, no sampling."""
    header_names: Dict[str, Set[str]] = {}
    for shard, header in headers_by_file.items():
        header_names[shard] = set(header) - {"__metadata__"}
    claims: Dict[str, Set[str]] = {shard: set() for shard in headers_by_file}
    for key, shard in weight_map.items():
        if not isinstance(shard, str) or not _safe_relpath(shard):
            raise AuditError("weight_map[%r]: unsafe shard path" % (key[:80],))
        if shard not in headers_by_file:
            raise AuditError(
                "weight_map claims %r from missing shard %s" % (key[:80], shard)
            )
        if key not in header_names[shard]:
            raise AuditError(
                "tensor %r claimed in index but absent from %s header"
                % (key[:80], shard)
            )
        claims[shard].add(key)
    for shard, names in header_names.items():
        if names != claims.get(shard, set()):
            extra = sorted(names - claims.get(shard, set()))
            raise AuditError(
                "shard %s carries tensors absent from the index: %s"
                % (shard, extra[:4])
            )


# ------------------------------------------------- MTP expert family


def validate_mtp_expert_family(header: Dict[str, Any], contract: Dict[str, Any]) -> int:
    """Validate all MTP routed-expert quant entries; return matched count.

    Every expert id in range(experts) must carry all three projections
    with contracted weight dtype/shape and a validated scale grid.
    Extra quant-family tensors are rejected.
    """
    mtp = contract["mtp"]
    matched: Set[str] = set()
    seen_experts: Dict[int, Set[str]] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        match = _MTP_EXPERT_KEY_RE.match(name)
        if not match:
            continue
        layer, expert, proj, kind = (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            match.group(4),
        )
        if name != "mtp.layers.%d.mlp.experts.%d.%s.%s" % (layer, expert, proj, kind):
            raise AuditError("noncanonical MTP tensor name")
        if layer != 0:
            raise AuditError("unexpected MTP layer index in %r" % (name,))
        if expert < 0 or expert >= mtp["experts"]:
            raise AuditError("MTP expert id out of range: %r" % (name,))
        if kind == "weight":
            if entry["dtype"] != mtp["weight_dtype"]:
                raise AuditError(
                    "MTP %r dtype %r != %r"
                    % (name, entry["dtype"], mtp["weight_dtype"])
                )
            if entry["shape"] != mtp["projections"][proj]:
                raise AuditError(
                    "MTP %r shape %r != %r"
                    % (name, entry["shape"], mtp["projections"][proj])
                )
        else:  # weight_scale_inv
            if entry["dtype"] not in mtp["scale_dtypes"]:
                raise AuditError(
                    "MTP %r scale dtype %r not in %r"
                    % (name, entry["dtype"], mtp["scale_dtypes"])
                )
            if entry["shape"] != mtp["scale_grids"][proj]:
                raise AuditError(
                    "MTP %r scale grid %r != %r"
                    % (name, entry["shape"], mtp["scale_grids"][proj])
                )
        matched.add(name)
        seen_experts.setdefault(expert, set()).add(proj)
    expected_names = mtp["experts"] * len(mtp["projections"]) * 2
    if len(matched) != expected_names:
        raise AuditError(
            "MTP expert family incomplete: %d quant entries, expected %d "
            "(missing or misnamed tensors)" % (len(matched), expected_names)
        )
    for expert in range(mtp["experts"]):
        if seen_experts.get(expert, set()) != set(mtp["projections"]):
            raise AuditError(
                "MTP expert %d does not carry all projections %r"
                % (expert, sorted(mtp["projections"]))
            )
    return len(matched)


# ------------------------------------------------------- PLE family


def validate_ple_family(header: Dict[str, Any], contract: Dict[str, Any]) -> int:
    """Validate the PLE ngram-embedding shard family; return derived payload
    bytes (sum of header spans for the shard tensors)."""
    ple = contract["ple"]
    scale_name = ple["scale_name"]
    shard_entries: Dict[int, Dict[str, Any]] = {}
    scale = header.get(scale_name)
    if scale is None:
        raise AuditError(
            "PLE scale tensor %r missing (index claims do not count; the "
            "header must carry it)" % (scale_name,)
        )
    if scale["dtype"] != ple["scale_dtype"]:
        raise AuditError(
            "PLE scale dtype %r != %r" % (scale["dtype"], ple["scale_dtype"])
        )
    if scale["shape"] != ple["scale_shape"]:
        raise AuditError(
            "PLE scale shape %r != %r" % (scale["shape"], ple["scale_shape"])
        )
    for name, entry in header.items():
        match = _PLE_SHARD_KEY_RE.match(name)
        if not match:
            continue
        index = int(match.group(1))
        if name != ple["shard_name"].format(index) or index in shard_entries:
            raise AuditError("noncanonical/duplicate PLE shard name")
        shard_entries[index] = entry
    if len(shard_entries) != ple["shard_count"]:
        raise AuditError(
            "PLE tensor-vs-file count mismatch: header carries %d shard "
            "tensors, contract requires %d" % (len(shard_entries), ple["shard_count"])
        )
    if sorted(shard_entries) != list(range(ple["shard_count"])):
        raise AuditError("PLE shard numbering must be contiguous 0..N-1")
    payload_bytes = 0
    per_shard = 1
    for dim in ple["shard_shape"]:
        per_shard *= dim
    for idx in range(ple["shard_count"]):
        entry = shard_entries[idx]
        if entry["dtype"] != ple["shard_dtype"]:
            raise AuditError(
                "PLE shard %d dtype %r != %r"
                % (idx, entry["dtype"], ple["shard_dtype"])
            )
        if entry["shape"] != ple["shard_shape"]:
            raise AuditError(
                "PLE shard %d shape %r != %r"
                % (idx, entry["shape"], ple["shard_shape"])
            )
        payload_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]
    if payload_bytes != per_shard * ple["shard_count"]:
        raise AuditError(
            "PLE derived payload bytes %d != %d (shape x count)"
            % (payload_bytes, per_shard * ple["shard_count"])
        )
    return payload_bytes


# ------------------------------------------------- metadata documents


def validate_config(config: Dict[str, Any], contract: Dict[str, Any]) -> None:
    architecture = config.get("architectures")
    if (
        not isinstance(architecture, list)
        or len(architecture) != 1
        or architecture[0] != contract["architecture"]
    ):
        raise AuditError(
            "unexpected architecture %r (expected %r)"
            % (architecture, contract["architecture"])
        )
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise AuditError("config.text_config must be an object")
    for field in (
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "max_position_embeddings",
    ):
        if not _is_int(text.get(field)):
            raise AuditError("text_config.%s must be an integer" % field)
    if text.get("num_hidden_layers") != contract["num_hidden_layers"]:
        raise AuditError(
            "text_config.num_hidden_layers %r != %r"
            % (text.get("num_hidden_layers"), contract["num_hidden_layers"])
        )
    if text.get("num_experts") != contract["num_experts"]:
        raise AuditError(
            "text_config.num_experts %r != %r"
            % (text.get("num_experts"), contract["num_experts"])
        )
    if text.get("num_experts_per_tok") != contract["num_experts_per_tok"]:
        raise AuditError(
            "text_config.num_experts_per_tok %r != %r"
            % (text.get("num_experts_per_tok"), contract["num_experts_per_tok"])
        )
    if text.get("max_position_embeddings") != contract["native_context"]:
        raise AuditError(
            "native context %r != %r"
            % (text.get("max_position_embeddings"), contract["native_context"])
        )
    vision = config.get("vision_config")
    if (
        not isinstance(vision, dict)
        or vision.get("dtype") != "bfloat16"
        or text.get("dtype") != "bfloat16"
    ):
        raise AuditError("config.vision_config with dtype is required")


def validate_quant_metadata(
    config_quant: Dict[str, Any],
    hf_quant: Dict[str, Any],
    contract: Dict[str, Any],
) -> None:
    """Both entry points must agree semantically, each under its own
    sanctioned literal alias for the MTP routed-expert entry."""
    if config_quant.get("quant_algo") != contract["quant_algo"]:
        raise AuditError(
            "config quantization_config.quant_algo %r != %r"
            % (config_quant.get("quant_algo"), contract["quant_algo"])
        )
    if hf_quant.get("quant_algo") != contract["quant_algo"]:
        raise AuditError(
            "hf_quant_config quantization.quant_algo %r != %r"
            % (hf_quant.get("quant_algo"), contract["quant_algo"])
        )
    config_layers = config_quant.get("quantized_layers")
    hf_layers = hf_quant.get("quantized_layers")
    if not isinstance(config_layers, dict) or not isinstance(hf_layers, dict):
        raise AuditError("quantized_layers must be objects in both entry points")
    mtp_key = "mtp.layers.0.mlp.experts"
    config_mtp = config_layers.get(mtp_key)
    hf_mtp = hf_layers.get(mtp_key)
    if not isinstance(config_mtp, dict) or not isinstance(hf_mtp, dict):
        raise AuditError("MTP quant entry %r missing" % (mtp_key,))
    aliases = contract["mtp_alias_by_entrypoint"]
    if config_mtp.get("quant_algo") != aliases["config.json"]:
        raise AuditError(
            "config.json MTP alias %r != %r"
            % (config_mtp.get("quant_algo"), aliases["config.json"])
        )
    if hf_mtp.get("quant_algo") != aliases["hf_quant_config.json"]:
        raise AuditError(
            "hf_quant_config.json MTP alias %r != %r"
            % (hf_mtp.get("quant_algo"), aliases["hf_quant_config.json"])
        )
    if config_mtp.get("group_size") != contract["mtp"]["group_size"]:
        raise AuditError(
            "MTP group_size %r != %r"
            % (config_mtp.get("group_size"), contract["mtp"]["group_size"])
        )
    if hf_mtp.get("group_size") != contract["mtp"]["group_size"]:
        raise AuditError(
            "hf MTP group_size %r != %r"
            % (hf_mtp.get("group_size"), contract["mtp"]["group_size"])
        )
    if hf_mtp != config_mtp:
        # the semantic payload must agree once the sanctioned alias
        # difference is normalized
        normalized_config = dict(config_mtp)
        normalized_config["quant_algo"] = aliases["hf_quant_config.json"]
        if normalized_config != hf_mtp:
            raise AuditError("MTP quant entries disagree beyond the sanctioned alias")
    ple_key = _PLE_PREFIX
    config_ple = config_layers.get(ple_key)
    hf_ple = hf_layers.get(ple_key)
    if not isinstance(config_ple, dict) or not isinstance(hf_ple, dict):
        raise AuditError("PLE quant entry %r missing" % (ple_key,))
    if config_ple != hf_ple or config_ple.get("quant_algo") != "FP8":
        raise AuditError("PLE quant entries disagree or are not FP8")
    targets = {
        "model.language_model.layers.%d.mlp.experts" % i
        for i in range(contract["num_hidden_layers"])
    }
    allowed = targets | {mtp_key, ple_key}
    for mapping in (config_layers, hf_layers):
        if set(mapping) != allowed:
            raise AuditError(
                "quantized_layers does not match all target/MTP/PLE families"
            )
        for key in targets:
            entry = mapping[key]
            if entry != {
                "quant_algo": "NVFP4",
                "group_size": contract["nvfp4"]["group_size"],
            }:
                raise AuditError("target NVFP4 quant metadata mismatch: %s" % key)


def derive_index_facts(
    index: Dict[str, Any], contract: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive counts from the weight index itself (never assumed)."""
    if not isinstance(index, dict) or "weight_map" not in index:
        raise AuditError("index missing weight_map")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict):
        raise AuditError("weight_map must be an object")
    referenced: Dict[str, int] = {}
    mtp_prefix = contract["mtp_prefix"]
    mtp_count = 0
    ple_count = 0
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not key:
            raise AuditError("weight_map keys must be non-empty strings")
        if not isinstance(shard, str) or not _safe_relpath(shard):
            raise AuditError("unsafe/non-string index shard")
        referenced[shard] = referenced.get(shard, 0) + 1
        if key.startswith(mtp_prefix):
            mtp_count += 1
        if _PLE_SHARD_KEY_RE.match(key):
            ple_count += 1
    facts = {
        "key_count": len(weight_map),
        "mtp_key_count": mtp_count,
        "ple_shard_count": ple_count,
        "file_count": len(referenced),
        "referenced_files": referenced,
        "weight_map": weight_map,
    }
    if facts["key_count"] != contract["index_key_count"]:
        raise AuditError(
            "derived index key count %d != contract %d"
            % (facts["key_count"], contract["index_key_count"])
        )
    if facts["mtp_key_count"] != contract["mtp_key_count"]:
        raise AuditError(
            "derived MTP key count %d != contract %d"
            % (facts["mtp_key_count"], contract["mtp_key_count"])
        )
    if facts["ple_shard_count"] != contract["ple"]["shard_count"]:
        raise AuditError(
            "derived PLE shard count %d != contract %d"
            % (facts["ple_shard_count"], contract["ple"]["shard_count"])
        )
    if facts["file_count"] != contract["shard_count"]:
        raise AuditError(
            "derived shard file count %d != contract %d"
            % (facts["file_count"], contract["shard_count"])
        )
    return facts


# ------------------------------------------------- shard header audit


def audit_safetensors_headers(
    root: str, shard_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Read each referenced shard header (bounded to 64 MiB) and apply
    structural checks. Returns (report, headers_by_file)."""
    shards: List[Dict[str, Any]] = []
    headers: Dict[str, Dict[str, Any]] = {}
    present: Set[str] = set()
    if shard_names is None:
        shard_names = sorted(
            entry for entry in os.listdir(root) if entry.endswith(".safetensors")
        )
    for name in shard_names:
        if not _safe_relpath(name):
            raise AuditError("unsafe shard name %r" % (name[:80],))
        path = os.path.join(root, name)
        if os.path.islink(path):
            raise AuditError("shard %s is a symlink" % (name,))
        realpath = os.path.realpath(path)
        root_real = os.path.realpath(root) + os.sep
        if not realpath.startswith(root_real):
            raise AuditError("shard %s escapes the checkpoint root" % (name,))
        if not os.path.isfile(path):
            raise AuditError("referenced shard missing: %s" % (name,))
        st = os.lstat(path)
        present.add(name)
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
            header_bytes = handle.read(header_len)
            if len(header_bytes) < header_len:
                raise AuditError("shard %s: truncated header" % (name,))
        size = os.path.getsize(path)
        if size < 8 + header_len:
            raise AuditError("shard %s: file smaller than header" % (name,))
        parsed = strict_json_loads(
            header_bytes.decode("utf-8"), "shard %s header" % (name,)
        )
        if not isinstance(parsed, dict):
            raise AuditError("shard %s: header must be a JSON object" % (name,))
        validate_tensor_entries(name, parsed)
        data_size = size - 8 - header_len
        spans = 0
        max_end = 0
        intervals = []
        for tensor_name, entry in parsed.items():
            if tensor_name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            if end > data_size:
                raise AuditError(
                    "shard %s: tensor %r data end %d outside file data "
                    "region (%d bytes)" % (name, tensor_name[:80], end, data_size)
                )
            spans += end - start
            max_end = max(max_end, end)
            intervals.append((start, end))
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise AuditError("shard %s: overlapping spans or payload gap" % name)
            cursor = end
        if spans != data_size or max_end != data_size:
            raise AuditError(
                "shard %s: tensor spans (%d bytes, end %d) do not tile the "
                "data region (%d bytes)" % (name, spans, max_end, data_size)
            )
        shards.append(
            {
                "name": name,
                "size": size,
                "header_len": header_len,
                "tensor_count": len(parsed) - (1 if "__metadata__" in parsed else 0),
                "inode": st.st_ino,
                "mtime_ns": st.st_mtime_ns,
                "ctime_ns": st.st_ctime_ns,
            }
        )
        headers[name] = parsed
    return {
        "shards": shards,
        "shard_count": len(shards),
        "total_bytes": sum(s["size"] for s in shards),
        "present_files": present,
    }, headers


def validate_nvfp4_families(headers_by_file, contract):
    """Validate the full layer × expert × projection × tensor-kind set."""
    q = contract["nvfp4"]
    seen = set()
    for header in headers_by_file.values():
        for name, entry in header.items():
            match = _MAIN_EXPERT_KEY_RE.fullmatch(name)
            if not match:
                continue
            layer, expert = int(match[1]), int(match[2])
            projection, kind = match[3], match[4]
            identity = (layer, expert, projection, kind)
            canonical = "model.language_model.layers.%d.mlp.experts.%d.%s.%s" % identity
            if (
                name != canonical
                or identity in seen
                or not 0 <= layer < contract["num_hidden_layers"]
                or not 0 <= expert < q["experts"]
            ):
                raise AuditError("unexpected/duplicate NVFP4 expert entry: %s" % name)
            if kind == "weight":
                dtype, shape = q["weight_dtype"], q["weight_shapes"][projection]
            elif kind == "weight_scale":
                dtype, shape = q["scale_dtype"], q["scale_shapes"][projection]
            else:
                dtype = (
                    q["scale_2_dtype"]
                    if kind == "weight_scale_2"
                    else q["input_scale_dtype"]
                )
                shape = []
            if entry["dtype"] != dtype or entry["shape"] != shape:
                raise AuditError("NVFP4 dtype/shape mismatch: %s" % name)
            seen.add(identity)
    expected_count = (
        contract["num_hidden_layers"] * q["experts"] * len(q["projections"]) * 4
    )
    if len(seen) != expected_count:
        raise AuditError("NVFP4 expert tensor set is incomplete")
    return {
        "layer_count": len({key[0] for key in seen}),
        "expert_entries": len(seen) // 4,
        "tensor_count": len(seen),
        "group_size": q["group_size"],
    }


def _validate_dtype_families(
    headers_by_file: Dict[str, Dict[str, Any]], contract: Dict[str, Any]
) -> None:
    """Non-expert tensors stay BF16 (indexer/embedding meta: I64)."""
    allowed = contract["nonexpert_dtypes"]
    for shard, header in headers_by_file.items():
        for name, entry in header.items():
            if name == "__metadata__" or name == contract["ple"]["scale_name"]:
                continue
            if (
                _MAIN_EXPERT_KEY_RE.match(name)
                or _MTP_EXPERT_KEY_RE.match(name)
                or _PLE_SHARD_KEY_RE.match(name)
            ):
                continue
            if entry["dtype"] not in allowed:
                raise AuditError(
                    "shard %s: non-expert tensor %r carries quantized dtype %r "
                    "(excluded modules must remain unquantized)"
                    % (shard, name[:80], entry["dtype"])
                )


def audit_checkpoint(
    root: str, contract: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Structural admission audit of the checkpoint at ``root``.

    Uses the trusted pinned-model contract by default (production CLI);
    tests may pass an explicit expected contract. Validates both metadata
    entry points, derives index/tensor facts from the actual documents and
    headers, and cross-checks every name. Never modifies any file.
    """
    if contract is None:
        contract = TRUSTED_MODEL_CONTRACT
    root = os.path.abspath(root)
    config_path = os.path.join(root, "config.json")
    quant_path = os.path.join(root, "hf_quant_config.json")
    index_path = os.path.join(root, "model.safetensors.index.json")

    for required in (config_path, quant_path, index_path):
        if not os.path.isfile(required):
            raise AuditError("missing required metadata file: %s" % (required,))

    config = _read_json(config_path, "config.json")
    if not isinstance(config, dict):
        raise AuditError("config.json must be an object")
    validate_config(config, contract)
    quant_doc = _read_json(quant_path, "hf_quant_config.json")
    if not isinstance(quant_doc, dict):
        raise AuditError("hf_quant_config.json must be an object")
    hf_quant = quant_doc.get("quantization")
    if not isinstance(hf_quant, dict):
        raise AuditError("hf_quant_config.json missing quantization object")
    validate_quant_metadata(config.get("quantization_config", {}), hf_quant, contract)

    index = _read_json(index_path, "model.safetensors.index.json")
    facts = derive_index_facts(index, contract)
    weight_map = facts["weight_map"]

    report, headers = audit_safetensors_headers(root)
    present = report["present_files"]
    missing = sorted(set(facts["referenced_files"]) - present)
    if missing:
        raise AuditError("weight_map references missing shards: %s" % (missing[:4],))
    extra = sorted(present - set(facts["referenced_files"]))
    if extra:
        raise AuditError("unreferenced .safetensors files present: %s" % (extra[:4],))
    if report["shard_count"] != contract["shard_count"]:
        raise AuditError(
            "shard file count %d != contract %d (count is derived, not assumed)"
            % (report["shard_count"], contract["shard_count"])
        )
    for shard in report["shards"]:
        if os.path.islink(os.path.join(root, shard["name"])):
            raise AuditError("shard %s is a symlink" % (shard["name"],))

    mtp_count = 0
    ple_bytes = 0
    for shard_name, header in headers.items():
        if shard_name == contract["mtp_shard"]:
            mtp_count = validate_mtp_expert_family(header, contract)
            ple_bytes = validate_ple_family(header, contract)
            if ple_bytes != contract["ple_payload_bytes"]:
                raise AuditError(
                    "PLE payload bytes %d != contract %d"
                    % (ple_bytes, contract["ple_payload_bytes"])
                )
    if not mtp_count:
        raise AuditError(
            "MTP shard %s missing from checkpoint" % (contract["mtp_shard"],)
        )
    validate_index_header_agreement(headers, weight_map)
    _validate_dtype_families(headers, contract)
    nvfp4_facts = validate_nvfp4_families(headers, contract)

    root_stat = os.stat(root)
    findings: Dict[str, Any] = {
        "model": {
            "id": contract["model_id"],
            "sha": contract["model_sha"],
        },
        "root": root,
        "root_stat": {
            "inode": root_stat.st_ino,
            "dev": root_stat.st_dev,
            "mtime_ns": root_stat.st_mtime_ns,
            "ctime_ns": root_stat.st_ctime_ns,
        },
        "architecture": config["architectures"][0],
        "metadata": {
            "config_json": {
                "mtp_alias": contract["mtp_alias_by_entrypoint"]["config.json"],
            },
            "hf_quant_config": {
                "mtp_alias": contract["mtp_alias_by_entrypoint"][
                    "hf_quant_config.json"
                ],
            },
        },
        "index_key_count": facts["key_count"],
        "mtp_key_count": facts["mtp_key_count"],
        "checkpoint_files": {
            "shards": report["shards"],
            "shard_count": report["shard_count"],
            "total_bytes": report["total_bytes"],
        },
        "mtp": {
            "expert_count": contract["mtp"]["experts"],
            "projection_count": mtp_count // 2 if headers else 0,
        },
        "ple": {
            "shard_count": contract["ple"]["shard_count"],
            "payload_bytes": ple_bytes,
        },
        "nvfp4": nvfp4_facts,
    }
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
            "verdict=%s shards=%d index_keys=%d mtp_keys=%d "
            "mtp_experts=%d ple_shards=%d ple_bytes=%d"
            % (
                findings["verdict"],
                findings["checkpoint_files"]["shard_count"],
                findings["index_key_count"],
                findings["mtp_key_count"],
                findings["mtp"]["expert_count"],
                findings["ple"]["shard_count"],
                findings["ple"]["payload_bytes"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
