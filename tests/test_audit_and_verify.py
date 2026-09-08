"""Audit + verify tests: genuine tensor/schema checks against the pinned
checkpoint contract.

Two layers:

1. Real-metadata tests — byte-real fixtures captured from
   nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54 (see tests/metadata_fixtures/).
   The real config, both quant entry points, the full 299,545-key weight
   index and the real mtp-ple / shard-10 safetensors headers are validated
   entry-by-entry against the trusted contract. No model payload is stored
   or hashed here.

2. Physical checkpoint tests — a tiny synthetic checkpoint that is
   structurally isomorphic to the real one (same name grammar, same tensor
   families, same alias split) is built in tmp_path and audited END-TO-END
   through ``audit_checkpoint`` with an explicit scaled contract, and every
   rejection path is exercised by mutating it. The production CLI default
   contract is proven strict: the tiny checkpoint must be REJECTED when the
   trusted (real) contract is used.

Full-file verification tests cover the actual locks/sources.json schema
(id/sha/files[{path,size,sha256,git_blob}]), streamed SHA-256 for LFS
entries, Git blob SHA-1 for ordinary metadata, symlink escape, wrong
remote hash, receipt forgery and stat drift. No test touches the real
model tree beyond reading the captured fixture directory, and no test
creates or hashes huge files.
"""

import hashlib
import gzip
import json
import os
import struct

import pytest

from scripts import audit_checkpoint as ac
from scripts import verify_files as vf

FIXTURES = os.path.join(os.path.dirname(__file__), "metadata_fixtures")

ITEMSIZE = {
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

REAL_MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REAL_MODEL_SHA = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
PLE_SHARD_NAME = PLE_PREFIX + ".shard_{}.weight"
PLE_SCALE_NAME = PLE_PREFIX + ".weight_scale"


# ------------------------------------------------------------- shard I/O


def _layout(tensors):
    """Assign sequential data_offsets; return (entries, data_bytes)."""
    entries = {}
    pos = 0
    for name, spec in tensors.items():
        nbytes = ITEMSIZE[spec["dtype"]]
        for dim in spec["shape"]:
            nbytes *= dim
        entries[name] = {
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "data_offsets": [pos, pos + nbytes],
        }
        pos += nbytes
    return entries, pos


def _write_shard(root, name, tensors, payload=None, header_bytes=None):
    """Write a safetensors shard. By default offsets are laid out to match
    the payload exactly. ``header_bytes`` overrides the header raw bytes
    (payload must then cover its declared spans)."""
    path = os.path.join(root, name)
    if header_bytes is None:
        entries, data = _layout(tensors)
        header_bytes = json.dumps(entries).encode()
        if payload is None:
            payload = b"\0" * data
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload if payload is not None else b"")
    return path


# --------------------------------------------- tiny isomorphic contract


def tiny_params():
    return {
        "num_hidden_layers": 2,
        "num_experts": 2,
        "num_experts_per_tok": 2,
        "nvfp4_weight_shapes": {
            "gate_proj": [8, 16],
            "up_proj": [8, 16],
            "down_proj": [16, 8],
        },
        "mtp_group_size": 16,
        "mtp_weight_shapes": {
            "gate_proj": [64, 256],
            "up_proj": [64, 256],
            "down_proj": [256, 64],
        },
        "mtp_experts": 2,
        "ple_shard_count": 2,
        "ple_shard_shape": [1000, 160],
    }


def expected_contract(**overrides):
    """Build an explicit expected contract (same schema as the trusted one)."""
    p = tiny_params()
    p.update(overrides)
    gs = p["mtp_group_size"]
    w = p["mtp_weight_shapes"]
    mtp_scale_grids = {
        proj: [max(1, -(-shape[0] // gs)), max(1, -(-shape[1] // gs))]
        for proj, shape in w.items()
    }
    nw = p["nvfp4_weight_shapes"]
    nvfp4_scale_shapes = {
        proj: [shape[0], shape[1] * 2 // 16] for proj, shape in nw.items()
    }
    contract = {
        "model_id": REAL_MODEL_ID,
        "model_sha": REAL_MODEL_SHA,
        "architecture": "Qwen4ExpForConditionalGeneration",
        "num_hidden_layers": p["num_hidden_layers"],
        "num_experts": p["num_experts"],
        "num_experts_per_tok": p["num_experts_per_tok"],
        "native_context": 262144,
        "quant_algo": "MIXED_PRECISION",
        "mtp_alias_by_entrypoint": {
            "config.json": "FP8_PB_WO",
            "hf_quant_config.json": "FP8_BLOCK_SCALES",
        },
        "mtp_prefix": "mtp.",
        "mtp": {
            "experts": p["mtp_experts"],
            "group_size": gs,
            "weight_dtype": "F8_E4M3",
            "scale_name": "weight_scale_inv",
            "scale_dtypes": ("BF16", "F32"),
            "projections": {k: list(v) for k, v in w.items()},
            "scale_grids": mtp_scale_grids,
        },
        "ple": {
            "shard_count": p["ple_shard_count"],
            "shard_name": PLE_SHARD_NAME,
            "shard_dtype": "F8_E4M3",
            "shard_shape": list(p["ple_shard_shape"]),
            "scale_name": PLE_SCALE_NAME,
            "scale_dtype": "BF16",
            "scale_shape": [1],
        },
        "nvfp4": {
            "experts": p["num_experts"],
            "group_size": 16,
            "weight_dtype": "U8",
            "scale_dtype": "F8_E4M3",
            "scale_2_dtype": "F32",
            "input_scale_dtype": "F32",
            "projections": ("gate_proj", "up_proj", "down_proj"),
            "weight_shapes": {k: list(v) for k, v in nw.items()},
            "scale_shapes": nvfp4_scale_shapes,
        },
        "bf16_families": (
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.",
            ".mlp.gate.weight",
            "shared_expert.",
        ),
        "bf16_families_count": None,
        "nonexpert_dtypes": ("BF16", "I64"),
        "mtp_shard": "model-fp8-mtp-ple.safetensors",
    }
    # self-consistent derived counts for the tiny fixture
    contract["index_key_count"] = (
        # NVFP4 experts: layers x experts x projections x 4 tensor kinds
        contract["num_hidden_layers"] * contract["nvfp4"]["experts"] * 3 * 4
        # + 6 hand-written BF16 family tensors
        + 6
        # + per-layer BF16 q_proj
        + contract["num_hidden_layers"]
        + contract["mtp"]["experts"] * 3 * 2
        + 4
        + contract["ple"]["shard_count"]
        + 1
    )
    contract["mtp_key_count"] = (
        contract["mtp"]["experts"] * 3 * 2  # expert weights + scale_inv
        + 4  # hand-written BF16 MTP tensors
    )
    contract["shard_count"] = 2
    contract["ple_payload_bytes"] = (
        contract["ple"]["shard_shape"][0]
        * contract["ple"]["shard_shape"][1]
        * contract["ple"]["shard_count"]
    )
    return contract


def build_tiny_checkpoint(root, contract=None):
    """Physically build a checkpoint matching ``expected_contract()``."""
    os.makedirs(root, exist_ok=True)
    c = contract or expected_contract()
    P = "model.language_model.layers.%d.mlp.experts.%d.%s.%s"
    main = {}
    for layer in range(c["num_hidden_layers"]):
        for expert in range(c["nvfp4"]["experts"]):
            for proj in c["nvfp4"]["projections"]:
                main[P % (layer, expert, proj, "weight")] = {
                    "dtype": c["nvfp4"]["weight_dtype"],
                    "shape": c["nvfp4"]["weight_shapes"][proj],
                }
                main[P % (layer, expert, proj, "weight_scale")] = {
                    "dtype": c["nvfp4"]["scale_dtype"],
                    "shape": c["nvfp4"]["scale_shapes"][proj],
                }
                main[P % (layer, expert, proj, "weight_scale_2")] = {
                    "dtype": c["nvfp4"]["scale_2_dtype"],
                    "shape": [],
                }
                main[P % (layer, expert, proj, "input_scale")] = {
                    "dtype": c["nvfp4"]["input_scale_dtype"],
                    "shape": [],
                }
    for name in (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.visual.patch_embed.proj.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ):
        main[name] = {"dtype": "BF16", "shape": [4, 4]}
    for layer in range(c["num_hidden_layers"]):
        main["model.language_model.layers.%d.self_attn.q_proj.weight" % layer] = {
            "dtype": "BF16",
            "shape": [4, 4],
        }

    mtp = {}
    M = "mtp.layers.0.mlp.experts.%d.%s.%s"
    for expert in range(c["mtp"]["experts"]):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            mtp[M % (expert, proj, "weight")] = {
                "dtype": c["mtp"]["weight_dtype"],
                "shape": c["mtp"]["projections"][proj],
            }
            mtp[M % (expert, proj, c["mtp"]["scale_name"])] = {
                "dtype": "BF16",
                "shape": c["mtp"]["scale_grids"][proj],
            }
    for name in (
        "mtp.fc_embedding.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    ):
        mtp[name] = {"dtype": "BF16", "shape": [4, 4]}
    for n in range(c["ple"]["shard_count"]):
        mtp[c["ple"]["shard_name"].format(n)] = {
            "dtype": c["ple"]["shard_dtype"],
            "shape": c["ple"]["shard_shape"],
        }
    mtp[c["ple"]["scale_name"]] = {
        "dtype": c["ple"]["scale_dtype"],
        "shape": c["ple"]["scale_shape"],
    }

    main_shard = "model-00001-of-00002.safetensors"
    _write_shard(root, main_shard, main)
    _write_shard(root, c["mtp_shard"], mtp)

    weight_map = {name: main_shard for name in main}
    weight_map.update({name: c["mtp_shard"] for name in mtp})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": None}, "weight_map": weight_map}, fh)

    config = {
        "architectures": [c["architecture"]],
        "model_type": "qwen4_exp",
        "quantization_config": {
            "quant_algo": "MIXED_PRECISION",
            "quant_method": "modelopt",
            "quantized_layers": {
                "mtp.layers.0.mlp.experts": {
                    "quant_algo": c["mtp_alias_by_entrypoint"]["config.json"],
                    "group_size": c["mtp"]["group_size"],
                },
                PLE_PREFIX: {"quant_algo": "FP8"},
            },
        },
        "text_config": {
            "dtype": "bfloat16",
            "num_hidden_layers": c["num_hidden_layers"],
            "num_experts": c["num_experts"],
            "num_experts_per_tok": c["num_experts_per_tok"],
            "max_position_embeddings": c["native_context"],
        },
        "vision_config": {"dtype": "bfloat16"},
    }
    for layer in range(c["num_hidden_layers"]):
        config["quantization_config"]["quantized_layers"][
            "model.language_model.layers.%d.mlp.experts" % layer
        ] = {"quant_algo": "NVFP4", "group_size": c["nvfp4"]["group_size"]}
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(config, fh)
    hf_quant = {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "group_size": c["nvfp4"]["group_size"],
            "quantized_layers": {
                "mtp.layers.0.mlp.experts": {
                    "quant_algo": c["mtp_alias_by_entrypoint"]["hf_quant_config.json"],
                    "group_size": c["mtp"]["group_size"],
                },
                PLE_PREFIX: {"quant_algo": "FP8"},
            },
        },
    }
    for layer in range(c["num_hidden_layers"]):
        hf_quant["quantization"]["quantized_layers"][
            "model.language_model.layers.%d.mlp.experts" % layer
        ] = {"quant_algo": "NVFP4", "group_size": c["nvfp4"]["group_size"]}
    with open(os.path.join(root, "hf_quant_config.json"), "w") as fh:
        json.dump(hf_quant, fh)
    return root


@pytest.fixture
def tiny_checkpoint(tmp_path):
    return build_tiny_checkpoint(str(tmp_path / "model"))


@pytest.fixture
def tiny():
    return expected_contract()


# ============================================================ REAL metadata


def _load_fixture(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(os.path.join(FIXTURES, name), "rt") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def real_config():
    return _load_fixture("config.json")


@pytest.fixture(scope="module")
def real_hf_quant():
    return _load_fixture("hf_quant_config.json")


@pytest.fixture(scope="module")
def real_index():
    return _load_fixture("model.safetensors.index.json.gz")


@pytest.fixture(scope="module")
def real_mtp_ple_header():
    return _load_fixture("mtp_ple.header.json")


@pytest.fixture(scope="module")
def real_shard10_header():
    return _load_fixture("shard10.header.json")


def test_real_config_matches_trusted_contract(real_config):
    c = ac.TRUSTED_MODEL_CONTRACT
    assert real_config["architectures"] == [c["architecture"]]
    tc = real_config["text_config"]
    assert tc["num_hidden_layers"] == c["num_hidden_layers"]
    assert tc["num_experts"] == c["num_experts"]
    assert tc["num_experts_per_tok"] == c["num_experts_per_tok"]
    assert tc["max_position_embeddings"] == c["native_context"]
    ac.validate_config(real_config, ac.TRUSTED_MODEL_CONTRACT)


def test_real_quant_alias_pair(real_config, real_hf_quant):
    """The two entry points are semantically equal but literally different:
    config.json says FP8_PB_WO, hf_quant_config.json says FP8_BLOCK_SCALES.
    Both must be accepted against the trusted contract."""
    ac.validate_quant_metadata(
        real_config["quantization_config"],
        real_hf_quant["quantization"],
        ac.TRUSTED_MODEL_CONTRACT,
    )
    assert (
        real_config["quantization_config"]["quantized_layers"][
            "mtp.layers.0.mlp.experts"
        ]["quant_algo"]
        == "FP8_PB_WO"
    )
    assert (
        real_hf_quant["quantization"]["quantized_layers"]["mtp.layers.0.mlp.experts"][
            "quant_algo"
        ]
        == "FP8_BLOCK_SCALES"
    )


def test_real_quant_alias_wrong_branch_rejected(real_config, real_hf_quant):
    """A file claiming the *other* entry point's literal alias is rejected."""
    import copy

    swapped = copy.deepcopy(real_hf_quant)
    swapped["quantization"]["quantized_layers"]["mtp.layers.0.mlp.experts"][
        "quant_algo"
    ] = "FP8_PB_WO"
    with pytest.raises(ac.AuditError):
        ac.validate_quant_metadata(
            real_config["quantization_config"],
            swapped["quantization"],
            ac.TRUSTED_MODEL_CONTRACT,
        )


def test_real_index_shape_facts(real_index):
    c = ac.TRUSTED_MODEL_CONTRACT
    facts = ac.derive_index_facts(real_index, c)
    assert facts["key_count"] == 299545 == c["index_key_count"]
    assert facts["mtp_key_count"] == 3101 == c["mtp_key_count"]
    assert facts["file_count"] == 11 == c["shard_count"]
    assert facts["ple_shard_count"] == 128 == c["ple"]["shard_count"]
    assert (
        facts["referenced_files"][
            real_index["weight_map"]["mtp.layers.0.mlp.experts.0.gate_proj.weight"]
        ]
        > 0
    )


def test_real_mtp_ple_header_entry_facts(real_mtp_ple_header):
    """Every MTP expert entry in the REAL header: F8_E4M3 [640,2560]/[2560,640]
    weights with BF16 scale_inv grids [5,20]/[20,5] for all 512 experts."""
    c = ac.TRUSTED_MODEL_CONTRACT
    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    ac.validate_mtp_expert_family(header, c)
    ac.validate_ple_family(header, c)
    # independently re-derive the PLE payload bytes from the header spans
    spans = sum(
        header[PLE_SHARD_NAME.format(n)]["data_offsets"][1]
        - header[PLE_SHARD_NAME.format(n)]["data_offsets"][0]
        for n in range(c["ple"]["shard_count"])
    )
    assert spans == 51200245760


def test_real_mtp_scale_missing_rejected_by_pure_check(real_mtp_ple_header):
    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    probe = "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"
    removed = header.pop(probe)
    assert removed["dtype"] == "BF16"
    assert removed["shape"] == [5, 20]
    with pytest.raises(ac.AuditError):
        ac.validate_mtp_expert_family(header, ac.TRUSTED_MODEL_CONTRACT)


def test_real_mtp_scale_wrong_dtype_rejected(real_mtp_ple_header):
    import copy

    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    probe = "mtp.layers.0.mlp.experts.17.down_proj.weight_scale_inv"
    assert header[probe]["shape"] == [20, 5]
    header = copy.deepcopy(header)
    header[probe]["dtype"] = "F4"
    with pytest.raises(ac.AuditError):
        ac.validate_mtp_expert_family(header, ac.TRUSTED_MODEL_CONTRACT)


def test_real_ple_shard_count_vs_file_names(real_mtp_ple_header):
    """Tensor-vs-file count: the index claims 128 PLE shards; the header must
    carry exactly shard_0..shard_127 plus the single BF16 weight_scale."""
    c = ac.TRUSTED_MODEL_CONTRACT
    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    # drop one shard tensor: count mismatch must be rejected
    header.pop(PLE_SHARD_NAME.format(127))
    with pytest.raises(ac.AuditError):
        ac.validate_ple_family(header, c)


def test_real_shard10_header_all_bf16(real_shard10_header):
    header = dict(real_shard10_header)
    header.pop("__metadata__", None)
    ac.validate_tensor_entries("model-00010", header)
    assert all(v["dtype"] == "BF16" for v in header.values())
    assert header["mtp.layers.0.mlp.gate.weight"]["shape"] == [512, 2560]
    assert header["mtp.layers.0.mlp.shared_expert.gate_proj.weight"]["shape"] == [
        640,
        2560,
    ]


def test_real_index_claims_but_header_absent(real_index, real_mtp_ple_header):
    """A name the index assigns to the mtp-ple file but the header lacks is a
    cross-reference failure even though the index is internally consistent."""
    c = ac.TRUSTED_MODEL_CONTRACT
    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    victim = PLE_SCALE_NAME
    assert real_index["weight_map"][victim] == c["mtp_shard"]
    header.pop(victim)
    with pytest.raises(ac.AuditError):
        ac.validate_index_header_agreement(
            {c["mtp_shard"]: header}, real_index["weight_map"]
        )


def test_real_index_key_count_is_derived_not_assumed(real_index, real_mtp_ple_header):
    """If a key were added to the index without a header entry, the derived
    count check must fail — even though the raw count would still be >0."""
    import copy

    index = copy.deepcopy(real_index)
    header = dict(real_mtp_ple_header)
    header.pop("__metadata__", None)
    index["weight_map"]["brand.new.tensor"] = "model-fp8-mtp-ple.safetensors"
    with pytest.raises(ac.AuditError):
        ac.validate_index_header_agreement(
            {"model-fp8-mtp-ple.safetensors": header}, index["weight_map"]
        )


def test_real_config_duplicate_key_rejected():
    raw = (
        '{"architectures": ["Qwen4ExpForConditionalGeneration"], '
        '"architectures": ["X"]}'
    )
    with pytest.raises(ac.AuditError):
        ac.strict_json_loads(raw, "config.json")


# ============================================ trusted default is STRICT


def test_production_cli_rejects_tiny_checkpoint(tiny_checkpoint):
    """The production entrypoint uses the trusted real contract. A tiny
    synthetic checkpoint that is perfectly valid under a scaled contract
    must still be REJECTED by the production CLI default."""
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint)


def test_production_rejects_tampered_config_architecture(tmp_path):
    root = build_tiny_checkpoint(str(tmp_path / "model"))
    path = os.path.join(root, "config.json")
    with open(path) as handle:
        config = json.load(handle)
    config["architectures"] = ["LlamaForCausalLM"]
    with open(path, "w") as handle:
        json.dump(config, handle)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(root, contract=expected_contract())


# ================================================== end-to-end tiny audit


def test_tiny_checkpoint_admitted_end_to_end(tiny_checkpoint, tiny):
    findings = ac.audit_checkpoint(tiny_checkpoint, contract=tiny)
    assert findings["verdict"] == ac.HEADERS_ADMITTED
    assert findings["checkpoint_files"]["shard_count"] == 2
    assert findings["index_key_count"] == tiny["index_key_count"]
    assert findings["mtp_key_count"] == tiny["mtp_key_count"]
    assert findings["mtp"]["expert_count"] == tiny["mtp"]["experts"]
    assert findings["mtp"]["projection_count"] == tiny["mtp"]["experts"] * 3
    assert findings["ple"]["shard_count"] == tiny["ple"]["shard_count"]
    ple_bytes = tiny["ple"]["shard_shape"][0] * tiny["ple"]["shard_shape"][1]
    assert findings["ple"]["payload_bytes"] == ple_bytes * tiny["ple"]["shard_count"]
    assert findings["nvfp4"]["layer_count"] == tiny["num_hidden_layers"]
    assert findings["nvfp4"]["expert_entries"] == (
        tiny["num_hidden_layers"] * tiny["nvfp4"]["experts"] * 3
    )


def test_audit_verdict_is_not_full_verification(tiny_checkpoint, tiny):
    findings = ac.audit_checkpoint(tiny_checkpoint, contract=tiny)
    assert "NOT full integrity" in findings["verdict_note"]
    assert findings["verdict"] != vf.RECEIPT_KIND


def test_audit_preserves_original_files(tiny_checkpoint, tiny):
    before = {
        name: open(os.path.join(tiny_checkpoint, name), "rb").read()
        for name in os.listdir(tiny_checkpoint)
    }
    ac.audit_checkpoint(tiny_checkpoint, contract=tiny)
    after = {
        name: open(os.path.join(tiny_checkpoint, name), "rb").read()
        for name in os.listdir(tiny_checkpoint)
    }
    assert before == after


def test_audit_records_file_identity(tiny_checkpoint, tiny):
    findings = ac.audit_checkpoint(tiny_checkpoint, contract=tiny)
    root_stat = os.stat(tiny_checkpoint)
    assert findings["root"] == os.path.abspath(tiny_checkpoint)
    assert findings["root_stat"]["inode"] == root_stat.st_ino
    assert findings["checkpoint_files"]["total_bytes"] == sum(
        s["size"] for s in findings["checkpoint_files"]["shards"]
    )


# --------------------------------------------- dimension/dtype mutations


def _mutate_shard_entry(root, shard, tensor, mutate):
    """Load a shard header, mutate one tensor entry, rewrite the shard."""
    path = os.path.join(root, shard)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        payload = fh.read()
    entry = header[tensor]
    mutate(entry)
    _write_shard(
        root, shard, None, payload=payload, header_bytes=json.dumps(header).encode()
    )


def test_audit_rejects_wrong_weight_dimension(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("shape", [64, 128]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_wrong_weight_dtype(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("dtype", "BF16"),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_missing_scale_tensor(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.layers.0.mlp.experts.1.down_proj.weight_scale_inv"
    with open(os.path.join(tiny_checkpoint, shard), "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        payload = fh.read()
    del header[tensor]
    _write_shard(
        tiny_checkpoint,
        shard,
        None,
        payload=payload,
        header_bytes=json.dumps(header).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_wrong_scale_grid(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("shape", [5, 20]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_offsets_outside_file(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("data_offsets", [0, 1 << 40]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_bool_offsets(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.fc_embedding.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("data_offsets", [True, 16]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_bool_shape_dim(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.fc_embedding.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("shape", [True, 4]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_unknown_dtype(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    tensor = "mtp.fc_embedding.weight"
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("dtype", "F4_X9"),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


# -------------------------------------------------- index-vs-header facts


def test_audit_rejects_claim_in_index_absent_in_header(tiny_checkpoint, tiny):
    index_path = os.path.join(tiny_checkpoint, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    index["weight_map"]["ghost.tensor"] = tiny["mtp_shard"]
    with open(index_path, "w") as fh:
        json.dump(index, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_header_entry_absent_from_index(tiny_checkpoint, tiny):
    """Directional: an extra header-only tensor is also a mismatch (the
    audit derives both directions independently)."""
    shard = tiny["mtp_shard"]
    with open(os.path.join(tiny_checkpoint, shard), "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        payload = fh.read()
    header["sneaky.extra.weight"] = {
        "dtype": "BF16",
        "shape": [1, 1],
        "data_offsets": [0, 2],
    }
    _write_shard(
        tiny_checkpoint,
        shard,
        None,
        payload=payload + b"\0\0",
        header_bytes=json.dumps(header).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_missing_index_shard(tiny_checkpoint, tiny):
    index_path = os.path.join(tiny_checkpoint, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    index["weight_map"]["a"] = "model-99999-of-00002.safetensors"
    with open(index_path, "w") as fh:
        json.dump(index, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_extra_shard_file(tiny_checkpoint, tiny):
    """A present shard the index never references — file-count alone must
    not admit it and name-set equality must fail."""
    _write_shard(
        tiny_checkpoint,
        "model-extra.safetensors",
        {"x.weight": {"dtype": "BF16", "shape": [1, 1]}},
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_escaping_weight_map_path(tiny_checkpoint, tiny):
    index_path = os.path.join(tiny_checkpoint, "model.safetensors.index.json")
    with open(index_path) as fh:
        index = json.load(fh)
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = "../../etc/passwd"
    with open(index_path, "w") as fh:
        json.dump(index, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


# ---------------------------------------------------------- PLE payload


def test_audit_rejects_ple_payload_drift(tiny_checkpoint, tiny):
    """PLE payload bytes are derived from header spans, not file counts."""
    shard = tiny["mtp_shard"]
    tensor = PLE_SHARD_NAME.format(0)
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        tensor,
        lambda e: e.__setitem__("shape", [999, 160]),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_ple_scale_wrong_dtype(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    _mutate_shard_entry(
        tiny_checkpoint,
        shard,
        PLE_SCALE_NAME,
        lambda e: e.__setitem__("dtype", "F32"),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


# ------------------------------------------------------ header mechanics


def test_audit_rejects_oversized_header(tiny_checkpoint, tiny):
    _write_shard(
        tiny_checkpoint,
        "model-00003-of-00002.safetensors",
        {},
        header_bytes=json.dumps(
            {"k": "y" * (ac.SAFETENSORS_HEADER_LIMIT + 1)}
        ).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_corrupt_header_json(tiny_checkpoint, tiny):
    _write_shard(
        tiny_checkpoint,
        "model-00004-of-00002.safetensors",
        {},
        header_bytes=b"{not json",
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_escaping_tensor_name(tiny_checkpoint, tiny):
    _write_shard(
        tiny_checkpoint,
        "model-00005-of-00002.safetensors",
        {"../escape": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_truncated_header(tiny_checkpoint, tiny):
    path = os.path.join(tiny_checkpoint, tiny["mtp_shard"])
    size = os.path.getsize(path)
    with open(path, "r+b") as fh:
        fh.truncate(size - 8)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_lfs_pointer_shard(tiny_checkpoint, tiny):
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abcd\nsize 12\n"
    _write_shard(
        tiny_checkpoint,
        "model-00006-of-00002.safetensors",
        {},
        header_bytes=pointer,
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_shard_smaller_than_header_declares(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    path = os.path.join(tiny_checkpoint, shard)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        payload = fh.read()
    # claim a huge payload span but keep the file small
    big = max(header.values(), key=lambda e: e["data_offsets"][1])
    big["data_offsets"][1] += 10 * 1024 * 1024
    _write_shard(
        tiny_checkpoint,
        shard,
        None,
        payload=payload,
        header_bytes=json.dumps(header).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_missing_metadata(tiny_checkpoint, tiny):
    os.remove(os.path.join(tiny_checkpoint, "hf_quant_config.json"))
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_audit_rejects_symlink_shard(tiny_checkpoint, tiny):
    shard = tiny["mtp_shard"]
    os.rename(
        os.path.join(tiny_checkpoint, shard),
        os.path.join(tiny_checkpoint, shard + ".real"),
    )
    os.symlink(shard + ".real", os.path.join(tiny_checkpoint, shard))
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


# --------------------------------------------------- alias branch checks


def test_alias_wrong_branch_rejected(tiny_checkpoint, tiny):
    """Both aliases are semantic equals only in their sanctioned pairing;
    swapping the literals between the two entry points must fail."""
    config_path = os.path.join(tiny_checkpoint, "config.json")
    with open(config_path) as fh:
        config = json.load(fh)
    config["quantization_config"]["quantized_layers"]["mtp.layers.0.mlp.experts"][
        "quant_algo"
    ] = "FP8_BLOCK_SCALES"
    with open(config_path, "w") as fh:
        json.dump(config, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_alias_unknown_value_rejected(tiny_checkpoint, tiny):
    config_path = os.path.join(tiny_checkpoint, "config.json")
    with open(config_path) as fh:
        config = json.load(fh)
    config["quantization_config"]["quantized_layers"]["mtp.layers.0.mlp.experts"][
        "quant_algo"
    ] = "INT8"
    with open(config_path, "w") as fh:
        json.dump(config, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


def test_quant_algo_mismatch_between_entrypoints_rejected(tiny_checkpoint, tiny):
    config_path = os.path.join(tiny_checkpoint, "config.json")
    with open(config_path) as fh:
        config = json.load(fh)
    config["quantization_config"]["quant_algo"] = "FULL_PRECISION"
    with open(config_path, "w") as fh:
        json.dump(config, fh)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(tiny_checkpoint, contract=tiny)


# ============================================================== VERIFY

LOCK_ENTRY_FIELDS = ("path", "size", "sha256", "git_blob")


def _hash_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _git_blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _lock_doc(root, names, model=None, *, omit_sha=False):
    files = []
    for name in names:
        path = os.path.join(root, name)
        entry = {
            "path": name,
            "size": os.path.getsize(path),
            "sha256": None if omit_sha else _hash_file(path),
            "git_blob": _git_blob(path),
        }
        files.append(entry)
    return {
        "model": model or {"id": REAL_MODEL_ID, "sha": REAL_MODEL_SHA, "files": files}
    }


def _write_lock(tmp_path, doc):
    path = tmp_path / "sources.lock.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_verify_streams_full_hashes_from_real_lock_schema(
    tmp_path, tiny_checkpoint, tiny
):
    """sources.json['model'] schema: id/sha/files[{path,size,sha256,git_blob}].
    LFS entries (sha256) stream the whole file; metadata entries verify by
    Git blob SHA-1."""
    names = ["config.json", "hf_quant_config.json", tiny["mtp_shard"]]
    lock = _lock_doc(tiny_checkpoint, names)
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    assert receipt["kind"] == vf.RECEIPT_KIND
    assert receipt["model"] == {"id": REAL_MODEL_ID, "sha": REAL_MODEL_SHA}
    assert receipt["file_count"] == 3
    assert receipt["verification"] == "FULL"
    by_path = {f["path"]: f for f in receipt["files"]}
    assert by_path["config.json"]["source_git_blob"] == _git_blob(
        os.path.join(tiny_checkpoint, "config.json")
    )
    assert (
        by_path[tiny["mtp_shard"]]["expected_sha256"]
        == lock["model"]["files"][2]["sha256"]
    )
    assert "locks_sha256" in receipt


def test_verify_receipt_stat_identity(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    entry = receipt["files"][0]
    st = os.stat(os.path.join(tiny_checkpoint, "config.json"))
    assert entry["inode"] == st.st_ino
    assert entry["mtime_ns"] == st.st_mtime_ns
    assert entry["ctime_ns"] == st.st_ctime_ns
    assert entry["size"] == st.st_size
    assert entry["dev"] == st.st_dev
    assert receipt["root_stat"]["inode"] == os.stat(tiny_checkpoint).st_ino


def test_verify_rejects_wrong_remote_hash(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0]["sha256"] = "0" * 64
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_wrong_git_blob(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"], omit_sha=True)
    lock["model"]["files"][0]["git_blob"] = "0" * 40
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_blob_only_entry_accepted(tmp_path, tiny_checkpoint):
    """A lock entry with sha256:null and a trusted git_blob must PASS on
    matching blob bytes — this is the real shape of metadata entries."""
    lock = _lock_doc(tiny_checkpoint, ["config.json"], omit_sha=True)
    assert lock["model"]["files"][0]["sha256"] is None
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    assert receipt["file_count"] == 1


def test_verify_rejects_hashless_entry(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0].pop("sha256")
    lock["model"]["files"][0].pop("git_blob")
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_wrong_byte_count(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0]["size"] += 1
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_escaping_lock_path(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0]["path"] = "../../escape"
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_absolute_lock_path(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0]["path"] = "/etc/passwd"
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_duplicate_lock_path(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"].append(dict(lock["model"]["files"][0]))
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_missing_file(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    lock["model"]["files"][0]["path"] = "gone.bin"
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_symlink_file(tmp_path, tiny_checkpoint):
    target = os.path.join(tiny_checkpoint, "config.json")
    outside = tmp_path / "outside.json"
    outside.write_bytes(open(target, "rb").read())
    os.rename(target, target + ".real")
    os.symlink(str(outside), target)
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    # Keep the symlink in place: replacing it with the regular file would
    # test a normal path rather than the requested rejection boundary.
    inv_path = _write_lock(tmp_path, lock)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(tiny_checkpoint, inv_path)


def test_verify_rejects_constant_model_sha_without_lock(tmp_path, tiny_checkpoint):
    """The receipt model id/sha must come from the trusted lock; a lock
    claiming a different revision must poison the receipt, not be silently
    replaced by the pinned constant."""
    lock = _lock_doc(
        tiny_checkpoint,
        ["config.json"],
        model={
            "id": "evil/model",
            "sha": "f" * 40,
            "files": _lock_doc(tiny_checkpoint, ["config.json"])["model"]["files"],
        },
    )
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    assert receipt["model"] == {"id": "evil/model", "sha": "f" * 40}


def test_verify_files_missing_inventory_fails(tmp_path):
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(tmp_path / "nope"), str(tmp_path / "nope2"))


# ------------------------------------------------------ receipt forging


def test_forged_receipt_rejected_by_tree_recheck(tmp_path, tiny_checkpoint):
    """A coordinated caller forges a receipt (no verification ever ran).
    The re-check must recompute file hashes and reject the fake."""
    forged = {
        "kind": vf.RECEIPT_KIND,
        "model": {"id": REAL_MODEL_ID, "sha": REAL_MODEL_SHA},
        "root": os.path.abspath(tiny_checkpoint),
        "file_count": 1,
        "total_bytes": 1,
        "locks_sha256": "0" * 64,
        "verification": "FULL",
        "files": [
            {
                "path": "config.json",
                "size": os.path.getsize(os.path.join(tiny_checkpoint, "config.json")),
                "sha256": "0" * 64,
                "expected_sha256": "0" * 64,
                "expected_git_blob": None,
                "inode": 1,
                "dev": 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            }
        ],
    }
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(forged, receipt_path)
    loaded = vf.load_receipt(receipt_path)
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(loaded, tiny_checkpoint)


def test_receipt_recheck_recomputes_hashes(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(receipt, receipt_path)
    loaded = vf.load_receipt(receipt_path)
    vf.check_receipt_against_tree(loaded, tiny_checkpoint)
    # flip one byte: live hash no longer matches the verified expectation
    target = os.path.join(tiny_checkpoint, "config.json")
    data = bytearray(open(target, "rb").read())
    data[5] ^= 0xFF
    with open(target, "r+b") as fh:
        fh.seek(5)
        fh.write(bytes(data[5:6]))
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(loaded, tiny_checkpoint)


def test_receipt_recheck_detects_every_stat_drift(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    inv_path = _write_lock(tmp_path, lock)
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(vf.verify_files(tiny_checkpoint, inv_path), receipt_path)
    receipt = vf.load_receipt(receipt_path)
    target = os.path.join(tiny_checkpoint, "config.json")

    st = os.stat(target)
    os.utime(target, ns=(st.st_mtime_ns + 1, st.st_ctime_ns))
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, tiny_checkpoint)
    os.utime(target, ns=(st.st_mtime_ns, st.st_ctime_ns))

    os.rename(target, target + ".hidden")
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, tiny_checkpoint)
    os.rename(target + ".hidden", target)

    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, str(tmp_path / "elsewhere"))


def test_load_receipt_rejects_corrupt_kind(tmp_path, tiny_checkpoint):
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    inv_path = _write_lock(tmp_path, lock)
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(vf.verify_files(tiny_checkpoint, inv_path), receipt_path)
    with open(receipt_path) as handle:
        receipt = json.load(handle)
    receipt["kind"] = "SOMETHING_ELSE"
    with open(receipt_path, "w") as handle:
        json.dump(receipt, handle)
    with pytest.raises(vf.VerifyError):
        vf.load_receipt(receipt_path)


def test_receipt_written_by_verify_has_no_full_status_without_files(
    tmp_path, tiny_checkpoint
):
    """HEADERS_ADMITTED (audit) and CHECKPOINT_VERIFIED (verify) are distinct
    verdicts; a receipt must never be produced from header admission alone."""
    lock = _lock_doc(tiny_checkpoint, ["config.json"])
    inv_path = _write_lock(tmp_path, lock)
    receipt = vf.verify_files(tiny_checkpoint, inv_path)
    assert receipt["kind"] == vf.RECEIPT_KIND
    assert receipt["kind"] != ac.HEADERS_ADMITTED
