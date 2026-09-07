"""Audit + verify tests. All fixtures are synthetic tiny checkpoints in
tmp_path; no test touches the real model tree or any real path."""

import hashlib
import json
import os
import struct

import pytest

from scripts import audit_checkpoint as ac
from scripts import verify_files as vf

ARCH = "Qwen4ExpForConditionalGeneration"


def _write_shard(root, name, tensors, payload=b"\0" * 16, header=None):
    path = os.path.join(root, name)
    header_bytes = (
        json.dumps(tensors).encode() if header is None else header
    )
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)
    return path


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    r = str(root)
    _write_shard(
        r,
        "model-00001.safetensors",
        {
            "model.layers.0.mlp.experts.0.weight": {
                "dtype": "F4",
                "shape": [16, 64],
                "data_offsets": [0, 16],
            }
        },
    )
    _write_shard(
        r,
        "model-fp8-mtp-ple.safetensors",
        {"mtp.layer.0.weight": {"dtype": "F8_E4M3", "shape": [4, 4],
                                "data_offsets": [0, 16]}},
    )
    with open(os.path.join(r, "config.json"), "w") as handle:
        json.dump(
            {
                "architectures": [ARCH],
                "quantization_config": {"group_size": 128},
            },
            handle,
        )
    with open(os.path.join(r, "hf_quant_config.json"), "w") as handle:
        json.dump({"quant_method": "modelopt", "group_size": 128}, handle)
    with open(os.path.join(r, "model.safetensors.index.json"), "w") as handle:
        json.dump(
            {
                "weight_map": {
                    "a": "model-00001.safetensors",
                    "b": "model-fp8-mtp-ple.safetensors",
                }
            },
            handle,
        )
    return r


# ---------------------------------------------------------------- audit

def test_audit_admits_synthetic_checkpoint(checkpoint):
    findings = ac.audit_checkpoint(checkpoint)
    assert findings["verdict"] == ac.HEADERS_ADMITTED
    assert findings["checkpoint_files"]["shard_count"] == 2
    assert findings["checkpoint_files"]["ple_shard_count"] == 1
    assert findings["architecture"] == ARCH


def test_audit_verdict_is_not_full_verification(checkpoint):
    findings = ac.audit_checkpoint(checkpoint)
    assert "NOT full integrity" in findings["verdict_note"]


def test_audit_preserves_original_files(checkpoint):
    before = {
        name: open(os.path.join(checkpoint, name), "rb").read()
        for name in os.listdir(checkpoint)
    }
    ac.audit_checkpoint(checkpoint)
    after = {
        name: open(os.path.join(checkpoint, name), "rb").read()
        for name in os.listdir(checkpoint)
    }
    assert before == after


def test_audit_rejects_missing_metadata(checkpoint):
    os.remove(os.path.join(checkpoint, "hf_quant_config.json"))
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_wrong_architecture(checkpoint):
    path = os.path.join(checkpoint, "config.json")
    with open(path) as handle:
        config = json.load(handle)
    config["architectures"] = ["LlamaForCausalLM"]
    with open(path, "w") as handle:
        json.dump(config, handle)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_missing_index_shard(checkpoint):
    path = os.path.join(checkpoint, "model.safetensors.index.json")
    with open(path) as handle:
        index = json.load(handle)
    index["weight_map"]["c"] = "model-99999.safetensors"
    with open(path, "w") as handle:
        json.dump(index, handle)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_escaping_weight_map_path(checkpoint):
    path = os.path.join(checkpoint, "model.safetensors.index.json")
    with open(path) as handle:
        index = json.load(handle)
    index["weight_map"]["a"] = "../../etc/passwd"
    with open(path, "w") as handle:
        json.dump(index, handle)
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_oversized_header(checkpoint):
    _write_shard(
        checkpoint,
        "model-00002.safetensors",
        {},
        header=json.dumps({"x": "y" * (ac.SAFETENSORS_HEADER_LIMIT + 1)}).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_corrupt_header_json(checkpoint):
    _write_shard(
        checkpoint,
        "model-00003.safetensors",
        None,
        header=b"{not json",
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


def test_audit_rejects_escaping_tensor_name(checkpoint):
    _write_shard(
        checkpoint,
        "model-00004.safetensors",
        {"../escape": {"dtype": "F4", "shape": [1], "data_offsets": [0, 16]}},
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(checkpoint)


# --------------------------------------------------------------- verify

def _inventory(root, files, model=None):
    entries = []
    for name in files:
        data = open(os.path.join(root, name), "rb").read()
        entries.append(
            {
                "path": name,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    doc = {
        "model": model
        or {
            "id": ac.EXPECTED_MODEL_ID,
            "sha": ac.EXPECTED_MODEL_SHA,
        },
        "files": entries,
    }
    return doc


def _write_inventory(tmp_path, doc):
    path = tmp_path / "model.files.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def test_verify_streams_full_hashes_and_writes_receipt(tmp_path, checkpoint):
    inventory = _inventory(
        checkpoint, ["config.json", "model-00001.safetensors"]
    )
    inv_path = _write_inventory(tmp_path, inventory)
    receipt = vf.verify_files(checkpoint, inv_path)
    assert receipt["kind"] == vf.RECEIPT_KIND
    assert receipt["file_count"] == 2
    assert receipt["model"]["sha"] == ac.EXPECTED_MODEL_SHA
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(receipt, receipt_path)
    loaded = vf.load_receipt(receipt_path)
    vf.check_receipt_against_tree(loaded, checkpoint)


def test_verify_receipt_captures_stat_identity(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inv_path = _write_inventory(tmp_path, inventory)
    receipt = vf.verify_files(checkpoint, inv_path)
    entry = receipt["files"][0]
    st = os.stat(os.path.join(checkpoint, "config.json"))
    assert entry["inode"] == st.st_ino
    assert entry["mtime_ns"] == st.st_mtime_ns
    assert entry["ctime_ns"] == st.st_ctime_ns
    assert entry["size"] == st.st_size


def test_verify_fails_wrong_bytes(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["model-00001.safetensors"])
    inv_path = _write_inventory(tmp_path, inventory)
    with open(os.path.join(checkpoint, "model-00001.safetensors"), "r+b") as fh:
        fh.seek(20)
        fh.write(b"X")
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_verify_fails_size_mismatch(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inventory["files"][0]["size"] += 1
    inv_path = _write_inventory(tmp_path, inventory)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_verify_fails_missing_hash_field(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    del inventory["files"][0]["sha256"]
    inv_path = _write_inventory(tmp_path, inventory)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_verify_fails_escaping_inventory_path(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inventory["files"][0]["path"] = "../../escape"
    inv_path = _write_inventory(tmp_path, inventory)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_verify_fails_duplicate_inventory_path(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inventory["files"].append(dict(inventory["files"][0]))
    inv_path = _write_inventory(tmp_path, inventory)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_verify_fails_missing_file(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inventory["files"][0]["path"] = "gone.bin"
    inv_path = _write_inventory(tmp_path, inventory)
    with pytest.raises(vf.VerifyError):
        vf.verify_files(checkpoint, inv_path)


def test_load_receipt_rejects_corrupt_kind(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inv_path = _write_inventory(tmp_path, inventory)
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(vf.verify_files(checkpoint, inv_path), receipt_path)
    with open(receipt_path) as handle:
        receipt = json.load(handle)
    receipt["kind"] = "SOMETHING_ELSE"
    with open(receipt_path, "w") as handle:
        json.dump(receipt, handle)
    with pytest.raises(vf.VerifyError):
        vf.load_receipt(receipt_path)


def test_receipt_tree_recheck_detects_every_stat_drift(tmp_path, checkpoint):
    inventory = _inventory(checkpoint, ["config.json"])
    inv_path = _write_inventory(tmp_path, inventory)
    receipt_path = str(tmp_path / "receipt.json")
    vf.write_receipt(vf.verify_files(checkpoint, inv_path), receipt_path)
    receipt = vf.load_receipt(receipt_path)
    target = os.path.join(checkpoint, "config.json")

    # mtime drift
    st = os.stat(target)
    os.utime(target, ns=(st.st_mtime_ns + 1, st.st_ctime_ns))
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, checkpoint)
    os.utime(target, ns=(st.st_mtime_ns, st.st_ctime_ns))

    # missing file
    os.rename(target, target + ".hidden")
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, checkpoint)
    os.rename(target + ".hidden", target)

    # wrong root
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, str(tmp_path / "elsewhere"))


def test_verify_files_missing_inventory_fails(tmp_path):
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(tmp_path / "nope"), str(tmp_path / "nope2"))
