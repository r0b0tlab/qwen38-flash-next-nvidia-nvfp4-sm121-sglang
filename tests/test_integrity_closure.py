"""Independent metadata/integrity closure probes, test-only checkpoints."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from scripts import audit_checkpoint as ac, verify_files as vf

# Load this repository's helper file without depending on a site-packages tests package.
_spec = importlib.util.spec_from_file_location(
    "integrity_fixture_helpers", Path(__file__).with_name("test_audit_and_verify.py")
)
assert _spec is not None and _spec.loader is not None
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)


@pytest.fixture
def checkpoint(tmp_path):
    c = f.expected_contract()
    root = Path(f.build_tiny_checkpoint(str(tmp_path / "model"), c))
    # Match the actual metadata contract's per-layer NVFP4 entries.
    for filename, field in (
        ("config.json", "quantization_config"),
        ("hf_quant_config.json", "quantization"),
    ):
        path = root / filename
        doc = json.loads(path.read_text())
        for layer in range(c["num_hidden_layers"]):
            doc[field]["quantized_layers"][
                f"model.language_model.layers.{layer}.mlp.experts"
            ] = {"quant_algo": "NVFP4", "group_size": 16}
        path.write_text(json.dumps(doc))
    return root, c


@pytest.mark.parametrize(
    "suffix,field,value",
    [
        ("weight", "dtype", "I8"),
        ("weight", "shape", [16, 8]),
        ("weight_scale", "dtype", "I8"),
        ("weight_scale_2", "dtype", "I32"),
    ],
)
def test_main_nvfp4_family_is_validated(checkpoint, suffix, field, value):
    root, contract = checkpoint
    tensor = "model.language_model.layers.0.mlp.experts.0.gate_proj." + suffix
    f._mutate_shard_entry(
        str(root),
        "model-00001-of-00002.safetensors",
        tensor,
        lambda e: e.__setitem__(field, value),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(str(root), contract=contract)


def test_target_nvfp4_metadata_group_enforced(checkpoint):
    root, contract = checkpoint
    p = root / "config.json"
    doc = json.loads(p.read_text())
    doc["quantization_config"]["quantized_layers"][
        "model.language_model.layers.0.mlp.experts"
    ]["group_size"] = 32
    p.write_text(json.dumps(doc))
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(str(root), contract=contract)


def test_header_dtype_shape_must_equal_span(tmp_path):
    h = {"test.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}}
    f._write_shard(
        str(tmp_path),
        "test.safetensors",
        None,
        payload=b"00",
        header_bytes=json.dumps(h).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_safetensors_headers(str(tmp_path))


def test_sum_and_max_do_not_prove_nonoverlapping_tiling(tmp_path):
    h = {
        "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "b": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
        "c": {"dtype": "U8", "shape": [2], "data_offsets": [4, 6]},
    }
    f._write_shard(
        str(tmp_path),
        "test.safetensors",
        None,
        payload=b"000000",
        header_bytes=json.dumps(h).encode(),
    )
    with pytest.raises(ac.AuditError):
        ac.audit_safetensors_headers(str(tmp_path))


def test_scalar_safetensors_are_supported(tmp_path):
    f._write_shard(
        str(tmp_path), "scalar.safetensors", {"scale": {"dtype": "F32", "shape": []}}
    )
    report, _ = ac.audit_safetensors_headers(str(tmp_path))
    assert report["shard_count"] == 1


def test_lfs_git_blob_is_pointer_identity_not_payload_hash(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    payload = b"test-only LFS payload"
    (model / "weight.bin").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(payload)}\n".encode()
    blob = hashlib.sha1(
        b"blob " + str(len(pointer)).encode() + b"\0" + pointer
    ).hexdigest()
    lock = {
        "model": {
            "id": f.REAL_MODEL_ID,
            "sha": f.REAL_MODEL_SHA,
            "files": [
                {
                    "path": "weight.bin",
                    "size": len(payload),
                    "sha256": digest,
                    "git_blob": blob,
                }
            ],
        }
    }
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    r = vf.verify_files(str(model), str(p))
    assert r["files"][0]["sha256"] == digest
    assert r["files"][0]["source_git_blob"] == blob
    assert r["files"][0]["git_blob"] is None


@pytest.mark.parametrize("revision", ["", "f" * 64, True])
def test_verifier_validates_revision_format_before_hashing(
    tmp_path, checkpoint, revision
):
    root, _ = checkpoint
    lock = f._lock_doc(str(root), ["config.json"])
    lock["model"]["sha"] = revision
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(root), str(p))


def test_empty_inventory_is_not_full_verification(tmp_path, checkpoint):
    root, _ = checkpoint
    p = tmp_path / "lock.json"
    p.write_text(
        json.dumps(
            {"model": {"id": f.REAL_MODEL_ID, "sha": f.REAL_MODEL_SHA, "files": []}}
        )
    )
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(root), str(p))


def test_lock_duplicate_keys_rejected(tmp_path, checkpoint):
    root, _ = checkpoint
    p = tmp_path / "lock.json"
    text = json.dumps(f._lock_doc(str(root), ["config.json"]))
    p.write_text(text[:-1] + ',"model":' + json.dumps(json.loads(text)["model"]) + "}")
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(root), str(p))


def test_symlink_ancestor_rejected(tmp_path, checkpoint):
    root, _ = checkpoint
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.bin").write_bytes(b"test-only")
    (root / "alias").symlink_to(outside, target_is_directory=True)
    lock = f._lock_doc(str(root), ["alias/data.bin"])
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    with pytest.raises(vf.VerifyError):
        vf.verify_files(str(root), str(p))


def test_hashless_receipt_with_correct_stats_rejected(tmp_path, checkpoint):
    root, _ = checkpoint
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(f._lock_doc(str(root), ["config.json"])))
    receipt = vf.verify_files(str(root), str(p))
    for row in receipt["files"]:
        for key in ("verified_sha256", "verified_git_blob", "sha256", "git_blob"):
            row.pop(key, None)
    with pytest.raises(vf.VerifyError):
        vf.check_receipt_against_tree(receipt, str(root))


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "mtp.layers.0.mlp.experts.0.gate_proj.weight",
            "mtp.layers.0.mlp.experts.00.gate_proj.weight",
        ),
        (f.PLE_SHARD_NAME.format(1), f.PLE_SHARD_NAME.format("01")),
    ],
)
def test_tensor_names_are_literal_not_numeric_aliases(checkpoint, old, new):
    import struct

    root, contract = checkpoint
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_path = root / index["weight_map"][old]
    with shard_path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
        payload = stream.read()
    header[new] = header.pop(old)
    index["weight_map"][new] = index["weight_map"].pop(old)
    f._write_shard(
        str(root),
        shard_path.name,
        None,
        payload=payload,
        header_bytes=json.dumps(header).encode(),
    )
    index_path.write_text(json.dumps(index))
    with pytest.raises(ac.AuditError):
        ac.audit_checkpoint(str(root), contract=contract)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_nonfinite_json_spellings_all_rejected(tmp_path, literal):
    text = '{"ignored":' + literal + "}"
    with pytest.raises(ac.AuditError):
        ac.strict_json_loads(text, "test-only")
    p = tmp_path / "input.json"
    p.write_text(text)
    with pytest.raises(vf.VerifyError):
        vf.load_inventory(str(p))
    from scripts import guard

    with pytest.raises(guard.LaunchError):
        guard.read_json(p)
