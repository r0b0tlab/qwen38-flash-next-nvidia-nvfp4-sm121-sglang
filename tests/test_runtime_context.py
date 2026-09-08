"""Owned runtime context boundaries; no Docker or HTTP is executed here."""

import pytest
from runtime import profile_from_dict
from scripts import guard, runtime_context as c


def inputs():
    profile = profile_from_dict(
        {
            "schema": 1,
            "mode": "ar",
            "context_length": 32768,
            "max_total_tokens": 32768,
            "vision": {"backend": "triton_attn", "cuda_graph": True},
        }
    )
    record = {
        "cid": "a" * 64,
        "image": "sha256:" + "b" * 64,
        "nonce": "c" * 32,
        "profile_sha256": "d" * 64,
        "model": c.MODEL_ID,
        "model_sha": "e" * 40,
    }
    lock = {"sglang": {"tree": "f" * 40}}
    labels = {
        guard.OWNER_LABEL_KEY: record["nonce"],
        guard.LABEL_IMAGE: record["image"],
        guard.LABEL_PROFILE: record["profile_sha256"],
        "io.r0b0tlab.sglang.tree": lock["sglang"]["tree"],
        "io.r0b0tlab.model.sha": record["model_sha"],
    }
    doc = {
        "Id": record["cid"],
        "Image": record["image"],
        "Config": {"Labels": labels},
        "State": {"Running": True, "StartedAt": "test-only-epoch"},
        "HostConfig": {
            "PortBindings": {
                "30000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "30080"}]
            }
        },
    }
    server = {
        "status": "ready",
        "max_total_num_tokens": 32768,
        "max_req_input_len": 32767,
        "startup_time": 1.25,
        "launch_command": ["test-only"],
    }
    model = {
        "served_model_name": c.MODEL_ID,
        "model_path": "/model",
        "has_image_understanding": True,
        "architectures": ["Qwen4ExpForConditionalGeneration"],
    }
    return record, profile, lock, doc, server, model


def test_capture_reads_the_public_inputs_and_rechecks_container(tmp_path):
    import hashlib
    import json

    record, profile, lock, doc, server, model = inputs()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.raw))
    record["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    doc["Config"]["Labels"][guard.LABEL_PROFILE] = record["profile_sha256"]
    (tmp_path / "record.json").write_text(json.dumps(record))
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    calls = []

    class Transport:
        def inspect(self, cid):
            calls.append(cid)
            return doc

    def reader(base, path):
        return server if path == "/server_info" else model

    result = c.capture(
        tmp_path / "record.json",
        tmp_path / "lock.json",
        "http://127.0.0.1:30080",
        transport=Transport(),
        reader=reader,
    )
    assert (
        calls == [record["cid"]] * 2
        and result["epoch"]["profile_sha256"] == record["profile_sha256"]
    )


def test_context_is_derived_from_observed_capacity_and_identity():
    result = c.observed_context(*inputs(), "http://127.0.0.1:30080")
    assert (
        result["total_pool"] == 32768
        and result["epoch"]["container_started_at"] == "test-only-epoch"
    )


@pytest.mark.parametrize("fault", ["no_start", "wrong_port", "capacity", "model"])
def test_unbound_runtime_context_is_rejected(fault):
    data = list(inputs())
    if fault == "no_start":
        data[3]["State"].pop("StartedAt")
    if fault == "wrong_port":
        data[3]["HostConfig"]["PortBindings"]["30000/tcp"][0]["HostPort"] = "39999"
    if fault == "capacity":
        data[4]["max_total_num_tokens"] = 16384
    if fault == "model":
        data[5]["served_model_name"] = "other/model"
    with pytest.raises(ValueError):
        c.observed_context(*data, "http://127.0.0.1:30080")


@pytest.mark.parametrize(
    "url",
    [
        "http://user:secret@127.0.0.1:30080",
        "http://127.0.0.1:30080?secret=1",
        "http://127.0.0.1:30080#x",
    ],
)
def test_base_with_credentials_or_extra_url_components_refused_first(tmp_path, url):
    with pytest.raises(ValueError, match="loopback"):
        c.capture(tmp_path / "missing-record", tmp_path / "missing-lock", url)
