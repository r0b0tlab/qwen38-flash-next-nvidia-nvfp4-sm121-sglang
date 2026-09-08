"""Bind the native OAI positive-only cache-detail semantics to exact source.

The pinned OutputStreamer returns None iff all cache counters are zero;
positive hits survive the pinned detokenizer/tokenizer/OAI path when requested.
This permits a labeled zero-elision interpretation, never a synthetic SSE field.
"""

SCHEMA = "qwen38fn.cache-details-positive-only.v1"
EXPECTED_SOURCE_HASHES = {
    "srt/managers/scheduler_components/output_streamer.py": "67bd9278d0680c6a8b4ead37faa1970e2a5c200b676f62b73125a8461e3ea676",
    "srt/managers/detokenizer_manager.py": "90042a763006dba5a77d2a0e43abe17937ee287abaff13b9bf4b26f28f139bf4",
    "srt/managers/tokenizer_manager.py": "103c4f37de437850099c3a1727cbaa5ab3c387c5a1b411a2f34e7e94a9a88df0",
    "srt/managers/multi_tokenizer_mixin.py": "e3051fa6b5d84c78a0b27eaff80bd5a465eb27ed22b6628568428de4001cea51",
    "srt/managers/io_struct.py": "cb482b4c70faecea58b862cb6a8d87e26e2f605b944f5918657cfe2d54f82548",
    "srt/entrypoints/openai/serving_completions.py": "d015081efb5acf560b755a28e4964113e724c51a73f9151e86e65849125405e9",
    "srt/entrypoints/openai/protocol.py": "c9f05204379a6dd611d662c0350f97654800d76fcdeda5ba6f8dcc5f58b424f9",
}


def from_runtime_lock(lock):
    files = lock.get("sglang", {}).get("python_files", {})
    actual = {name: files.get(name) for name in EXPECTED_SOURCE_HASHES}
    contract = {"schema": SCHEMA, "source_hashes": actual}
    validate(contract)
    return contract


def validate(contract):
    if (
        not isinstance(contract, dict)
        or set(contract) != {"schema", "source_hashes"}
        or contract.get("schema") != SCHEMA
        or contract.get("source_hashes") != EXPECTED_SOURCE_HASHES
    ):
        raise ValueError("unverified native cache-reporting semantics")
    return True
