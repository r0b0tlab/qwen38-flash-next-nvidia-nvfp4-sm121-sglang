"""Client-only observed-wire adapter for the pinned upstream OAI benchmark.

No serving code changes. Preserve upstream request/timing/aggregation code;
capture actual payloads and usage, safely skip usage-only events in its text
parser, and reject rather than inherit requested-token count defaults.
"""

import contextvars
import hashlib
import inspect
import json
import os
from pathlib import Path

from .benchmark_evidence import input_hash

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
MAX_WIRE_BYTES = 4 * 1024 * 1024


class CaptureRecord:
    def __init__(self, expected, index, warmup, directory, sequence=0):
        if (
            expected.get("model") != MODEL_ID
            or not isinstance(expected.get("prompt"), list)
            or not expected["prompt"]
        ):
            raise ValueError("invalid expected request identity")
        if any(type(token) is not int or token < 0 for token in expected["prompt"]):
            raise ValueError("input token IDs must be exact nonnegative integers")
        self.expected = json.loads(json.dumps(expected, allow_nan=False))
        self.index, self.warmup = index, warmup
        self.request_value = None
        self.usage = None
        self.finish_reason = None
        self.models = set()
        self.cache_details = None
        self.saw_done = False
        self.errors = []
        self.bytes_received = 0
        self.stored_bytes = 0
        self.wire_hash = hashlib.sha256()
        self.path = (
            Path(directory)
            / f"{'warmup' if warmup else 'measured'}-{index}-{sequence}.sse"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("xb", buffering=0)

    def request(self, payload):
        if self.request_value is not None or input_hash(payload) != input_hash(
            self.expected
        ):
            raise ValueError("actual upstream request differs from frozen payload")
        self.request_value = json.loads(json.dumps(payload, allow_nan=False))

    def chunk(self, raw):
        if not isinstance(raw, bytes):
            raise ValueError("SSE capture requires bytes")
        self.bytes_received += len(raw)
        retained = raw[: max(0, MAX_WIRE_BYTES - self.stored_bytes)]
        self.stream.write(retained)
        self.wire_hash.update(retained)
        self.stored_bytes += len(retained)
        if self.bytes_received > MAX_WIRE_BYTES:
            self.errors.append("wire_limit_exceeded")
            raise ValueError("bounded SSE capture exceeded")

    def event(self, data):
        if not isinstance(data, dict):
            self.errors.append("nonobject_event")
            return
        if isinstance(data.get("model"), str):
            self.models.add(data["model"])
        usage = data.get("usage")
        if usage is not None:
            if not isinstance(usage, dict) or any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                self.errors.append("invalid_usage")
            else:
                if (
                    self.usage
                    and usage["completion_tokens"] < self.usage["completion_tokens"]
                ):
                    self.errors.append("decreasing_usage")
                self.usage = dict(usage)
        choices = data.get("choices") or []
        if (
            choices
            and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") is not None
        ):
            self.finish_reason = choices[0]["finish_reason"]
        extension = data.get("sglext") or {}
        if (
            isinstance(extension, dict)
            and extension.get("cached_tokens_details") is not None
        ):
            self.cache_details = extension["cached_tokens_details"]

    def done(self):
        self.saw_done = True

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def finalize(self, output):
        self.close()  # raw bytes survive even when validation below fails
        if self.request_value is None:
            self.errors.append("request_not_observed")
        if not self.stored_bytes:
            self.errors.append("wire_not_observed")
        if self.models != {MODEL_ID}:
            self.errors.append("wrong_or_missing_model")
        if not self.saw_done:
            self.errors.append("missing_done")
        if self.finish_reason != "length":
            self.errors.append("invalid_finish_reason")
        if self.usage is None:
            self.errors.append("missing_usage")
        else:
            if (
                self.usage["prompt_tokens"] != len(self.expected["prompt"])
                or self.usage["completion_tokens"] != self.expected["max_tokens"]
            ):
                self.errors.append("usage_length_mismatch")
            if (
                self.usage["total_tokens"]
                != self.usage["prompt_tokens"] + self.usage["completion_tokens"]
            ):
                self.errors.append("usage_total_mismatch")
        cached = None
        if isinstance(self.cache_details, dict) and all(
            type(self.cache_details.get(k)) is int and self.cache_details[k] >= 0
            for k in ("device", "host")
        ):
            storage = self.cache_details.get("storage")
            if storage is None or (type(storage) is int and storage >= 0):
                cached = (
                    self.cache_details["device"]
                    + self.cache_details["host"]
                    + (storage or 0)
                )
        if cached is None:
            self.errors.append("cache_observation_missing")
        elif cached and not self.warmup:
            self.errors.append("cold_request_cache_hit")
        if output.success is not True:
            self.errors.append("upstream_request_failed")
        valid = not self.errors
        output.success = valid
        if valid:
            assert self.usage is not None
            output.output_len = self.usage["completion_tokens"]
            output.prompt_len = self.usage["prompt_tokens"]
        else:
            output.output_len = 0
            output.prompt_len = 0
        output.cached_tokens = cached
        if not valid:
            output.error = (
                "capture: "
                + ",".join(dict.fromkeys(self.errors))
                + "; "
                + (output.error or "")
            )
        document = {
            "schema": "qwen38fn.upstream-capture.v1",
            "request_index": self.index,
            "warmup": self.warmup,
            "request": self.request_value,
            "request_sha256": input_hash(self.request_value)
            if self.request_value
            else None,
            "input_ids_sha256": input_hash(self.expected["prompt"]),
            "observed_usage": self.usage,
            "finish_reason": self.finish_reason,
            "cache_details": self.cache_details,
            "cached_tokens": cached,
            "model_ids": sorted(self.models),
            "saw_done": self.saw_done,
            "wire_file": self.path.name,
            "wire_sha256": self.wire_hash.hexdigest(),
            "wire_bytes": self.stored_bytes,
            "wire_truncated": self.bytes_received > MAX_WIRE_BYTES,
            "valid": valid,
            "error": output.error or "",
        }
        with self.path.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return document


def instrument_source(text):
    edits = [
        (
            "async def async_request_openai_completions(",
            "async def _qwen38_captured_oai(",
        ),
        (
            "        headers = get_request_headers()",
            "        _qwen38_capture_request(payload)\n        headers = get_request_headers()",
        ),
        (
            "                    async for chunk_bytes in response.content:",
            "                    async for chunk_bytes in response.content:\n                        _qwen38_capture_chunk(chunk_bytes)",
        ),
        (
            '                        if chunk == "[DONE]":\n                            pass',
            '                        if chunk == "[DONE]":\n                            _qwen38_capture_done()',
        ),
        (
            "                            data = json.loads(chunk)",
            '                            data = json.loads(chunk)\n                            _qwen38_capture_event(data)\n                            if not data.get("choices"):\n                                continue',
        ),
    ]
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError("upstream capture anchor drift")
        text = text.replace(before, after, 1)
    compile(text, "pinned_upstream_capture", "exec")
    return text


def install_capture(upstream, payloads, directory, expected_module_sha256):
    path = Path(upstream.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_module_sha256:
        raise ValueError("upstream benchmark source differs from frozen runtime")
    if hasattr(upstream, "_qwen38_captured_oai"):
        raise ValueError("capture adapter already installed; use one process per round")
    original = inspect.getsource(upstream.async_request_openai_completions)
    instrumented = instrument_source(original)
    current = contextvars.ContextVar("qwen38_benchmark_capture")
    upstream._qwen38_capture_request = lambda value: current.get().request(value)
    upstream._qwen38_capture_chunk = lambda value: current.get().chunk(value)
    upstream._qwen38_capture_event = lambda value: current.get().event(value)
    upstream._qwen38_capture_done = lambda: current.get().done()
    # This executes reviewed upstream CLIENT source, never model-generated code.
    exec(
        compile(instrumented, str(path) + ":client-capture", "exec"), upstream.__dict__
    )
    identities = {
        input_hash(payload["prompt"]): index for index, payload in enumerate(payloads)
    }
    if len(identities) != len(payloads):
        raise ValueError("frozen batch has duplicate token arrays")
    records = []
    sequence = 0

    async def wrapped(request_func_input, pbar=None):
        nonlocal sequence
        index = identities.get(input_hash(request_func_input.prompt))
        if index is None or request_func_input.prompt != payloads[index]["prompt"]:
            raise ValueError("upstream attempted an unfrozen token prompt")
        expected = json.loads(json.dumps(payloads[index]))
        warmup = request_func_input.output_len == 32
        if warmup:
            expected["max_tokens"] = 32
        elif request_func_input.output_len != expected["max_tokens"]:
            raise ValueError("upstream output budget drift")
        record = CaptureRecord(expected, index, warmup, directory, sequence)
        sequence += 1
        token = current.set(record)
        try:
            output = await upstream._qwen38_captured_oai(request_func_input, pbar)
        except Exception as error:
            output = upstream.RequestFuncOutput.init_new(request_func_input)
            output.success = False
            output.error = f"capture wrapper: {type(error).__name__}: {error}"
        finally:
            current.reset(token)
        records.append(record.finalize(output))
        return output

    upstream.ASYNC_REQUEST_FUNCS["sglang-oai"] = wrapped
    return records, hashlib.sha256(instrumented.encode()).hexdigest()
