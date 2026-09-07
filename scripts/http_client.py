#!/usr/bin/env python3
"""Shared stdlib-only OpenAI-compatible HTTP/SSE client for the qualification harness.

Contract (see benchmarking skill: openai-streaming-benchmark-contract):
- The base URL is ALWAYS explicit. There is no implicit live endpoint default.
- The model id is the exact frozen constant ``MODEL_ID``; no silent fallback or
  retry against any other model, ever.
- Request timing starts inside the request call, i.e. after the caller has
  acquired concurrency admission (use :class:`ConcurrencyGate`). The clock
  therefore never includes admission wait.
- TTFT is the first NONEMPTY generated ``delta.content`` OR
  ``delta.reasoning_content``/``delta.reasoning`` fragment. Role-only or
  metadata-only events never start the token clock.
- SSE chunk gaps are transport events, NOT inter-token latencies. This module
  deliberately exposes no "ITL"/token-rate-from-chunks metric; the only token
  rate is usage.completion_tokens / measured wall time.
- Failures are preserved: every failed or partial stream keeps its parsed raw
  events, HTTP status, error class and detail on the returned result. Nothing
  is silently discarded.
- ``POST /flush_cache`` exists and may return plaintext success. It may only be
  used by the owner-admitted idle campaign endpoint; :meth:`flush_cache`
  refuses unless ``admitted=True`` is passed explicitly.
- Usage objects must be nonnegative JSON integers. Booleans, fractional floats,
  missing fields and silent coercion are rejected (claim-bearing traffic).
"""
from __future__ import annotations

import argparse
import codecs
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
DONE_SENTINEL = "[DONE]"
REASONING_KEYS = ("reasoning_content", "reasoning")


class ClientError(Exception):
    """Base class for all client failures."""


class ConfigError(ClientError):
    """Misconfiguration (missing explicit base, owner admission, ...)."""


class OwnerAdmissionError(ConfigError):
    """A privileged operation was attempted without explicit owner admission."""


class TransportError(ClientError):
    """Network-level failure (DNS, connect, reset, timeout)."""


class HTTPStatusError(ClientError):
    """Non-2xx HTTP response. Body snippet preserved."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class StreamFormatError(ClientError):
    """Malformed SSE framing or JSON."""


class MissingDoneError(StreamFormatError):
    """Stream reached EOF without a [DONE] sentinel (partial/truncated)."""


class InvalidUsageError(StreamFormatError):
    """Usage object present but not nonnegative JSON integers."""


class MissingUsageError(StreamFormatError):
    """No usage object at all in a claim-bearing response."""


class ModelMismatchError(ClientError):
    """The endpoint does not serve exactly MODEL_ID."""


# ---------------------------------------------------------------------------
# SSE decoding
# ---------------------------------------------------------------------------


class SSEDecoder:
    """Incremental text-event decoder tolerant of fragmented bytes.

    Handles: arbitrary chunk splits (including inside multibyte UTF-8),
    CRLF or LF framing, ``:`` comment lines, multi-line ``data:`` fields
    (joined with "\n"), and blank-line event termination.
    """

    def __init__(self) -> None:
        self._buf = b""
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._lines: List[str] = []

    def feed(self, chunk: bytes) -> List[str]:
        """Feed raw bytes; return completed event data payloads (strings)."""
        self._buf += chunk
        events: List[str] = []
        while True:
            nl = self._find_newline()
            if nl is None:
                break
            raw_line = self._buf[: nl[0]]
            self._buf = self._buf[nl[1] :]
            line = self._decoder.decode(raw_line)
            self._lines.append(line)
            if line == "":
                # blank line terminates the event
                data_lines = [
                    ln[5:].lstrip(" ") if ln.startswith("data:") else None
                    for ln in self._lines
                ]
                self._lines = []
                if all(d is None for d in data_lines):
                    continue  # comment/keepalive only event
                payload = "\n".join(d for d in data_lines if d is not None)
                events.append(payload)
        return events

    def close(self) -> List[str]:
        """Flush any pending buffered event at EOF (returns trailing event)."""
        events: List[str] = []
        # an unterminated line may still sit in the byte buffer
        pending = b""
        if self._buf:
            pending = self._decoder.decode(self._buf, final=True)
            self._buf = b""
        tail = self._decoder.decode(b"", final=True)
        line: str = (pending or "") + tail
        if line:
            # a lone trailing CR (ambiguous at feed time) terminates the line
            self._lines.append(line[:-1] if line.endswith("\r") else line)
        if self._lines:
            data_lines = [
                ln[5:].lstrip(" ") if ln.startswith("data:") else None
                for ln in self._lines
            ]
            self._lines = []
            if not all(d is None for d in data_lines):
                events.append("\n".join(d for d in data_lines if d is not None))
        return events

    def _find_newline(self) -> Optional[Tuple[int, int]]:
        """Find the next line terminator (SSE: LF, CRLF or bare CR).

        A CR that is the last buffered byte is ambiguous — the next byte
        arriving in a later TCP chunk may be LF — so it is held until more
        data arrives or close() flushes the decoder.
        """
        i_lf = self._buf.find(b"\n")
        i_cr = self._buf.find(b"\r")
        candidates = []
        if i_lf != -1:
            candidates.append((i_lf, i_lf + 1))
        if i_cr != -1:
            if i_cr + 1 < len(self._buf):
                # resolved: CRLF consumes both bytes, bare CR consumes one
                end = i_cr + 2 if self._buf[i_cr + 1 : i_cr + 2] == b"\n" else i_cr + 1
                candidates.append((i_cr, end))
            # else: trailing CR held — do not emit a candidate
        if not candidates:
            return None
        return min(candidates)


# ---------------------------------------------------------------------------
# Usage validation (fail closed, no coercion)
# ---------------------------------------------------------------------------


def validate_usage_object(usage: Any) -> Dict[str, int]:
    """Validate a usage object; raise InvalidUsageError on any violation."""
    if not isinstance(usage, dict):
        raise InvalidUsageError(f"usage is not an object: {usage!r}")
    out: Dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key not in usage:
            raise InvalidUsageError(f"usage missing field {key!r}")
        val = usage[key]
        if isinstance(val, bool) or not isinstance(val, int):
            raise InvalidUsageError(f"usage.{key} is not a JSON integer: {val!r}")
        if val < 0:
            raise InvalidUsageError(f"usage.{key} is negative: {val!r}")
        out[key] = val
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    """Complete outcome of one streaming request, including failures.

    ``error`` is None on success, else one of:
    http_status / transport / timeout / stream_format /
    missing_done / invalid_usage / missing_usage.
    Failed and partial streams keep their raw parsed events in ``raw_events``.
    """

    content: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model_reported: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    error_detail: str = ""
    ttft_s: Optional[float] = None
    first_fragment_kind: Optional[str] = None  # "content" | "reasoning"
    wall_s: float = 0.0
    sse_event_count: int = 0  # transport events, NOT tokens
    saw_done: bool = False
    raw_events: List[Any] = field(default_factory=list)
    raw_error_body: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def e2e_output_tok_per_s(self) -> Optional[float]:
        """usage.completion_tokens / actual wall time. None if no valid usage."""
        if not self.usage:
            return None
        if self.wall_s <= 0:
            return None
        return self.usage["completion_tokens"] / self.wall_s


@dataclass
class CompletionResult:
    """Outcome of a non-streaming request (chat or /v1/completions)."""

    text: str = ""
    reasoning: str = ""
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model_reported: Optional[str] = None
    http_status: Optional[int] = None
    error: Optional[str] = None
    error_detail: str = ""
    wall_s: float = 0.0
    raw_body: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def thinking_request_fields(enabled: bool, effort: Optional[str] = "low") -> Dict[str, Any]:
    """Request fields controlling native thinking.

    Default harness policy: throughput diagnostics run with
    ``enable_thinking=False``; ``--thinking`` style runs enable native thinking
    at low effort. Only documented request fields are ever sent.
    """
    fields: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}
    if enabled and effort:
        fields["reasoning_effort"] = effort
    return fields


def build_chat_payload(
    messages: List[Dict[str, Any]],
    *,
    model: str = MODEL_ID,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    stream: bool = True,
    include_usage: bool = True,
    thinking: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": bool(stream),
    }
    if stream and include_usage:
        payload["stream_options"] = {"include_usage": True}
    if thinking is None:
        # harness default: explicit enable_thinking=False for throughput
        # diagnostics (never rely on server-side template defaults)
        thinking = thinking_request_fields(False)
    if thinking:
        payload.update(thinking)
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# Policy helpers shared by benchmark drivers
# ---------------------------------------------------------------------------

_ERROR_CLASS_PREFIXES = (
    "http_status_",
    "transport",
    "timeout",
    "stream_format",
    "missing_done",
    "invalid_usage",
    "missing_usage",
)


def row_validity(
    result: StreamResult,
    *,
    require_finish: Optional[str] = "stop",
    require_content: bool = True,
    min_content_chars: int = 1,
) -> Tuple[bool, Optional[str]]:
    """Fail-closed validity of a measured streaming row.

    Returns (valid, reason). A length-limited stream is NEVER a silent pass:
    it is reported as ``finish_length`` and the caller must fail the run.
    """
    if result.error is not None:
        for prefix in _ERROR_CLASS_PREFIXES:
            if result.error.startswith(prefix):
                return False, result.error
        return False, f"error_{result.error}"
    if require_finish is not None and result.finish_reason != require_finish:
        return False, f"finish_{result.finish_reason or 'none'}"
    if result.usage is None:
        return False, "missing_usage"
    if require_content:
        final = (result.content or "").strip()
        if not final:
            return False, "empty_final_content"
        if len(final) < min_content_chars:
            return False, "content_too_short"
    return True, None


# ---------------------------------------------------------------------------
# Concurrency admission
# ---------------------------------------------------------------------------


class ConcurrencyGate:
    """Admission gate. Request timing starts only after ``slot()`` is held."""

    def __init__(self, concurrency: int = 1):
        if concurrency < 1:
            raise ConfigError("concurrency must be >= 1")
        self._sem = threading.BoundedSemaphore(concurrency)

    from contextlib import contextmanager

    @staticmethod
    @contextmanager
    def _slot_ctx(sem: threading.BoundedSemaphore):
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def slot(self):
        return ConcurrencyGate._slot_ctx(self._sem)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OpenAICompatClient:
    """Robust stdlib client. Base URL is mandatory and explicit."""

    def __init__(
        self,
        base: Optional[str],
        *,
        model: str = MODEL_ID,
        default_timeout_s: float = 600.0,
        verify_tls: bool = True,
    ):
        if not base or not str(base).strip():
            raise ConfigError(
                "an explicit --base endpoint URL is required; "
                "this harness never targets an implicit live endpoint"
            )
        self.base = str(base).rstrip("/")
        if not self.base.startswith(("http://", "https://")):
            raise ConfigError(f"base must be an http(s) URL, got {base!r}")
        self.model = model
        self.default_timeout_s = float(default_timeout_s)
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(
                context=None if verify_tls else _NoVerifyContext.create()
            )
        )

    # -- low level ----------------------------------------------------------

    def _open(self, method: str, path: str, body: Optional[Dict[str, Any]], timeout: float):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "User-Agent": "qual-harness-http-client/1.0",
            },
        )
        return self.opener.open(req, timeout=timeout)

    @staticmethod
    def _classify_transport(exc: Exception) -> TransportError:
        import socket

        if isinstance(exc, (socket.timeout, TimeoutError)):
            return TransportError(f"timeout: {exc}")
        return TransportError(f"transport: {exc}")

    # -- model identity -----------------------------------------------------

    def list_models(self, timeout: float = 30.0) -> List[str]:
        """GET /v1/models. Only standard ``data[].id`` fields are read."""
        try:
            with self._open("GET", "/v1/models", None, timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise HTTPStatusError(exc.code, exc.read().decode("utf-8", "replace")) from exc
        except Exception as exc:  # URLError, socket, json
            raise self._classify_transport(exc) from exc
        ids = [item.get("id") for item in body.get("data", []) if isinstance(item, dict)]
        return [i for i in ids if isinstance(i, str)]

    def verify_model(self, timeout: float = 30.0) -> None:
        """Fail closed unless /v1/models serves exactly the frozen model id."""
        ids = self.list_models(timeout=timeout)
        if self.model not in ids:
            raise ModelMismatchError(
                f"model {self.model!r} not served; endpoint offers {ids[:10]}"
            )

    # -- streaming chat -----------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        capture_raw: bool = True,
    ) -> StreamResult:
        """POST /v1/chat/completions with stream=true.

        The clock starts immediately before the HTTP request is sent: callers
        must already hold concurrency admission (ConcurrencyGate.slot()).
        """
        payload = build_chat_payload(
            messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            thinking=thinking,
            extra=extra,
        )
        return self._stream_request("/v1/chat/completions", payload, timeout or self.default_timeout_s, capture_raw)

    def _stream_request(self, path: str, payload: Dict[str, Any], timeout: float, capture_raw: bool) -> StreamResult:
        res = StreamResult()
        decoder = SSEDecoder()
        t0 = time.perf_counter()
        try:
            resp = self._open("POST", path, payload, timeout)
        except urllib.error.HTTPError as exc:
            res.wall_s = time.perf_counter() - t0
            body = exc.read().decode("utf-8", "replace")
            res.error = f"http_status_{exc.code}"
            res.http_status = exc.code
            res.raw_error_body = body[:5000]
            return res
        except Exception as exc:
            res.wall_s = time.perf_counter() - t0
            terr = self._classify_transport(exc)
            res.error = "timeout" if "timeout" in str(terr).lower() else "transport"
            res.error_detail = str(terr)
            return res

        res.http_status = resp.status

        def _fail(err: str, detail: str = "") -> StreamResult:
            res.wall_s = time.perf_counter() - t0
            res.error = err
            res.error_detail = detail
            try:
                resp.close()
            except Exception:
                pass
            return res

        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                for payload_str in decoder.feed(chunk):
                    self._consume_event(payload_str, res, t0, capture_raw)
                    if res.saw_done:
                        break
                if res.saw_done:
                    break
            if not res.saw_done:
                for payload_str in decoder.close():
                    self._consume_event(payload_str, res, t0, capture_raw)
        except Exception as exc:
            return _fail("stream_format", f"SSE read/decode failed: {exc}")

        res.wall_s = time.perf_counter() - t0
        if not res.saw_done:
            return _fail("missing_done", "stream ended at EOF without [DONE]; partial content preserved")
        if res.error == "invalid_usage":
            return _fail("invalid_usage", res.error_detail)
        if res.usage is None:
            return _fail("missing_usage", "stream completed but no final usage event was received")
        return res

    def _consume_event(self, payload_str: str, res: StreamResult, t0: float, capture_raw: bool) -> None:
        if payload_str == DONE_SENTINEL:
            res.saw_done = True
            return
        # An event may legally carry several `data:` lines, each its own JSON
        # object; SGLang emits usage this way. Dispatch each line separately.
        # (Raw newlines cannot occur inside a JSON string — they are escaped —
        # so a newline here always means multiple data lines.)
        if "\n" in payload_str:
            for sub in payload_str.split("\n"):
                if sub:
                    self._consume_event(sub, res, t0, capture_raw)
            return
        try:
            evt = json.loads(payload_str)
        except json.JSONDecodeError as exc:
            # preserve the unparseable (possibly truncated) payload as evidence
            res.raw_events.append({"_unparseable": payload_str})
            if res.error is None:
                res.error = "stream_format"
                res.error_detail = f"unparseable SSE JSON: {exc}"
            return
        if not isinstance(evt, dict):
            return
        res.sse_event_count += 1
        if capture_raw:
            res.raw_events.append(evt)
        if isinstance(evt.get("model"), str):
            res.model_reported = evt["model"]
        usage = evt.get("usage")
        if usage is not None:
            try:
                res.usage = validate_usage_object(usage)
            except InvalidUsageError as exc:
                res.error = "invalid_usage"
                res.error_detail = str(exc)
        choices = evt.get("choices") or []
        if not choices:
            return
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason"):
            res.finish_reason = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}
        now = time.perf_counter()
        content_piece = delta.get("content")
        if isinstance(content_piece, str) and content_piece:
            if res.ttft_s is None:
                res.ttft_s = now - t0
                res.first_fragment_kind = "content"
            res.content += content_piece
        for key in REASONING_KEYS:
            piece = delta.get(key)
            if isinstance(piece, str) and piece:
                if res.ttft_s is None:
                    res.ttft_s = now - t0
                    res.first_fragment_kind = "reasoning"
                res.reasoning += piece
                break

    # -- non-streaming ------------------------------------------------------

    def chat_nonstream(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        payload = build_chat_payload(
            messages,
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            thinking=thinking,
            extra=extra,
        )
        return self._json_request("/v1/chat/completions", payload, timeout or self.default_timeout_s, kind="chat")

    def completions_tokens(
        self,
        prompt_token_ids: List[int],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CompletionResult:
        """POST /v1/completions with a list[int] prompt (exact-token NIAH)."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": [int(t) for t in prompt_token_ids],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if extra:
            payload.update(extra)
        return self._json_request("/v1/completions", payload, timeout or self.default_timeout_s, kind="completions")

    def _json_request(self, path: str, payload: Dict[str, Any], timeout: float, kind: str) -> CompletionResult:
        res = CompletionResult()
        t0 = time.perf_counter()
        try:
            resp = self._open("POST", path, payload, timeout)
        except urllib.error.HTTPError as exc:
            res.wall_s = time.perf_counter() - t0
            res.error = f"http_status_{exc.code}"
            res.http_status = exc.code
            res.raw_body = exc.read().decode("utf-8", "replace")[:5000]
            return res
        except Exception as exc:
            res.wall_s = time.perf_counter() - t0
            terr = self._classify_transport(exc)
            res.error = "timeout" if "timeout" in str(terr).lower() else "transport"
            res.error_detail = str(terr)
            return res
        res.http_status = resp.status
        try:
            body = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            res.wall_s = time.perf_counter() - t0
            res.error = "transport"
            res.error_detail = f"body read failed: {exc}"
            return res
        res.wall_s = time.perf_counter() - t0
        res.raw_body = body[:20000]
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            res.error = "stream_format"
            res.error_detail = f"response is not JSON: {exc}"
            return res
        choices = data.get("choices") or []
        if not choices:
            res.error = "stream_format"
            res.error_detail = "response has no choices"
            return res
        choice = choices[0]
        res.finish_reason = choice.get("finish_reason")
        if kind == "chat":
            msg = choice.get("message") or {}
            res.text = msg.get("content") or ""
            res.reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        else:
            res.text = choice.get("text") or ""
        if data.get("usage") is not None:
            try:
                res.usage = validate_usage_object(data["usage"])
            except InvalidUsageError as exc:
                res.error = "invalid_usage"
                res.error_detail = str(exc)
        else:
            res.error = "missing_usage"
            res.error_detail = "non-stream response carried no usage object"
        if isinstance(data.get("model"), str):
            res.model_reported = data["model"]
        return res

    # -- privileged ----------------------------------------------------------

    def flush_cache(self, *, admitted: bool, timeout: float = 30.0) -> Tuple[int, str]:
        """POST /flush_cache. Refuses without explicit owner admission.

        The endpoint may answer plaintext success; any 200 body is accepted.
        """
        if not admitted:
            raise OwnerAdmissionError(
                "flush_cache was called without explicit owner admission; "
                "only the owner-admitted idle campaign endpoint may be flushed"
            )
        try:
            resp = self._open("POST", "/flush_cache", {}, timeout)
            return resp.status, resp.read().decode("utf-8", "replace")[:2000]
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")[:2000]


class _NoVerifyContext:
    """Lazy SSL context factory disabling cert verification (only if opted in)."""

    @staticmethod
    def create():
        import ssl

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


# ---------------------------------------------------------------------------
# Minimal smoke CLI (library first; useful for owner-admitted checks)
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Shared qualification HTTP/SSE client (smoke CLI)")
    p.add_argument("--base", required=True, help="explicit endpoint base URL (mandatory)")
    p.add_argument("--verify-model", action="store_true", help="GET /v1/models and require the exact frozen model id")
    p.add_argument("--flush-cache", action="store_true", help="POST /flush_cache (requires owner admission env)")
    p.add_argument(
        "--smoke-chat",
        type=int,
        default=0,
        metavar="MAX_TOKENS",
        help="one tiny streaming chat request (diagnostic only)",
    )
    args = p.parse_args(argv)

    client = OpenAICompatClient(args.base)
    if args.verify_model:
        client.verify_model()
        print(f"model-identity OK: {MODEL_ID}")
    if args.flush_cache:
        admitted = os.environ.get("QUAL_HARNESS_OWNER_ADMITTED") == "1"
        status, body = client.flush_cache(admitted=admitted)
        print(f"flush_cache status={status} body={body[:200]!r}")
        if status != 200:
            return 1
    if args.smoke_chat:
        gate = ConcurrencyGate(1)
        with gate.slot():  # admission precedes timing
            res = client.chat_stream(
                [{"role": "user", "content": "Reply with the single word: ready."}],
                max_tokens=args.smoke_chat,
                temperature=0.0,
                top_p=1.0,
            )
        valid, reason = row_validity(res)
        print(json.dumps({
            "ok": res.ok, "error": res.error, "finish_reason": res.finish_reason,
            "valid": valid, "reason": reason, "ttft_s": res.ttft_s, "wall_s": res.wall_s,
            "usage": res.usage, "rate": res.e2e_output_tok_per_s,
        }, indent=2))
        return 0 if valid else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
