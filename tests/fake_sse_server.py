#!/usr/bin/env python3
"""Threaded in-process fake OpenAI-compatible server for client scaffold tests.

This is TEST INFRASTRUCTURE ONLY. Responses produced by this server are
synthetic test vectors; they must never be presented as real model output.

Scenarios are addressed by path: ``/s/<scenario>``. The handler is byte-level
scripted so tests can exercise fragmented SSE writes, writes that split
multibyte UTF-8 codepoints, partial EOF, CRLF framing, multiline ``data``
fields, role-only events, HTTP error statuses, non-stream JSON responses,
missing/invalid usage, and stall-then-close timeouts.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"


def sse_event(obj: Any) -> bytes:
    if isinstance(obj, str):
        return f"data: {obj}\n\n".encode("utf-8")
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def delta_evt(
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
    role: Optional[str] = None,
    finish: Optional[str] = None,
) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    choice: Dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish}
    return {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "model": MODEL_ID, "choices": [choice]}


def usage_evt(prompt: int = 12, completion: int = 6) -> Dict[str, Any]:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "model": MODEL_ID,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


class Scenario:
    """A scripted byte-level response."""

    def __init__(
        self,
        status: int = 200,
        chunks: Optional[List[bytes]] = None,
        body: Optional[bytes] = None,
        content_type: str = "text/event-stream",
        stall_s: float = 0.0,
        respond_nonstream: bool = False,
    ):
        self.status = status
        self.chunks = chunks or []
        self.body = body
        self.content_type = content_type
        self.stall_s = stall_s
        self.respond_nonstream = respond_nonstream


def _happy_stream(fragmented: bool, split_utf8: bool) -> Scenario:
    events: List[bytes] = []
    events.append(sse_event(delta_evt(role="assistant")))
    events.append(sse_event(delta_evt(content="")))
    if split_utf8:
        # "héllo 🌀 wörld" — multibyte codepoints split across events
        events.append(sse_event(delta_evt(content="héllo \U0001f300 wörld — 你好")))
    else:
        events.append(sse_event(delta_evt(content="Hello world")))
    events.append(sse_event(delta_evt(reasoning="thinking...")))
    events.append(sse_event(delta_evt(content="!", finish="stop")))
    events.append(sse_event(usage_evt(prompt=9, completion=7)))
    events.append(b"data: [DONE]\n\n")
    if fragmented:
        chunks: List[bytes] = []
        for ev in events:
            # split each event into 3-byte pieces (splits UTF-8 + SSE framing)
            chunks += [ev[i : i + 3] for i in range(0, len(ev), 3)]
        return Scenario(chunks=chunks)
    return Scenario(chunks=events)


SCENARIOS: Dict[str, Scenario] = {
    "happy": _happy_stream(fragmented=False, split_utf8=False),
    "fragmented-utf8": _happy_stream(fragmented=True, split_utf8=True),
    "role-only": Scenario(
        chunks=[
            sse_event(delta_evt(role="assistant")),
            sse_event({"id": "x", "object": "chat.completion.chunk", "model": MODEL_ID, "choices": []}),
            sse_event(delta_evt(finish="stop")),
            sse_event(usage_evt()),
            b"data: [DONE]\n\n",
        ]
    ),
    "partial-eof": Scenario(
        chunks=[
            sse_event(delta_evt(content="partial")),
            b"data: {\"id\": \"x\", \"choices\": [{\"delta\": {\"content\": \" trun",
        ]
    ),
    "invalid-usage-float": Scenario(
        chunks=[
            sse_event(delta_evt(content="hi", finish="stop")),
            sse_event({"usage": {"prompt_tokens": 1.5, "completion_tokens": 2, "total_tokens": 3}}),
            b"data: [DONE]\n\n",
        ]
    ),
    "invalid-usage-bool": Scenario(
        chunks=[
            sse_event(delta_evt(content="hi", finish="stop")),
            sse_event({"usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}}),
            b"data: [DONE]\n\n",
        ]
    ),
    "invalid-usage-missing": Scenario(
        chunks=[
            sse_event(delta_evt(content="hi", finish="stop")),
            sse_event({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
            b"data: [DONE]\n\n",
        ]
    ),
    "missing-usage": Scenario(
        chunks=[sse_event(delta_evt(content="hi", finish="stop")), b"data: [DONE]\n\n"]
    ),
    "finish-length": Scenario(
        chunks=[
            sse_event(delta_evt(content="truncated outp")),
            sse_event(delta_evt(finish="length")),
            sse_event(usage_evt(prompt=10, completion=4)),
            b"data: [DONE]\n\n",
        ]
    ),
    "reasoning-only": Scenario(
        chunks=[
            sse_event(delta_evt(reasoning="only reasoning here")),
            sse_event(delta_evt(finish="stop")),
            sse_event(usage_evt(prompt=5, completion=3)),
            b"data: [DONE]\n\n",
        ]
    ),
    "crlf-multiline": Scenario(
        chunks=[
            b"data: {\"choices\": [{\"delta\": {\"content\": \"a\"}}]}\r\n\r\n",
            b": keepalive comment\r\n\r\n",
            b"data: {\"choices\": [{\"delta\": {\"content\": \"b\"}}]}\r\ndata: {\"choices\": [{\"delta\": {\"content\": \"c\"}}]}\r\n\r\n",
            sse_event(delta_evt(finish="stop")),
            sse_event(usage_evt(prompt=3, completion=3)),
            b"data: [DONE]\n\n",
        ]
    ),
    "http-400": Scenario(status=400, body=json.dumps({"error": {"message": "bad request"}}).encode(), content_type="application/json"),
    "http-500": Scenario(status=500, body=b"upstream exploded", content_type="text/plain"),
    "stall-close": Scenario(chunks=[sse_event(delta_evt(content="stall")), sse_event(delta_evt(finish="stop"))], stall_s=3.0),
}


class FakeHandler(BaseHTTPRequestHandler):
    server_version = "FakeOpenAI/1.0"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    # -- routing --------------------------------------------------------------

    def _scenario_name(self) -> str:
        if self.path.startswith("/s/"):
            return self.path[3:].split("?")[0]
        return ""

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send(self, scenario: Scenario) -> None:
        if scenario.stall_s:
            time.sleep(scenario.stall_s)
        self.send_response(scenario.status)
        self.send_header("Content-Type", scenario.content_type)
        body = scenario.body
        if body is None:
            total = sum(len(c) for c in scenario.chunks)
            self.send_header("Content-Length", str(total))
            self.end_headers()
            for c in scenario.chunks:
                self.wfile.write(c)
                self.wfile.flush()
        else:
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models-wrong":
            body = json.dumps({"object": "list", "data": [{"id": "other/model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        name = self._scenario_name()
        body = self._read_json_body()
        if self.path == "/v1/chat/completions" and not name:
            if body.get("stream") is False:
                # non-stream JSON chat response
                msg = {"role": "assistant", "content": "nonstream reply"}
                if body.get("_reasoning"):
                    msg["reasoning_content"] = "hmm"
                payload = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
                }
                self._send(Scenario(body=json.dumps(payload).encode(), content_type="application/json"))
                return
            name = "happy"
        if self.path == "/v1/completions":
            usage = {"prompt_tokens": len(body.get("prompt") or []), "completion_tokens": 2, "total_tokens": len(body.get("prompt") or []) + 2}
            payload = {
                "id": "cmpl-fake",
                "object": "text_completion",
                "model": MODEL_ID,
                "choices": [{"index": 0, "text": "AB", "finish_reason": "stop"}],
                "usage": usage,
            }
            self._send(Scenario(body=json.dumps(payload).encode(), content_type="application/json"))
            return
        if self.path == "/flush_cache":
            self._send(Scenario(body=b"Cache flushed.\n", content_type="text/plain"))
            return
        scenario = SCENARIOS.get(name)
        if scenario is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._send(scenario)


class FakeServer:
    """Context manager running the fake server on an ephemeral port."""

    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def url(self, scenario: str) -> str:
        return f"{self.base}/s/{scenario}"

    def __enter__(self) -> "FakeServer":
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def run_fake_server() -> FakeServer:  # convenience for pytest fixtures
    return FakeServer()
