"""Non-streaming raw evidence is retained, not silently sliced."""

import io
import json
from scripts import http_client as h


def test_nonstream_response_over_20k_is_preserved(monkeypatch):
    text = "test-only " * 3000
    raw = json.dumps(
        {
            "model": h.MODEL_ID,
            "choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4096,
                "total_tokens": 4106,
            },
        }
    )

    class Response(io.BytesIO):
        status = 200

    client = h.OpenAICompatClient("http://unit.test")
    monkeypatch.setattr(client, "_open", lambda *a, **k: Response(raw.encode()))
    result = client.completions_tokens([1, 2, 3], max_tokens=4096)
    assert result.error is None
    assert result.raw_body == raw
