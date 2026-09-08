"""Pre-traffic request binding shared by text, visual and upstream producers."""

import hashlib
import json
from pathlib import Path

import compare


def input_hash(payload):
    """SHA256 of the complete canonical request value (media included)."""
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def bind_requests(manifest, lane, requests):
    """Validate all planned inputs before traffic; None is diagnostic-only."""
    if manifest is None:
        return {}
    try:
        fingerprint = compare.validate_manifest(manifest)
    except compare.Reject as error:
        raise ValueError(str(error)) from error
    if lane == "vision" and len(requests) != 43:
        raise ValueError("bound vision promotion requires the full 43-case envelope")
    actual = {case: input_hash(payload) for case, payload in requests.items()}
    if manifest["inputs"].get(lane) != actual:
        raise ValueError("actual request plan differs from frozen " + lane + " inputs")
    return {
        case: {
            "manifest_sha256": fingerprint,
            "lane": lane,
            "model_id": manifest["model_id"],
            "input_sha256": sha,
        }
        for case, sha in actual.items()
    }


def load_manifest(path):
    if path is None:
        return None

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate manifest key")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("nonfinite manifest constant")

    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("manifest exceeds byte limit")
    manifest = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    compare.validate_manifest(manifest)  # canonical hash rejects exponent overflow too
    return manifest
