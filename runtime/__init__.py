"""Runtime profile contract for the single-GB10 Qwen3.8-Flash-Next launch.

A profile is a strict, versioned JSON document. Every field is explicitly
typed, range-checked and enum-checked; unknown fields, duplicate JSON keys,
nonfinite numbers and booleans used as integers are all rejected. There is
deliberately no caller-supplied tensor-parallel size, no arbitrary extra
flag list and no caller-supplied draft-model path: the launch contract is
the code in ``runtime/entrypoint.py``, nothing else.

Status: NOT QUALIFIED. Nothing here has been validated against the real
GB10 image; this module defines the contract only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

PROFILE_SCHEMA = 1

#: modes: "ar" = plain autoregressive serving, "nextn" = native integrated
#: MTP speculative decoding (no external drafter, no unquantized draft).
MODES = ("ar", "nextn")

CONTEXT_LENGTH_REQUIRED = 32768
CONTEXT_LENGTH_ALLOWED = (32768, 262144)

MAX_RUNNING_REQUESTS_ALLOWED = (1, 2, 4, 8)
MAX_RUNNING_REQUESTS_DEFAULT = 1

CHUNKED_PREFILL_ALLOWED = (1024, 2048, 4096)
CHUNKED_PREFILL_DEFAULT = 1024

MAX_MAMBA_CACHE_SIZE_DEFAULT = 8
MAX_MAMBA_CACHE_SIZE_MIN = 4
MAX_MAMBA_CACHE_SIZE_MAX = 64

MEM_FRACTION_STATIC_DEFAULT = 0.80
MEM_FRACTION_STATIC_MIN = 0.70
MEM_FRACTION_STATIC_MAX = 0.88

KV_CACHE_DTYPES = ("bf16", "fp8_e4m3")
KV_CACHE_DTYPE_DEFAULT = "bf16"

SPECULATIVE_STEPS_MIN = 1
SPECULATIVE_STEPS_MAX = 3

VISION_BACKENDS = ("triton_attn", "flashinfer_cudnn", "fa4")

PLE_RSS_GIB_ALLOWED = (4, 8, 12)
PLE_RSS_GIB_DEFAULT = 8

MM_PROCESSOR_WORKER_NUM_ALLOWED = (1, 2)
MM_PROCESSOR_WORKER_NUM_DEFAULT = 1

_SPECULATIVE_FIELDS = frozenset({"steps"})
_VISION_FIELDS = frozenset({"backend", "cuda_graph"})


class ProfileError(ValueError):
    """Raised for any profile that does not satisfy the versioned contract."""


def _reject_constant(token: str) -> Any:
    raise ProfileError("nonfinite JSON number %r is not allowed" % (token,))


def _object_pairs(pairs: list) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ProfileError("duplicate JSON field %r" % (key,))
        seen[key] = value
    return seen


def _fail(reason: str) -> None:
    raise ProfileError(reason)


def _require_int(raw: Dict[str, Any], key: str) -> int:
    if key not in raw:
        _fail("missing required integer field %r" % (key,))
    value = raw[key]
    if type(value) is not int:  # bool is not int here: bool-as-int rejected
        _fail("field %r must be an integer, got %r" % (key, value))
    return value


def _optional_int(
    raw: Dict[str, Any],
    key: str,
    *,
    allowed: Tuple[int, ...] = (),
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
    default: Optional[int] = None,
) -> int:
    if key not in raw:
        if default is None:
            _fail("missing required integer field %r" % (key,))
        return int(default)
    value = raw[key]
    if type(value) is not int:
        _fail("field %r must be an integer, got %r" % (key, value))
    if allowed and value not in allowed:
        _fail(
            "field %r must be one of %s, got %r"
            % (key, list(allowed), value)
        )
    if minimum is not None and value < minimum:
        _fail("field %r must be >= %d, got %r" % (key, minimum, value))
    if maximum is not None and value > maximum:
        _fail("field %r must be <= %d, got %r" % (key, maximum, value))
    return value


def _optional_float(
    raw: Dict[str, Any],
    key: str,
    *,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    if key not in raw:
        return default
    value = raw[key]
    if type(value) not in (int, float) or isinstance(value, bool):
        _fail("field %r must be a number, got %r" % (key, value))
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        _fail("field %r must be finite" % (key,))
    if not minimum <= value <= maximum:
        _fail(
            "field %r must be within [%s, %s], got %r"
            % (key, minimum, maximum, value)
        )
    return value


def _require_enum(raw: Dict[str, Any], key: str, allowed) -> str:
    if key not in raw:
        _fail("missing required field %r" % (key,))
    value = raw[key]
    if not isinstance(value, str) or value not in allowed:
        _fail("field %r must be one of %s, got %r" % (key, list(allowed), value))
    return value


def _optional_enum(raw: Dict[str, Any], key: str, allowed, default: str) -> str:
    if key not in raw:
        return default
    return _require_enum(raw, key, allowed)


def validate_payload(payload: str) -> Dict[str, Any]:
    """Parse and fully validate a profile JSON document.

    Returns the raw dict. Raises :class:`ProfileError` on any violation.
    """
    try:
        raw = json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_pairs,
        )
    except json.JSONDecodeError as exc:
        raise ProfileError("profile is not valid JSON: %s" % (exc,)) from exc
    if not isinstance(raw, dict):
        raise ProfileError("profile must be a JSON object")

    schema = _require_int(raw, "schema")
    if schema != PROFILE_SCHEMA:
        _fail("unsupported profile schema %r (expected %d)" % (schema, PROFILE_SCHEMA))

    mode = _require_enum(raw, "mode", MODES)

    context_length = _require_int(raw, "context_length")
    if context_length not in CONTEXT_LENGTH_ALLOWED:
        _fail(
            "field 'context_length' must be one of %s, got %r"
            % (list(CONTEXT_LENGTH_ALLOWED), context_length)
        )
    max_total_tokens = _require_int(raw, "max_total_tokens")
    if max_total_tokens != context_length:
        _fail(
            "field 'max_total_tokens' must equal 'context_length' (%d), got %r"
            % (context_length, max_total_tokens)
        )

    _optional_int(
        raw, "max_running_requests", allowed=MAX_RUNNING_REQUESTS_ALLOWED,
        default=MAX_RUNNING_REQUESTS_DEFAULT,
    )
    _optional_int(
        raw, "chunked_prefill_size", allowed=CHUNKED_PREFILL_ALLOWED,
        default=CHUNKED_PREFILL_DEFAULT,
    )
    _optional_int(
        raw, "max_mamba_cache_size",
        minimum=MAX_MAMBA_CACHE_SIZE_MIN, maximum=MAX_MAMBA_CACHE_SIZE_MAX,
        default=MAX_MAMBA_CACHE_SIZE_DEFAULT,
    )
    _optional_float(
        raw, "mem_fraction_static",
        minimum=MEM_FRACTION_STATIC_MIN, maximum=MEM_FRACTION_STATIC_MAX,
        default=MEM_FRACTION_STATIC_DEFAULT,
    )
    _optional_enum(raw, "kv_cache_dtype", KV_CACHE_DTYPES, KV_CACHE_DTYPE_DEFAULT)

    speculative = raw.get("speculative")
    if mode == "nextn":
        if not isinstance(speculative, dict):
            _fail("mode 'nextn' requires a 'speculative' object")
        if set(speculative) != _SPECULATIVE_FIELDS:
            _fail(
                "field 'speculative' must contain exactly %s, got %s"
                % (sorted(_SPECULATIVE_FIELDS), sorted(speculative))
            )
        _optional_int(
            speculative, "steps",
            minimum=SPECULATIVE_STEPS_MIN, maximum=SPECULATIVE_STEPS_MAX,
            default=None,
        )
    elif speculative is not None:
        _fail("field 'speculative' is only valid with mode 'nextn'")

    vision = raw.get("vision")
    if not isinstance(vision, dict):
        _fail("missing required object field 'vision'")
    if set(vision) != _VISION_FIELDS:
        _fail(
            "field 'vision' must contain exactly %s, got %s"
            % (sorted(_VISION_FIELDS), sorted(vision))
        )
    _require_enum(vision, "backend", VISION_BACKENDS)
    cuda_graph = vision["cuda_graph"]
    if not isinstance(cuda_graph, bool):
        _fail("field 'vision.cuda_graph' must be a boolean, got %r" % (cuda_graph,))

    _optional_int(
        raw, "ple_rss_gib", allowed=PLE_RSS_GIB_ALLOWED, default=PLE_RSS_GIB_DEFAULT
    )
    _optional_int(
        raw, "mm_processor_worker_num",
        allowed=MM_PROCESSOR_WORKER_NUM_ALLOWED, default=MM_PROCESSOR_WORKER_NUM_DEFAULT,
    )

    known = {
        "schema", "mode", "context_length", "max_total_tokens",
        "max_running_requests", "chunked_prefill_size", "max_mamba_cache_size",
        "mem_fraction_static", "kv_cache_dtype", "speculative", "vision",
        "ple_rss_gib", "mm_processor_worker_num",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        _fail("unknown profile field(s): %s" % (", ".join(repr(k) for k in unknown)))

    return raw


@dataclasses.dataclass(frozen=True)
class Profile:
    """Validated profile values."""

    schema: int
    mode: str
    context_length: int
    max_total_tokens: int
    max_running_requests: int
    chunked_prefill_size: int
    max_mamba_cache_size: int
    mem_fraction_static: float
    kv_cache_dtype: str
    speculative_steps: int  # 0 for mode "ar"
    vision_backend: str
    vision_cuda_graph: bool
    ple_rss_gib: int
    mm_processor_worker_num: int
    raw: Dict[str, Any] = dataclasses.field(repr=False, compare=False)

    def digest(self) -> str:
        """SHA-256 over the canonical JSON form of the raw profile."""
        canonical = json.dumps(
            self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self)["raw"], sort_keys=True, indent=2
        )


def profile_from_dict(raw: Dict[str, Any]) -> Profile:
    validated = validate_payload(json.dumps(raw))
    speculative = validated.get("speculative") or {}
    vision = validated["vision"]
    return Profile(
        schema=validated["schema"],
        mode=validated["mode"],
        context_length=validated["context_length"],
        max_total_tokens=validated["max_total_tokens"],
        max_running_requests=validated.get(
            "max_running_requests", MAX_RUNNING_REQUESTS_DEFAULT
        ),
        chunked_prefill_size=validated.get(
            "chunked_prefill_size", CHUNKED_PREFILL_DEFAULT
        ),
        max_mamba_cache_size=validated.get(
            "max_mamba_cache_size", MAX_MAMBA_CACHE_SIZE_DEFAULT
        ),
        mem_fraction_static=float(
            validated.get("mem_fraction_static", MEM_FRACTION_STATIC_DEFAULT)
        ),
        kv_cache_dtype=validated.get("kv_cache_dtype", KV_CACHE_DTYPE_DEFAULT),
        speculative_steps=int(speculative.get("steps", 0)),
        vision_backend=vision["backend"],
        vision_cuda_graph=bool(vision["cuda_graph"]),
        ple_rss_gib=validated.get("ple_rss_gib", PLE_RSS_GIB_DEFAULT),
        mm_processor_worker_num=validated.get(
            "mm_processor_worker_num", MM_PROCESSOR_WORKER_NUM_DEFAULT
        ),
        raw=validated,
    )


def load_profile(path: str) -> Profile:
    """Read, strictly parse and validate the profile at ``path``."""
    if not os.path.isfile(path):
        raise ProfileError("profile file not found: %s" % (path,))
    with open(path, "r", encoding="utf-8") as handle:
        payload = handle.read()
    return profile_from_json(payload)


def profile_from_json(payload: str) -> Profile:
    return profile_from_dict(validate_payload(payload))
