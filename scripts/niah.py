#!/usr/bin/env python3
"""Exact-token Needle-In-A-Haystack for the qualification harness.

Contract:
- Native context WINDOW = 262,144 tokens; generous RESERVE = 4,096 output
  tokens. The largest ("full window") prompt therefore uses
  WINDOW - RESERVE = 258,048 prompt slots.
- Nine cases: single-key needles at 8192@50%, 32768@50%, 131072@50% and
  258048@{5,25,50,75,95}% depth, plus one full-window ordered multi-key case
  with needles at 33% and 66% that must be answered in order.
- The SAME actual tokenizer and chat template render the traffic; prompts are
  POSTed to /v1/completions as list[int] token ids (exact-token, no crop, no
  silent truncation, no rope mutation). If the template-slot encoding does not
  match, this fails loudly — the encoder is never silently switched.
- Full input token arrays, their sha256, needle offsets and codes are frozen
  into a manifest BEFORE any traffic.
- Sampling: temperature 0, top_p 1, max_tokens 4096, timeout 43,200 s.
- Row validity: API usage.prompt_tokens == constructed length, finish_reason
  == stop, and the post-thinking final text contains ALL codes in order.
  Incomplete thinking is not an answer.
- Transport/HTTP exceptions are INFRA failures, never needle misses; they are
  recorded per row and fail the run. All rows persist, or the summary is
  clearly marked partial with a nonzero exit.
- This helper never starts, stops or kills any server or client process;
  cancellation/draining is the parent's responsibility.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from http_client import CompletionResult, OpenAICompatClient  # noqa: E402

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-NVFP4"
WINDOW = 262_144
RESERVE = 4_096
MAX_PROMPT = WINDOW - RESERVE  # 258,048 full-window prompt slots
MAX_TOKENS = 4_096
TEMPERATURE = 0.0
TOP_P = 1.0
TIMEOUT_S = 43_200.0

SYSTEM_PROMPT = (
    "You are a precise retrieval assistant. The user will hide one or more "
    "pass codes inside a long document. Answer with ONLY the requested pass "
    "codes, in the order they appear in the document, separated by single "
    "spaces. Do not add any other words."
)

NEEDLE_TEMPLATE = "One of the special pass codes is {code}."
QUERY_TEMPLATE = (
    "The document above contains {n} special pass code sentence(s). "
    "Return the pass code(s) in the order they appear, separated by single spaces. "
    "Answer with the code(s) only."
)

# frozen code pool: distinctive, unlikely to appear in filler prose
CODE_POOL: Tuple[str, ...] = (
    "ZEPHYR-4821",
    "QUARTZ-9173",
    "MIRAGE-3058",
    "LUMEN-7742",
    "COBALT-6519",
    "HOLLOW-2384",
    "SPRUCE-5067",
    "VELVET-8291",
    "GRANITE-3476",
    "ORCHID-9158",
    "FERRITE-6823",
    "NIMBUS-1509",
)

FILLER_SENTENCE = (
    "The old archive lists shipments of tea, paper, lantern oil and tools "
    "across the district warehouses for each season of the year. "
)


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------


class TokenizerProtocol(Protocol):
    """Minimal tokenizer/template surface used here (actual model tokenizer)."""

    name: str

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: str = "low",
    ) -> Any: ...


class FakeTokenizer:
    """Deterministic word-level tokenizer for CPU unit tests (TEST ONLY).

    Construction is permitted in CPU tests. The public run_cases boundary
    rejects this class before any client call; it is not an inference source.
    """

    def __init__(self, name: str = "fake-wordlevel"):
        self.name = f"test-only:{name}"
        self._vocab: Dict[str, int] = {}
        self._next = 1

    def _tok(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next
            self._next += 1
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not isinstance(text, str):
            raise ValueError("text input must be a string")
        return [self._tok(w) for w in text.split(" ") if w != ""]

    def decode(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        enable_thinking: bool = True,
        reasoning_effort: str = "low",
    ):
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        return self.encode(rendered) if tokenize else rendered


def load_real_tokenizer(tokenizer_dir: str) -> TokenizerProtocol:
    """Load the actual model tokenizer (requires the model dir; parent runs it)."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - parent env has transformers
        raise RuntimeError(f"transformers unavailable: {exc}") from exc
    root = Path(tokenizer_dir).resolve(strict=True)
    sources = json.loads(
        (Path(__file__).resolve().parents[1] / "locks/sources.json").read_text()
    )["model"]
    if sources["id"] != MODEL_ID:
        raise ValueError("tokenizer source lock names another model")
    assets = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
    }
    observed = {}
    for item in sources["files"]:
        if item["path"] not in assets:
            continue
        path = root / item["path"]
        if path.is_symlink():
            raise ValueError("symlink tokenizer asset")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        match = (
            sha
            if item.get("sha256")
            else hashlib.sha1(
                b"blob " + str(len(data)).encode() + b"\0" + data
            ).hexdigest()
        )
        if len(data) != item["size"] or match != (
            item.get("sha256") or item["git_blob"]
        ):
            raise ValueError(
                "tokenizer asset differs from pinned checkpoint: " + item["path"]
            )
        observed[item["path"]] = sha
    if (
        not {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        <= observed.keys()
    ):
        raise ValueError("tokenizer lock is incomplete")
    tok = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=False, local_files_only=True
    )
    tok.name = f"real:{root}"
    tok._qwen38_tokenizer_assets = observed
    tok._qwen38_model_sha = sources["sha"]
    return tok  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Needle:
    code: str
    text: str
    token_ids: List[int]
    start_offset: int  # first token index of the needle inside the prompt


@dataclasses.dataclass(frozen=True)
class NiahCase:
    case_id: str
    n_tokens: int  # constructed prompt token count
    depths: Tuple[float, ...]
    needles: Tuple[Needle, ...]
    prompt_ids: Tuple[int, ...]
    rendered_text_sha256: str
    prompt_sha256: str
    response_starts_in_thinking: bool = False


def _build_filler(
    tokenizer: TokenizerProtocol, sentence: str, n_tokens: int
) -> List[int]:
    """Pre-tokenize one filler sentence, then tile it to >= n_tokens tokens."""
    one = tokenizer.encode(sentence, add_special_tokens=False)
    if not one:
        raise RuntimeError("filler sentence encoded to zero tokens")
    reps = (n_tokens // len(one)) + 2
    return (one * reps)[:n_tokens]


def _encode_with_positions(
    tokenizer: TokenizerProtocol,
    system: str,
    needle_specs: Sequence[Tuple[float, List[int]]],
    query: str,
    n_target: int,
) -> Tuple[List[int], List[int], str]:
    """Splice exact IDs between the real template's encoded prefix/suffix.

    The wire format is token IDs, not a decode/re-encode estimate. A NUL
    placeholder is used only to locate the insertion point and is removed
    before any request; both tokenizer and template remain unchanged.
    """
    slot = "\x00"
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": " " + slot + " \n\n" + query},
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="low",
    )
    if not isinstance(rendered, str):
        raise RuntimeError("tokenize=False did not produce template text")
    template_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    slot_ids = list(tokenizer.encode(slot, add_special_tokens=False))
    if not slot_ids:
        raise RuntimeError("template insertion slot encoded empty")
    starts = [
        i
        for i in range(len(template_ids) - len(slot_ids) + 1)
        if template_ids[i : i + len(slot_ids)] == slot_ids
    ]
    if len(starts) != 1:
        raise RuntimeError("slot must be unique and contiguous in the actual template")
    start = starts[0]
    prefix, suffix = template_ids[:start], template_ids[start + len(slot_ids) :]
    body_size = n_target - len(prefix) - len(suffix)
    if body_size <= 0:
        raise RuntimeError("template/query consumes the entire prompt budget")
    body = _build_filler(tokenizer, FILLER_SENTENCE, body_size)
    offsets, occupied = [], []
    for depth, needle_ids in needle_specs:
        offset = int(depth * n_target)
        body_offset = offset - len(prefix)
        end = body_offset + len(needle_ids)
        if not 0 <= body_offset < end <= len(body):
            raise RuntimeError("requested needle position is outside the template body")
        if any(
            body_offset < previous_end and previous_start < end
            for previous_start, previous_end in occupied
        ):
            raise RuntimeError("needle token ranges overlap")
        body[body_offset:end] = needle_ids
        occupied.append((body_offset, end))
        offsets.append(offset)
    prompt_ids = prefix + body + suffix
    if len(prompt_ids) != n_target:
        raise RuntimeError("token-space construction violated its exact length")
    return prompt_ids, offsets, tokenizer.decode(prompt_ids)


def build_case(
    tokenizer: TokenizerProtocol,
    *,
    case_id: str,
    n_prompt_tokens: int,
    depths: Sequence[float],
    codes: Sequence[str],
    query: str,
    system: str = SYSTEM_PROMPT,
    filler: str = FILLER_SENTENCE,
) -> NiahCase:
    """Construct one exact-token case and validate its internal invariants.

    Each needle occupies a unique contiguous token slot at ``int(depth*N)`` in
    the final rendered stream. Any encoding mismatch raises; nothing is
    silently re-encoded, cropped or truncated, and the encoder is never
    switched.
    """
    if not codes or len(codes) != len(depths):
        raise ValueError("codes and depths must be nonempty and equal length")
    if len(set(codes)) != len(codes):
        raise ValueError("codes must be unique within a case")
    if n_prompt_tokens > MAX_PROMPT:
        raise ValueError(
            f"{case_id}: requested {n_prompt_tokens} prompt tokens exceeds "
            f"full-window budget {MAX_PROMPT} (WINDOW {WINDOW} - RESERVE {RESERVE})"
        )
    for d in depths:
        if not 0.0 < d < 1.0:
            raise ValueError(f"{case_id}: depth {d} outside (0,1)")

    needles: List[Needle] = []
    specs: List[Tuple[float, List[int]]] = []
    for code, depth in zip(codes, depths):
        text = NEEDLE_TEMPLATE.format(code=code)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise RuntimeError(f"{case_id}: needle for {code!r} encoded empty")
        needles.append(Needle(code=code, text=text, token_ids=ids, start_offset=-1))
        specs.append((depth, ids))

    prompt_ids, offsets, rendered = _encode_with_positions(
        tokenizer, system, specs, query, n_prompt_tokens
    )

    # --- hard invariants on the real rendered stream -----------------------
    prompt_list = list(prompt_ids)
    spans = sorted((off, off + len(n.token_ids)) for n, off in zip(needles, offsets))
    for (_, end), (start2, _) in zip(spans, spans[1:]):
        if start2 < end:
            raise RuntimeError(f"{case_id}: needle slots overlap: {spans}")
    for needle, offset in zip(needles, offsets):
        window = prompt_list[offset : offset + len(needle.token_ids)]
        if window != needle.token_ids:
            raise RuntimeError(
                f"{case_id}: slot encoding mismatch for {needle.code!r} at offset "
                f"{offset}: fix the encoder/template root cause; never switch "
                "encoders silently"
            )
        occurrences = sum(
            1
            for i in range(len(prompt_list) - len(needle.token_ids) + 1)
            if prompt_list[i : i + len(needle.token_ids)] == needle.token_ids
        )
        if occurrences != 1:
            raise RuntimeError(
                f"{case_id}: needle {needle.code!r} occurs {occurrences} times in "
                "the rendered stream; must be exactly 1 (unique slot)"
            )

    return NiahCase(
        case_id=case_id,
        n_tokens=len(prompt_ids),
        depths=tuple(depths),
        needles=tuple(
            Needle(code=n.code, text=n.text, token_ids=n.token_ids, start_offset=off)
            for n, off in zip(needles, offsets)
        ),
        prompt_ids=tuple(prompt_ids),
        response_starts_in_thinking=rendered.rfind("<think>")
        > rendered.rfind("</think>"),
        rendered_text_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        prompt_sha256=hashlib.sha256(
            json.dumps(list(prompt_ids)).encode("utf-8")
        ).hexdigest(),
    )


def build_default_cases(tokenizer: TokenizerProtocol) -> List[NiahCase]:
    """The nine frozen cases: 8 single-key + 1 full-window ordered multi-key."""
    cases: List[NiahCase] = []
    for ctx in (8_192, 32_768, 131_072):
        cases.append(
            build_case(
                tokenizer,
                case_id=f"single_{ctx}_d50",
                n_prompt_tokens=ctx,
                depths=(0.50,),
                codes=(CODE_POOL[len(cases)],),
                query=QUERY_TEMPLATE.format(n=1),
            )
        )
    for i, depth in enumerate((0.05, 0.25, 0.50, 0.75, 0.95)):
        cases.append(
            build_case(
                tokenizer,
                case_id=f"single_{MAX_PROMPT}_d{int(depth * 100)}",
                n_prompt_tokens=MAX_PROMPT,
                depths=(depth,),
                codes=(CODE_POOL[3 + i],),
                query=QUERY_TEMPLATE.format(n=1),
            )
        )
    cases.append(
        build_case(
            tokenizer,
            case_id=f"multi_{MAX_PROMPT}_d33_66",
            n_prompt_tokens=MAX_PROMPT,
            depths=(0.33, 0.66),
            codes=(CODE_POOL[8], CODE_POOL[9]),
            query=QUERY_TEMPLATE.format(n=2),
        )
    )
    return cases


# ---------------------------------------------------------------------------
# Manifest freezing
# ---------------------------------------------------------------------------


def freeze_manifest(
    cases: Sequence[NiahCase], tokenizer: TokenizerProtocol
) -> Dict[str, Any]:
    """Freeze token arrays, hashes, offsets and codes BEFORE any traffic."""
    return {
        "kind": "qualification-niah-manifest",
        "model_id": MODEL_ID,
        "window": WINDOW,
        "reserve": RESERVE,
        "max_prompt_tokens": MAX_PROMPT,
        "sampling": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        },
        "timeout_s": TIMEOUT_S,
        "tokenizer": getattr(tokenizer, "name", "unknown"),
        "cases": [
            {
                "case_id": c.case_id,
                "n_tokens": c.n_tokens,
                "depths": list(c.depths),
                "codes": [n.code for n in c.needles],
                "needle_offsets": [n.start_offset for n in c.needles],
                "needle_token_lens": [len(n.token_ids) for n in c.needles],
                "prompt_sha256": c.prompt_sha256,
                "prompt_ids": list(c.prompt_ids),
                "response_starts_in_thinking": c.response_starts_in_thinking,
                "rendered_text_sha256": c.rendered_text_sha256,
            }
            for c in cases
        ],
    }


# ---------------------------------------------------------------------------
# Response checking (fail-closed)
# ---------------------------------------------------------------------------


def check_response(case: NiahCase, res: CompletionResult) -> Tuple[str, str]:
    """Return (verdict, detail).

    verdict is one of: pass / needle_miss / invalid_usage_echo /
    invalid_finish / incomplete_thinking / empty_answer /
    invalid_usage / infra_error.
    """
    if res.error is not None:
        if res.error.startswith("http_status_") or res.error in (
            "transport",
            "timeout",
        ):
            return "infra_error", f"{res.error}: {res.error_detail[:300]}"
        return "invalid_usage", f"{res.error}: {res.error_detail[:300]}"
    if not isinstance(res.usage, dict) or any(
        type(res.usage.get(k)) is not int
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return "invalid_usage", "usage must contain exact JSON integers"
    if (
        not 0 < res.usage["completion_tokens"] <= MAX_TOKENS
        or res.usage["total_tokens"]
        != res.usage["prompt_tokens"] + res.usage["completion_tokens"]
    ):
        return "invalid_usage", "invalid completion count or usage total"
    if res.usage["prompt_tokens"] != case.n_tokens:
        return (
            "invalid_usage_echo",
            f"api prompt_tokens={res.usage['prompt_tokens']} != constructed {case.n_tokens}",
        )
    if res.finish_reason != "stop":
        return "invalid_finish", f"finish_reason={res.finish_reason}"
    text = res.text or ""
    # incomplete thinking is not an answer
    if (case.response_starts_in_thinking and "</think>" not in text) or text.rfind(
        "<think>"
    ) > text.rfind("</think>"):
        return "incomplete_thinking", "unclosed <think> block"
    final = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    if not final:
        return "empty_answer", "no final text after thinking"
    codes = [n.code for n in case.needles]
    if " ".join(final.split()) != " ".join(codes):
        return "needle_miss", "final answer is not exactly the ordered code list"
    return "pass", "exact ordered final codes"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


DEFAULT_CASE_IDS = tuple(
    [f"single_{n}_d50" for n in (8192, 32768, 131072)]
    + [f"single_{MAX_PROMPT}_d{d}" for d in (5, 25, 50, 75, 95)]
    + [f"multi_{MAX_PROMPT}_d33_66"]
)


def run_cases(client, cases, out_path, *, tokenizer=None, dry_run=False, on_row=None):
    """Public traffic gate: only pinned real-tokenizer default cases may run."""
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case IDs must be nonempty and unique")
    if not dry_run:
        if (
            tokenizer is None
            or isinstance(tokenizer, FakeTokenizer)
            or str(getattr(tokenizer, "name", "")).startswith("test-only:")
        ):
            raise ValueError(
                "real traffic requires the pinned real tokenizer, never FakeTokenizer"
            )
        from transformers import PreTrainedTokenizerBase

        if not isinstance(tokenizer, PreTrainedTokenizerBase) or not getattr(
            tokenizer, "_qwen38_tokenizer_assets", None
        ):
            raise ValueError("tokenizer was not admitted by load_real_tokenizer")
        expected = {c.case_id: c for c in build_default_cases(tokenizer)}
        if any(c.case_id not in expected or c != expected[c.case_id] for c in cases):
            raise ValueError(
                "traffic arrays differ from real-tokenizer default construction"
            )
    return _execute_cases(client, cases, out_path, dry_run=dry_run, on_row=on_row)


def _execute_cases(
    client: Any,
    cases: Sequence[NiahCase],
    out_path: Path,
    *,
    dry_run: bool = False,
    on_row: Any = None,
) -> Dict[str, Any]:
    """Run (or dry-run) cases; persist every row; return a fail-closed summary."""
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("case IDs must be nonempty and unique")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts: Dict[str, str] = {}
    ran = 0
    with out_path.open("x", encoding="utf-8") as fout:
        for case in cases:
            if dry_run:
                row = {
                    "case_id": case.case_id,
                    "dry_run": True,
                    "n_tokens": case.n_tokens,
                    "verdict": "not_run_dry",
                    "prompt_sha256": case.prompt_sha256,
                }
                verdicts[case.case_id] = "not_run_dry"
            else:
                try:
                    res = client.completions_tokens(
                        list(case.prompt_ids),
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        timeout=TIMEOUT_S,
                        extra={"skip_special_tokens": False},
                    )
                except Exception as exc:  # infra: recorded, never a needle miss
                    row = {
                        "case_id": case.case_id,
                        "n_tokens": case.n_tokens,
                        "verdict": "infra_error",
                        "detail": f"{type(exc).__name__}: {exc}"[:500],
                        "finish_reason": None,
                        "usage": None,
                        "wall_s": None,
                        "final_text": "",
                        "raw_body": "",
                        "prompt_sha256": case.prompt_sha256,
                    }
                    verdicts[case.case_id] = "infra_error"
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    if on_row:
                        on_row(row)
                    continue
                verdict, detail = check_response(case, res)
                verdicts[case.case_id] = verdict
                ran += 1
                row = {
                    "case_id": case.case_id,
                    "n_tokens": case.n_tokens,
                    "verdict": verdict,
                    "detail": detail,
                    "finish_reason": res.finish_reason,
                    "usage": res.usage,
                    "wall_s": res.wall_s,
                    "final_text": res.text or "",
                    "raw_body": res.raw_body,  # bounded by client; no silent evidence truncation
                    "prompt_sha256": case.prompt_sha256,
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if on_row:
                on_row(row)
    passed = [k for k, v in verdicts.items() if v == "pass"]
    missing = sorted(set(DEFAULT_CASE_IDS) - set(verdicts))
    partial = (
        bool(missing)
        or set(verdicts) != set(DEFAULT_CASE_IDS)
        or any(v == "not_run_dry" for v in verdicts.values())
    )
    return {
        "total_cases": len(DEFAULT_CASE_IDS),
        "selected_cases": len(cases),
        "selected_ok": not dry_run and len(passed) == len(cases),
        "missing_case_ids": missing,
        "ran": ran,
        "passed": len(passed),
        "partial": partial or ran != len(cases),
        "verdicts": verdicts,
        "ok": (not partial) and len(passed) == len(DEFAULT_CASE_IDS),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="exact-token NIAH (262144 window)")
    p.add_argument(
        "--base", help="explicit endpoint base URL (required unless --dry-run)"
    )
    p.add_argument(
        "--tokenizer-dir", default=os.environ.get("QUAL_HARNESS_TOKENIZER_DIR", "")
    )
    p.add_argument("--manifest", required=True, help="frozen manifest output path")
    p.add_argument(
        "--output", required=True, help="JSONL rows (exclusive, no overwrite)"
    )
    p.add_argument(
        "--only", action="append", help="run only these case_ids (repeatable)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="construct+freeze+validate, no traffic"
    )
    args = p.parse_args(argv)

    if args.dry_run:
        if not args.tokenizer_dir:
            print(
                "--dry-run still needs --tokenizer-dir to freeze real arrays",
                file=sys.stderr,
            )
            return 2
        tokenizer = load_real_tokenizer(args.tokenizer_dir)
    else:
        if not args.base:
            print("--base is required for real traffic", file=sys.stderr)
            return 2
        if not args.tokenizer_dir:
            print(
                "--tokenizer-dir (or QUAL_HARNESS_TOKENIZER_DIR) is required: the same "
                "actual tokenizer/template must render NIAH traffic",
                file=sys.stderr,
            )
            return 2
        tokenizer = load_real_tokenizer(args.tokenizer_dir)

    cases = build_default_cases(tokenizer)
    if args.only:
        want = set(args.only)
        if len(want) != len(args.only) or not want <= {c.case_id for c in cases}:
            print("--only has duplicate or unknown case IDs", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.case_id in want]
        if not cases:
            print("no cases matched --only", file=sys.stderr)
            return 2

    manifest = freeze_manifest(cases, tokenizer)
    manifest["partial_selection"] = bool(args.only)
    manifest_path = Path(args.manifest)
    out = Path(args.output)
    if (
        manifest_path.resolve() == out.resolve()
        or manifest_path.exists()
        or out.exists()
    ):
        print("manifest/output must be distinct fresh paths", file=sys.stderr)
        return 2
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if args.dry_run:
        summary = run_cases(None, cases, out, dry_run=True)
        summary["construction_ok"] = True
    else:
        client = OpenAICompatClient(args.base)
        client.verify_model()  # exact identity before heavy traffic
        summary = run_cases(client, cases, out, tokenizer=tokenizer)
    summary["partial_selection"] = bool(args.only)
    print(json.dumps(summary, indent=2))
    # Successful preparation is not a successful retrieval run: ok stays false.
    return 0 if args.dry_run or summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
