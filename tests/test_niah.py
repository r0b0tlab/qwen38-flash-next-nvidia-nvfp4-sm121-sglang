"""Unit tests for scripts/niah.py using a FakeTokenizer (construction only).

These tests validate CASE CONSTRUCTION and fail-closed checking. They never
produce real model output; the FakeTokenizer cannot generate traffic (the
runner refuses test-only tokenizers for real requests — enforced by
TestFakeTokenizerIsolation). Real-tokenizer integration requires the model
directory and is run by the parent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import niah as nz  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    return nz.FakeTokenizer()


# ---------------------------------------------------------------------------
# construction invariants
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_single_small_case_slots(self, tok):
        case = nz.build_case(
            tok,
            case_id="t_small",
            n_prompt_tokens=600,
            depths=(0.5,),
            codes=("ZEPHYR-4821",),
            query=nz.QUERY_TEMPLATE.format(n=1),
        )
        assert case.n_tokens == 600
        needle = case.needles[0]
        ids = list(case.prompt_ids)
        window = ids[needle.start_offset : needle.start_offset + len(needle.token_ids)]
        assert window == needle.token_ids  # contiguous exact slot

    def test_single_case_unique_needle(self, tok):
        case = nz.build_case(
            tok,
            case_id="t_unique",
            n_prompt_tokens=512,
            depths=(0.5,),
            codes=("QUARTZ-9173",),
            query=nz.QUERY_TEMPLATE.format(n=1),
        )
        ids = list(case.prompt_ids)
        n = case.needles[0]
        occurrences = sum(
            1
            for i in range(len(ids) - len(n.token_ids) + 1)
            if ids[i : i + len(n.token_ids)] == n.token_ids
        )
        assert occurrences == 1

    def test_window_budget_enforced(self, tok):
        with pytest.raises(ValueError, match="full-window budget"):
            nz.build_case(
                tok,
                case_id="t_over",
                n_prompt_tokens=nz.MAX_PROMPT + 1,
                depths=(0.5,),
                codes=("X-1",),
                query="q",
            )

    def test_depth_bounds(self, tok):
        with pytest.raises(ValueError, match="depth"):
            nz.build_case(
                tok,
                case_id="t_depth",
                n_prompt_tokens=512,
                depths=(1.5,),
                codes=("X-1",),
                query="q",
            )

    def test_duplicate_codes_rejected(self, tok):
        with pytest.raises(ValueError, match="unique"):
            nz.build_case(
                tok,
                case_id="t_dup",
                n_prompt_tokens=512,
                depths=(0.3, 0.6),
                codes=("X-1", "X-1"),
                query="q",
            )

    def test_depth_count_mismatch(self, tok):
        with pytest.raises(ValueError, match="equal length"):
            nz.build_case(
                tok,
                case_id="t_cnt",
                n_prompt_tokens=512,
                depths=(0.3, 0.6),
                codes=("X-1",),
                query="q",
            )

    def test_multi_key_ordered_case(self, tok):
        case = nz.build_case(
            tok,
            case_id="t_multi",
            n_prompt_tokens=900,
            depths=(0.33, 0.66),
            codes=("AAA-1111", "BBB-2222"),
            query=nz.QUERY_TEMPLATE.format(n=2),
        )
        a, b = case.needles
        assert a.start_offset < b.start_offset
        # both present exactly once, each contiguous at its offset
        ids = list(case.prompt_ids)
        for n in (a, b):
            assert (
                ids[n.start_offset : n.start_offset + len(n.token_ids)] == n.token_ids
            )

    def test_rendered_length_exact(self, tok):
        """The construction loop must land exactly on the requested count."""
        for n in (256, 511, 512, 513, 1000):
            case = nz.build_case(
                tok,
                case_id=f"t_len{n}",
                n_prompt_tokens=n,
                depths=(0.5,),
                codes=("ZEPHYR-4821",),
                query=nz.QUERY_TEMPLATE.format(n=1),
            )
            assert case.n_tokens == n

    def test_nine_default_cases_shape(self, tok):
        cases = nz.build_default_cases(tok)
        assert len(cases) == 9
        ids = [c.case_id for c in cases]
        assert ids[:3] == ["single_8192_d50", "single_32768_d50", "single_131072_d50"]
        full = [c for c in cases if c.n_tokens == nz.MAX_PROMPT]
        assert len(full) == 6  # 5 single full-window + 1 multi
        multi = cases[-1]
        assert len(multi.needles) == 2
        assert multi.needles[0].code != multi.needles[1].code
        depths = [round(d, 2) for d in multi.depths]
        assert depths == [0.33, 0.66]

    def test_freeze_manifest_fields(self, tok):
        cases = nz.build_default_cases(tok)
        m = nz.freeze_manifest(cases, tok)
        assert m["window"] == 262_144 and m["reserve"] == 4_096
        assert m["max_prompt_tokens"] == 258_044
        assert m["sampling"] == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 4_096}
        assert m["timeout_s"] == 43_200.0
        assert len(m["cases"]) == 9
        for entry in m["cases"]:
            assert set(entry) >= {
                "case_id",
                "n_tokens",
                "depths",
                "codes",
                "needle_offsets",
                "prompt_sha256",
                "rendered_text_sha256",
            }
            assert len(entry["prompt_sha256"]) == 64


# ---------------------------------------------------------------------------
# fail-closed response checking
# ---------------------------------------------------------------------------


def _res(**kw):
    from http_client import CompletionResult

    base = dict(
        text="<think>hmm</think>ZEPHYR-4821",
        finish_reason="stop",
        usage={"prompt_tokens": 600, "completion_tokens": 10, "total_tokens": 610},
    )
    base.update(kw)
    return CompletionResult(**base)


class TestCheckResponse:
    def _case(self, tok, codes=("ZEPHYR-4821",), n=600, depths=(0.5,)):
        return nz.build_case(
            tok,
            case_id="chk",
            n_prompt_tokens=n,
            depths=depths,
            codes=codes,
            query=nz.QUERY_TEMPLATE.format(n=len(codes)),
        )

    def test_pass(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(case, _res())
        assert v == "pass"

    def test_needle_miss(self, tok):
        case = self._case(tok, codes=("QUARTZ-9173",))
        v, _ = nz.check_response(case, _res(text="<think>h</think>ZEPHYR-4821"))
        assert v == "needle_miss"

    def test_order_matters(self, tok):
        case = self._case(
            tok, codes=("AAA-1111", "BBB-2222"), n=900, depths=(0.33, 0.66)
        )
        usage = {"prompt_tokens": 900, "completion_tokens": 10, "total_tokens": 910}
        v, _ = nz.check_response(case, _res(text="BBB-2222 AAA-1111", usage=usage))
        assert v == "needle_miss"
        v, _ = nz.check_response(case, _res(text="AAA-1111 BBB-2222", usage=usage))
        assert v == "pass"

    def test_usage_echo_mismatch(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(
            case,
            _res(
                usage={
                    "prompt_tokens": 599,
                    "completion_tokens": 3,
                    "total_tokens": 602,
                }
            ),
        )
        assert v == "invalid_usage_echo"

    def test_finish_length_fails(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(case, _res(finish_reason="length"))
        assert v == "invalid_finish"

    def test_incomplete_thinking_not_answer(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(case, _res(text="<think>still reasoning about codes"))
        assert v == "incomplete_thinking"

    def test_empty_after_thinking(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(case, _res(text="<think>done</think>"))
        assert v == "empty_answer"

    def test_transport_is_infra_not_miss(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(
            case, _res(error="timeout", error_detail="read timed out")
        )
        assert v == "infra_error"
        v, _ = nz.check_response(
            case, _res(error="http_status_503", error_detail="unavailable")
        )
        assert v == "infra_error"

    def test_invalid_usage_reported(self, tok):
        case = self._case(tok)
        v, _ = nz.check_response(
            case, _res(error="invalid_usage", error_detail="float usage")
        )
        assert v == "invalid_usage"


# ---------------------------------------------------------------------------
# fake tokenizer can never reach real traffic
# ---------------------------------------------------------------------------


class TestFakeTokenizerIsolation:
    def test_fake_name_is_marked_test_only(self, tok):
        assert tok.name.startswith("test-only:")

    def test_runner_refuses_fake_tokenizer(self, tok, tmp_path):
        """run_cases must reject a test-only tokenizer for real traffic."""
        cases = [
            nz.build_case(
                tok,
                case_id="iso",
                n_prompt_tokens=128,
                depths=(0.5,),
                codes=("ZEPHYR-4821",),
                query="code?",
            )
        ]

        class RefusingClient:
            def completions_tokens(self, *a, **kw):
                raise AssertionError("fake tokenizer must never produce traffic")

        with pytest.raises(ValueError, match="real traffic"):
            nz.run_cases(
                RefusingClient(), cases, tmp_path / "rows.jsonl", tokenizer=tok
            )
        assert not (tmp_path / "rows.jsonl").exists()

    def test_load_real_tokenizer_rejects_missing_dir(self):
        with pytest.raises(Exception):
            nz.load_real_tokenizer("/nonexistent/tokenizer/dir")


# ---------------------------------------------------------------------------
# manifest freezing determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_construction_same_hashes(self, tok):
        def build():
            return nz.build_case(
                tok,
                case_id="det",
                n_prompt_tokens=512,
                depths=(0.5,),
                codes=("ZEPHYR-4821",),
                query=nz.QUERY_TEMPLATE.format(n=1),
            )

        a, b = build(), build()
        assert a.prompt_sha256 == b.prompt_sha256
        assert a.rendered_text_sha256 == b.rendered_text_sha256
        assert list(a.prompt_ids) == list(b.prompt_ids)
