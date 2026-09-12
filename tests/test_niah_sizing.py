"""Regression: full-window NIAH prompts must fit native admission.

r01 evidence (niah-full-multi.jsonl): a 258,048-token prompt was rejected —
the server adds num_reserved_tokens (4 for NEXTN steps=3 native MTP) to every
request, so prompt + generation + 4 must stay within WINDOW 262,144.
r02 evidence (niah-full-multi-r02.jsonl): 258,044-token prompts were ADMITTED
(manifest reserve 4100 = 4096 generation + 4 admission).

This test pins the corrected budget: MAX_PROMPT == 258,044.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import niah as nz  # noqa: E402


class TestNativeSizing:
    def test_window_constant(self):
        assert nz.WINDOW == 262_144

    def test_admission_reserve_exists(self):
        # The native MTP reservation the server adds to every request.
        assert getattr(nz, "MTP_ADMISSION_RESERVE", None) == 4

    def test_max_prompt_fits_native_admission(self):
        assert nz.MAX_PROMPT == nz.WINDOW - nz.RESERVE - nz.MTP_ADMISSION_RESERVE
        assert nz.MAX_PROMPT == 258_044

    def test_default_cases_fit_native_admission(self):
        tok = nz.FakeTokenizer()
        cases = nz.build_default_cases(tok)
        assert len(cases) == 9
        for c in cases:
            assert c.n_tokens + nz.RESERVE + nz.MTP_ADMISSION_RESERVE <= nz.WINDOW, (
                f"{c.case_id}: {c.n_tokens} + {nz.RESERVE} + "
                f"{nz.MTP_ADMISSION_RESERVE} > {nz.WINDOW}"
            )
        biggest = {c.case_id: c.n_tokens for c in cases}
        full = [cid for cid, n in biggest.items() if cid.startswith(("single_258", "multi_258"))]
        assert len(full) == 6
        assert all(biggest[cid] == 258_044 for cid in full)

    def test_build_case_rejects_above_budget(self, ):
        tok = nz.FakeTokenizer()
        with pytest.raises(ValueError, match="full-window budget"):
            nz.build_case(
                tok,
                case_id="t_over",
                n_prompt_tokens=nz.MAX_PROMPT + 1,
                depths=(0.5,),
                codes=("CODE-1",),
                query="What code?",
            )
