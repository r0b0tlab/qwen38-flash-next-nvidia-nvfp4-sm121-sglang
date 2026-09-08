"""NIAH acceptance-boundary regressions; synthetic responses only."""

import dataclasses
import pytest
from scripts import niah as n
from scripts.http_client import CompletionResult


def tiny():
    tok = n.FakeTokenizer()
    case = n.build_case(
        tok,
        case_id="single_8192_d50",
        n_prompt_tokens=512,
        depths=(0.5,),
        codes=(n.CODE_POOL[0],),
        query=n.QUERY_TEMPLATE.format(n=1),
    )
    return tok, case


class Spy:
    def __init__(self):
        self.calls = 0

    def completions_tokens(self, ids, **kw):
        self.calls += 1
        return CompletionResult(
            text=n.CODE_POOL[0],
            finish_reason="stop",
            usage={
                "prompt_tokens": len(ids),
                "completion_tokens": 10,
                "total_tokens": len(ids) + 10,
            },
        )


def test_public_runner_blocks_fake_before_client_call(tmp_path):
    tok, case = tiny()
    tok.name = "real:forged-name"
    client = Spy()
    with pytest.raises(ValueError):
        n.run_cases(client, [case], tmp_path / "rows.jsonl", tokenizer=tok)
    assert client.calls == 0 and not (tmp_path / "rows.jsonl").exists()


def test_duplicate_cases_rejected_before_output(tmp_path):
    _, case = tiny()
    with pytest.raises(ValueError):
        n.run_cases(None, [case, case], tmp_path / "rows.jsonl", dry_run=True)
    assert not (tmp_path / "rows.jsonl").exists()


def test_selected_success_is_not_full_suite_success(tmp_path):
    _, case = tiny()
    summary = n._execute_cases(Spy(), [case], tmp_path / "rows.jsonl")
    assert summary["selected_ok"] is True
    assert summary["ok"] is False and summary["partial"] is True
    assert summary["total_cases"] == 9 and summary["selected_cases"] == 1
    assert len(summary["missing_case_ids"]) == 8


def test_prompt_opened_thinking_needs_close_marker():
    _, case = tiny()
    case = dataclasses.replace(case, response_starts_in_thinking=True)
    res = Spy().completions_tokens(case.prompt_ids)
    assert n.check_response(case, res)[0] == "incomplete_thinking"
    res.text = "thinking about a code</think>" + n.CODE_POOL[0]
    assert n.check_response(case, res)[0] == "pass"


def test_reasoning_contains_code_is_not_final_answer():
    _, case = tiny()
    res = Spy().completions_tokens(case.prompt_ids)
    res.text = "I found " + n.CODE_POOL[0] + " and should consider it."
    assert n.check_response(case, res)[0] != "pass"


def test_zero_completion_with_answer_is_invalid_usage():
    _, case = tiny()
    res = Spy().completions_tokens(case.prompt_ids)
    res.usage.update(completion_tokens=0, total_tokens=case.n_tokens)
    assert n.check_response(case, res)[0] == "invalid_usage"
