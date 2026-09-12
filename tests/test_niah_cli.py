"""Driver wiring tests only; no real inference or claim-bearing responses."""

import json
from scripts import niah as nz


def small_case():
    tokenizer = nz.FakeTokenizer()
    case = nz.build_case(
        tokenizer,
        case_id="test_cli",
        n_prompt_tokens=512,
        depths=(0.5,),
        codes=("ZEPHYR-4821",),
        query="Return the code.",
    )
    return tokenizer, case


def test_manifest_persists_actual_wire_token_arrays():
    tokenizer, case = small_case()
    manifest = nz.freeze_manifest([case], tokenizer)
    assert manifest["cases"][0]["prompt_ids"] == list(case.prompt_ids)


def test_live_cli_constructs_and_verifies_client_before_one_run(tmp_path, monkeypatch):
    tokenizer, case = small_case()
    monkeypatch.setattr(nz, "load_real_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(nz, "build_default_cases", lambda tok: [case])
    calls = []

    class Client:
        def __init__(self, base):
            calls.append("construct")

        def verify_model(self):
            calls.append("verify")

    def run(
        client, cases, out, *, tokenizer=None, dry_run=False, on_row=None, retries=0
    ):
        assert isinstance(client, Client)
        assert calls == ["construct", "verify"]
        assert not dry_run
        calls.append("run")
        return {"ok": True, "ran": 1, "passed": 1}

    monkeypatch.setattr(nz, "OpenAICompatClient", Client)
    monkeypatch.setattr(nz, "run_cases", run)
    rc = nz.main(
        [
            "--base",
            "http://unit.test",
            "--tokenizer-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "rows.jsonl"),
        ]
    )
    assert rc == 0
    assert calls == ["construct", "verify", "run"]


def test_dry_run_success_is_not_retrieval_success(tmp_path, monkeypatch, capsys):
    tokenizer, case = small_case()
    monkeypatch.setattr(nz, "load_real_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(nz, "build_default_cases", lambda tok: [case])

    def forbid(*args, **kwargs):
        raise AssertionError("dry run constructed HTTP client")

    monkeypatch.setattr(nz, "OpenAICompatClient", forbid)
    rc = nz.main(
        [
            "--dry-run",
            "--tokenizer-dir",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output",
            str(tmp_path / "rows.jsonl"),
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert summary["construction_ok"] is True
    assert summary["ok"] is False
    assert summary["ran"] == summary["passed"] == 0
