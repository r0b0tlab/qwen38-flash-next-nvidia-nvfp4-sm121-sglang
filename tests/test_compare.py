"""Tests for scripts/bench_real.py and scripts/compare.py (CPU scaffold only).

No live endpoint: the fake server produces synthetic vectors. These tests
verify freeze-before-traffic, row validity, exclusivity, and the fail-closed
comparison verdicts (NOT_OPTIMIZED vs PASS, 5%-per-case gate, upstream
count/keys gate, vision p95 TTFT gate).
"""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_real as br  # noqa: E402
import compare as cmp  # noqa: E402
from fake_sse_server import FakeServer  # noqa: E402

CASES = ("db_index_write_read", "city_heat_water", "compression_vs_random_access")


# ---------------------------------------------------------------------------
# frozen corpus
# ---------------------------------------------------------------------------


class TestFrozenCorpus:
    def test_three_meaningful_prose_cases(self):
        assert len(br.PROSE_CASES) == 3
        for case in br.PROSE_CASES:
            words = case["prompt"].split()
            assert len(words) > 180, f"{case['case_id']} not ~250 words"
            assert case["case_id"] in CASES

    def test_prompts_are_not_counting_tricks(self):
        """Prose prompts ask for explanations, not enumeration answers."""
        for case in br.PROSE_CASES:
            low = case["prompt"].lower()
            assert "explain" in low or "describe" in low or "advise" in low
            for banned in ("how many", "count the", "list all "):
                assert banned not in low

    def test_corpus_sha_stable(self):
        assert br.corpus_sha256() == br.corpus_sha256()
        assert len(br.corpus_sha256()) == 64

    def test_corpus_lock_write_and_refuse_overwrite(self, tmp_path):
        lock = tmp_path / "run.corpus.json"
        br.write_corpus_lock(lock, base="http://x", thinking={}, repeats=5, warmup=1)
        first = json.loads(lock.read_text())
        assert first["corpus_sha256"] == br.corpus_sha256()
        assert first["frozen_before_traffic_utc"]
        assert first["sampling"]["max_tokens"] == 2048
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            br.write_corpus_lock(
                lock, base="http://x", thinking={}, repeats=5, warmup=1
            )

    def test_default_thinking_controls(self):
        from http_client import thinking_request_fields

        assert thinking_request_fields(False) == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert thinking_request_fields(True, "low") == {
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_effort": "low",
        }
        # bench_real's default thinking policy is enable_thinking=False
        import inspect

        src = inspect.getsource(br.run_bench)
        assert "thinking_request_fields(False)" in src


# ---------------------------------------------------------------------------
# bench run against the fake server
# ---------------------------------------------------------------------------


class TestBenchRun:
    def _run(self, tmp_path, srv, **kw):
        out = tmp_path / "rows.jsonl"
        summary = br.run_bench(
            srv.base,
            out,
            repeats=kw.get("repeats", 2),
            warmup=kw.get("warmup", 1),
            timeout_s=30,
            variant="AR",
        )
        return out, summary

    def _set_default(self, srv, scenario):
        srv.httpd.chat_scenario_default = scenario

    def test_rows_and_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(br, "MIN_MEANINGFUL_CHARS", 20)
        with FakeServer() as srv:
            self._set_default(srv, "long-answer")
            out, summary = self._run(tmp_path, srv)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        warm = [r for r in rows if r["warmup"]]
        meas = [r for r in rows if not r["warmup"]]
        assert len(warm) == 3 and len(meas) == 6
        assert summary["measured_rows"] == 6
        assert summary["ok"] is True
        for r in meas:
            assert r["valid"] is True
            assert r["finish_reason"] == "stop"
            assert r["usage"]["completion_tokens"] > 0
            assert r["e2erate"] > 0
            assert r["model"] == br.MODEL_ID
            assert r["raw_events"], "raw evidence preserved per row"

    def test_finish_length_is_invalid_diagnostic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(br, "MIN_MEANINGFUL_CHARS", 20)
        with FakeServer() as srv:
            self._set_default(srv, "finish-length")
            out, summary = self._run(tmp_path, srv)
        summary_rows = summary["invalid_rows"]
        assert summary_rows, "length-limited diagnostic must not be a silent pass"
        assert all(r["reason"] == "finish_length" for r in summary_rows)
        assert summary["ok"] is False

    def test_flush_cold_requires_owner_admission(self, tmp_path):
        with FakeServer() as srv:
            with pytest.raises(SystemExit, match="QUAL_HARNESS_OWNER_ADMITTED"):
                br.run_bench(srv.base, tmp_path / "o.jsonl", flush_cold=True)

    def test_flush_cold_with_admission(self, tmp_path, monkeypatch):
        monkeypatch.setenv("QUAL_HARNESS_OWNER_ADMITTED", "1")
        monkeypatch.setattr(br, "MIN_MEANINGFUL_CHARS", 20)
        with FakeServer() as srv:
            self._set_default(srv, "long-answer")
            out, summary = self._run(tmp_path, srv)
        assert summary["ok"] is True

    def test_cli_refuses_overwrite(self, tmp_path):
        out = tmp_path / "x.jsonl"
        out.write_text("seed\n")
        rc = br.main(["--base", "http://127.0.0.1:1", "--output", str(out)])
        assert rc == 2


# ---------------------------------------------------------------------------
# compare.py: fail-closed reducers
# ---------------------------------------------------------------------------


def _vision_rows(ttft, count, error=None):
    return [
        {
            "ttft_s": ttft,
            "wall_s": 2.0,
            "warmup": False,
            "error": error,
            "case_id": f"test-{i}",
            "repeat": 0,
            "input_sha256": "a" * 64,
            "valid": True,
            "finish_reason": "stop",
        }
        for i in range(count)
    ]


def _mk_rows(rate: float, repeats: int = 5, usage_completion: int = 500) -> list:
    rows = []
    for case in CASES:
        for rep in range(repeats):
            wall = usage_completion / rate
            rows.append(
                {
                    "case": case,
                    "repeat": rep,
                    "warmup": False,
                    "valid": True,
                    "reason": None,
                    "ok": True,
                    "error": None,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 320,
                        "completion_tokens": usage_completion,
                        "total_tokens": 820,
                    },
                    "wall_s": wall,
                    "ttft_s": 0.2,
                    "e2erate": rate,
                    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
                    "variant": "X",
                    "raw_events": [{"test_only": True}],
                }
            )
    return rows


def _manifest(lever="none", **over):
    m = {
        "model_sha": "fc694b54fb0174e0913e6adf86691ef85a4ead47",
        "image_id": "img-123",
        "source_tree": "7f51cdd",
        "tokenizer": "qwen3",
        "input_token_sha": "abc",
        "sampling": {"temperature": 0, "top_p": 1},
        "context": 262144,
        "total_pool": 64,
        "concurrency": 8,
        "lever": lever,
    }
    m.update(over)
    return m


def _upstream(rate: float, rounds: int = 5, completed: int = 8) -> str:
    lines = []
    for repeat in range(rounds):
        dur = 2048 / rate
        lines.append(
            json.dumps(
                {
                    "duration": dur,
                    "repeat": repeat,
                    "native_result_sha256": hashlib.sha256(
                        f"test-only/{rate}/{repeat}".encode()
                    ).hexdigest(),
                    "completed": completed,
                    "total_input_tokens": 4096,
                    "total_output_tokens": 2048,
                    "output_throughput": rate,
                    "backend": "sglang-oai",
                    "max_concurrency": 1,
                    "random_input_len": 512,
                    "random_output_len": 256,
                    "input_lens": [512] * 8,
                    "output_lens": [256] * 8,
                    "usage_source": "observed_oai_sse",
                    "finish_source": "observed_oai_sse",
                    "finish_reasons": ["length"] * 8,
                    "observed_usage": [
                        {
                            "prompt_tokens": 512,
                            "completion_tokens": 256,
                            "total_tokens": 768,
                        }
                        for _ in range(8)
                    ],
                    "cached_tokens": [0] * 8,
                    "request_hashes": [f"{i:064x}" for i in range(8)],
                    "errors": [""] * 8,
                    "generated_texts": ["test-only synthetic fixture"] * 8,
                    "ttfts": [0.1] * 8,
                }
            )
        )
    return "\n".join(lines) + "\n"


class TestCompareFailClosed:
    def test_pass_requires_each_case_ge_5pct(self, tmp_path):
        b = tmp_path / "b.jsonl"
        b.write_text("\n".join(json.dumps(r) for r in _mk_rows(10.0)) + "\n")
        c = tmp_path / "c.jsonl"
        c.write_text("\n".join(json.dumps(r) for r in _mk_rows(11.0)) + "\n")
        report = cmp.compare(
            base_rows=cmp.measured_prose_rows(cmp.load_rows(b, "baseline"), "baseline"),
            cand_rows=cmp.measured_prose_rows(
                cmp.load_rows(c, "candidate"), "candidate"
            ),
            base_manifest=_manifest(),
            cand_manifest=_manifest(lever="nextn"),
        )
        assert report["verdict"] == "NOT_OPTIMIZED"
        assert set(report["gates"]["complete_lanes"]["missing"]) == {
            "short",
            "medium",
            "vision",
        }
        assert all(g["pass"] for g in report["gates"]["prose_3case"].values())

    def test_reject_when_only_one_case_improves(self, tmp_path):
        rows_b = _mk_rows(10.0)
        rows_c = _mk_rows(11.0)
        # sabotage one case on the candidate: no improvement there
        for r in rows_c:
            if r["case"] == "city_heat_water":
                r["wall_s"] = r["usage"]["completion_tokens"] / 10.0
                r["e2erate"] = 10.0
        b = tmp_path / "b.jsonl"
        b.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n")
        c = tmp_path / "c.jsonl"
        c.write_text("\n".join(json.dumps(r) for r in rows_c) + "\n")
        report = cmp.compare(
            base_rows=cmp.measured_prose_rows(cmp.load_rows(b, "b"), "b"),
            cand_rows=cmp.measured_prose_rows(cmp.load_rows(c, "c"), "c"),
            base_manifest=_manifest(),
            cand_manifest=_manifest(lever="nextn"),
        )
        assert report["verdict"] == "NOT_OPTIMIZED"
        assert "city_heat_water" in report["gates"]["prose_3case"]["rejected"]

    def test_reject_missing_repeat(self, tmp_path):
        rows_b = _mk_rows(10.0)
        rows_c = _mk_rows(11.0)[:-1]  # one row missing
        b = tmp_path / "b.jsonl"
        b.write_text("\n".join(json.dumps(r) for r in rows_b) + "\n")
        c = tmp_path / "c.jsonl"
        c.write_text("\n".join(json.dumps(r) for r in rows_c) + "\n")
        with pytest.raises(cmp.Reject, match="expected 5"):
            cmp.check_counts(
                cmp.measured_prose_rows(cmp.load_rows(c, "c"), "c"), "candidate"
            )

    def test_reject_nan_or_zero_rate(self):
        bad = _mk_rows(10.0)[0]
        bad["wall_s"] = 0.0
        with pytest.raises(cmp.Reject, match="finite and > 0"):
            cmp.measured_prose_rows([bad], "v")

    def test_reject_invalid_row(self):
        bad = _mk_rows(10.0)[0]
        bad["valid"] = False
        bad["reason"] = "finish_length"
        with pytest.raises(cmp.Reject, match="finish_length"):
            cmp.measured_prose_rows([bad], "v")

    def test_reject_e2erate_disagreement(self):
        bad = _mk_rows(10.0)[0]
        bad["e2erate"] = 99.0  # does not match usage/wall
        with pytest.raises(cmp.Reject, match="disagrees"):
            cmp.measured_prose_rows([bad], "v")

    def test_parity_missing_detail_rejected(self):
        m = _manifest()
        del m["tokenizer"]
        with pytest.raises(cmp.Reject, match="missing required detail"):
            cmp.check_parity(m, _manifest(lever="nextn"))

    def test_parity_mismatch_rejected(self):
        with pytest.raises(cmp.Reject, match="parity mismatch"):
            cmp.check_parity(
                _manifest(sampling={"temperature": 0}), _manifest(lever="nextn")
            )

    def test_parity_only_lever_may_differ(self):
        info = cmp.check_parity(_manifest(), _manifest(lever="nextn-spec"))
        assert info["lever"]["candidate"] == "nextn-spec"

    def test_mixed_epochs_rejected(self):
        with pytest.raises(cmp.Reject, match="mixed epochs"):
            cmp.check_parity(_manifest(lever="spec-k2"), _manifest(lever="spec-k4"))

    def test_upstream_keys_and_count_gate(self, tmp_path):
        good = tmp_path / "up.jsonl"
        good.write_text(_upstream(10.0))
        detail = cmp.load_upstream_detail(good, "baseline")
        assert cmp.upstream_median_rate(detail, "baseline") == pytest.approx(10.0)

    def test_upstream_wrong_completed_rejected(self, tmp_path):
        p = tmp_path / "up.jsonl"
        p.write_text(_upstream(10.0, completed=7))
        with pytest.raises(cmp.Reject, match="completed=7"):
            cmp.load_upstream_detail(p, "baseline")

    def test_upstream_missing_key_rejected(self, tmp_path):
        p = tmp_path / "up.jsonl"
        row = json.loads(_upstream(10.0).splitlines()[0])
        del row["total_input_tokens"]
        p.write_text(json.dumps(row) + "\n")
        with pytest.raises(cmp.Reject, match="missing keys"):
            cmp.load_upstream_detail(p, "baseline")

    def test_upstream_nan_rate_rejected(self, tmp_path):
        p = tmp_path / "up.jsonl"
        row = json.loads(_upstream(10.0).splitlines()[0])
        row["output_throughput"] = float("nan")
        row["total_output_tokens"] = 0
        p.write_text(json.dumps(row) + "\n")
        with pytest.raises(cmp.Reject):
            cmp.load_upstream_detail(p, "baseline")

    def test_upstream_stated_rate_disagreement_rejected(self, tmp_path):
        p = tmp_path / "up.jsonl"
        row = json.loads(_upstream(10.0).splitlines()[0])
        row["output_throughput"] = 99.0  # not tokens/duration
        p.write_text(json.dumps(row) + "\n")
        with pytest.raises(cmp.Reject, match="disagrees"):
            cmp.load_upstream_detail(p, "baseline")

    def test_upstream_rounds_must_be_five(self, tmp_path):
        p = tmp_path / "up.jsonl"
        p.write_text(_upstream(10.0, rounds=4))
        detail = cmp.load_upstream_detail(p, "baseline")
        with pytest.raises(cmp.Reject, match="expected 5"):
            cmp.upstream_median_rate(detail, "baseline")

    def test_upstream_five_percent_gate(self, tmp_path):
        b = tmp_path / "b.jsonl"
        b.write_text(_upstream(10.0))
        c = tmp_path / "c.jsonl"
        c.write_text(_upstream(10.4))  # 4% gain
        bd = cmp.load_upstream_detail(b, "baseline")
        cd = cmp.load_upstream_detail(c, "candidate")
        report = cmp.compare(
            base_rows=cmp.measured_prose_rows(
                cmp.load_rows(_write(tmp_path, "pb.jsonl", _mk_rows(10.0)), "b"), "b"
            ),
            cand_rows=cmp.measured_prose_rows(
                cmp.load_rows(_write(tmp_path, "pc.jsonl", _mk_rows(11.0)), "c"), "c"
            ),
            base_manifest=_manifest(),
            cand_manifest=_manifest(lever="nextn"),
            base_upstream=bd,
            cand_upstream=cd,
        )
        assert report["verdict"] == "NOT_OPTIMIZED"
        assert "short" in report["gates"]["upstream_short"]["rejected"]

    def test_vision_p95_ttft_regression_gate(self):
        bv = _vision_rows(1.0, 10)
        cv = _vision_rows(1.06, 10)
        gate = cmp.vision_ttft_gate(bv, cv)
        assert gate["pass"] is False  # 6% regression > 5% limit
        cv2 = _vision_rows(1.04, 10)
        assert cmp.vision_ttft_gate(bv, cv2)["pass"] is True

    def test_vision_rows_with_errors_rejected(self):
        bv = _vision_rows(1.0, 3)
        cv = _vision_rows(1.0, 3, error="http_status_500")
        with pytest.raises(cmp.Reject, match="http_status_500"):
            cmp.vision_ttft_gate(bv, cv)

    def test_cli_nonzero_on_not_optimized(self, tmp_path):
        b = tmp_path / "b.jsonl"
        b.write_text("\n".join(json.dumps(r) for r in _mk_rows(10.0)) + "\n")
        c = tmp_path / "c.jsonl"
        c.write_text("\n".join(json.dumps(r) for r in _mk_rows(10.3)) + "\n")
        bm = tmp_path / "bm.json"
        bm.write_text(json.dumps(_manifest()))
        cm = tmp_path / "cm.json"
        cm.write_text(json.dumps(_manifest(lever="nextn")))
        rc = cmp.main(
            [
                "--baseline",
                str(b),
                "--candidate",
                str(c),
                "--baseline-manifest",
                str(bm),
                "--candidate-manifest",
                str(cm),
            ]
        )
        assert rc == 1


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p
