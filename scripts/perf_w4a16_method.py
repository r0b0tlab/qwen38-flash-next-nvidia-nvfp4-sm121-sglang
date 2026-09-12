#!/usr/bin/env python3
"""Matched dedicated/ladder decode suite replicating the W4A16 TP=2 method.

Frozen method for cross-project comparability with the W4A16 TP=2 project's
``repo/scripts/run_perf_suite.py`` (sibling campaign repository):

- the same 16 frozen prompts, thinking OFF, temperature 0, top_p 1, streaming;
- 1 warmup (128 tokens), dedicated c1 x5 (2048 tokens), ladder c1/c2/c4 x3 (512);
- ``aggregate_output_tokens_per_second = sum(completion_tokens) / wall_seconds``.

This driver targets exactly one owned endpoint; it performs no admission and
must never be pointed at a shared or protected server.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from http_client import MODEL_ID, OpenAICompatClient  # noqa: E402


def parse_metrics(text: str) -> Dict[str, float]:
    """Extract the scheduler occupancy gauges from a /metrics scrape.

    Returns keys ``running`` and ``queue`` with the sum over label sets.
    Missing series are simply absent from the mapping.
    """
    totals: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name, key in (
            ("sglang:num_running_reqs", "running"),
            ("sglang:num_queue_reqs", "queue"),
        ):
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    value = float(line.split()[-1])
                except (ValueError, IndexError):
                    continue
                totals[key] = totals.get(key, 0.0) + value
    return totals


class OccupancySampler:
    """Background sampler proving a ladder level was truly admitted.

    ``queued_only`` is set when the server never ran the requested number of
    requests concurrently (they queued instead of running), which is exactly
    the failure mode a max_running_requests cap produces.
    """

    def __init__(self, base: str, expected: int, interval: float = 0.25):
        self.base = base.rstrip("/")
        self.expected = int(expected)
        self.interval = float(interval)
        self.max_running = 0.0
        self.max_queue = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _scrape(self) -> None:
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=5) as resp:
                values = parse_metrics(resp.read().decode("utf-8", "replace"))
        except Exception:
            return
        self.samples += 1
        self.max_running = max(self.max_running, values.get("running", 0.0))
        self.max_queue = max(self.max_queue, values.get("queue", 0.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._scrape()
            self._stop.wait(self.interval)

    def __enter__(self) -> "OccupancySampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def report(self) -> Dict[str, Any]:
        return {
            "expected_running": self.expected,
            "max_running_observed": self.max_running,
            "max_queue_observed": self.max_queue,
            "samples": self.samples,
            "true_concurrency": self.max_running >= self.expected,
        }

PROMPTS = [
    "Count from 1 to 700, one integer per line. No extra text.",
    "Write a complete Python implementation of an LRU cache with tests. Code only.",
    "Explain photosynthesis in twelve detailed paragraphs.",
    "List the first 300 prime numbers separated by commas. No explanation.",
    "Write a long technical guide to implementing a B-tree from scratch.",
    "Describe the causes and consequences of the French Revolution in detail.",
    "Write a complete SQL tutorial with progressively complex examples.",
    "Count backward from 900 to 1, one integer per line. No extra text.",
    "Write a detailed guide to distributed consensus and failure recovery.",
    "Implement a thread-safe work queue in Python with tests. Code only.",
    "Explain transformer inference optimization in twelve detailed paragraphs.",
    "Write a complete HTTP server using only the Python standard library. Code only.",
    "List 500 distinct English verbs separated by commas. No explanation.",
    "Write a detailed history of computer architecture from 1940 to today.",
    "Implement a persistent immutable map in Python with tests. Code only.",
    "Explain CUDA graphs, capture constraints, and replay semantics in depth.",
]

# Exact W4A16 request policy: both template keys, thinking disabled.
THINKING_OFF = {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}


def run_row(
    client: OpenAICompatClient, prompt: str, max_tokens: int, timeout: float
) -> Dict[str, Any]:
    result = client.chat_stream(
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        thinking=THINKING_OFF,
        timeout=timeout,
    )
    usage = result.usage
    valid = bool(
        result.ok
        and usage is not None
        and result.finish_reason in ("stop", "length")
        and result.wall_s > 0
    )
    row: Dict[str, Any] = {
        "error": result.error,
        "error_detail": result.error_detail,
        "http_status": result.http_status,
        "wall_s": result.wall_s,
        "ttft_s": result.ttft_s,
        "finish_reason": result.finish_reason,
        "usage": usage,
        "completion_tokens": usage["completion_tokens"] if usage else None,
        "valid": valid,
    }
    row["aggregate_output_tokens_per_second"] = (
        (row["completion_tokens"] / result.wall_s) if valid else None
    )
    return row


def run_group(
    client: OpenAICompatClient,
    prompts: List[str],
    max_tokens: int,
    timeout: float,
    concurrency: int,
    base: Optional[str] = None,
) -> Dict[str, Any]:
    sampler = OccupancySampler(base or client.base, concurrency)
    start = time.perf_counter()
    with sampler:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_row, client, p, max_tokens, timeout) for p in prompts]
            rows = [f.result() for f in futures]
    wall = time.perf_counter() - start
    valid = [r for r in rows if r["valid"]]
    errors = len(rows) - len(valid)
    total = sum(int(r["completion_tokens"]) for r in valid) if errors == 0 else 0
    aggregate = (
        total / wall if errors == 0 and wall > 0 and math.isfinite(wall) else None
    )
    return {
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "rows": rows,
        "errors": errors,
        "completion_tokens": total,
        "batch_wall_seconds": wall,
        "aggregate_output_tokens_per_second": aggregate,
        "occupancy": sampler.report(),
        "status": "PASS" if errors == 0 and aggregate else "FAIL",
    }


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _geomean(values: List[float]) -> float:
    if not values or any(v <= 0 or not math.isfinite(v) for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="explicit endpoint base URL")
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--dedicated-repeats", type=int, default=5)
    p.add_argument("--dedicated-max-tokens", type=int, default=2048)
    p.add_argument("--ladder-max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--variant", default="", help="label (e.g. baseline / marlin)")
    p.add_argument(
        "--quick",
        action="store_true",
        help="1 warmup/1 dedicated/1 repeat, first two concurrency levels",
    )
    args = p.parse_args(argv)

    dedicated_repeats = 1 if args.quick else args.dedicated_repeats
    repeats = 1 if args.quick else args.repeats
    concurrencies = args.concurrency if not args.quick else args.concurrency[:2]

    client = OpenAICompatClient(args.base, model=args.model)
    client.verify_model()

    warmups: List[Dict[str, Any]] = []
    for index in range(args.warmups):
        warmups.append(run_row(client, PROMPTS[index], 128, args.timeout))
        if not warmups[-1]["valid"]:
            raise SystemExit(f"warmup failed: {warmups[-1]}")

    dedicated: List[Dict[str, Any]] = []
    for repeat in range(dedicated_repeats):
        dedicated.append(
            run_group(
                client,
                [PROMPTS[repeat % len(PROMPTS)]],
                args.dedicated_max_tokens,
                args.timeout,
                1,
                base=args.base,
            )
        )

    ladder: Dict[str, List[Dict[str, Any]]] = {}
    for concurrency in concurrencies:
        rows: List[Dict[str, Any]] = []
        for repeat in range(repeats):
            offset = repeat * concurrency
            prompts = [
                PROMPTS[(offset + i) % len(PROMPTS)] for i in range(concurrency)
            ]
            rows.append(
                run_group(
                    client,
                    prompts,
                    args.ladder_max_tokens,
                    args.timeout,
                    concurrency,
                    base=args.base,
                )
            )
        ladder[str(concurrency)] = rows

    def rates(rows: List[Dict[str, Any]]) -> List[float]:
        return [
            float(r["aggregate_output_tokens_per_second"])
            for r in rows
            if r["status"] == "PASS"
            and isinstance(r["aggregate_output_tokens_per_second"], (int, float))
            and math.isfinite(float(r["aggregate_output_tokens_per_second"]))
            and float(r["aggregate_output_tokens_per_second"]) > 0
        ]

    dedicated_median = _median(rates(dedicated))
    ladder_medians = {
        key: _median(rates(rows)) for key, rows in ladder.items() if rates(rows)
    }
    levels: Dict[str, Any] = {}
    for key, rows in ladder.items():
        occ = [r["occupancy"] for r in rows]
        levels[key] = {
            "true_concurrency": bool(occ)
            and all(o["true_concurrency"] for o in occ),
            "max_running_observed": max(
                (o["max_running_observed"] for o in occ), default=0.0
            ),
            "max_queue_observed": max(
                (o["max_queue_observed"] for o in occ), default=0.0
            ),
        }
    true_medians = {
        key: value
        for key, value in ladder_medians.items()
        if levels.get(key, {}).get("true_concurrency")
    }
    errors = sum(r["errors"] for r in dedicated) + sum(
        r["errors"] for rows in ladder.values() for r in rows
    )
    report = {
        "schema": "r0b0tlab.qwen38fn.w4a16_method.v1",
        "variant": args.variant,
        "base": args.base,
        "model": args.model,
        "method": {
            "thinking": "off",
            "streaming": True,
            "warmups": args.warmups,
            "dedicated_repeats": dedicated_repeats,
            "dedicated_max_tokens": args.dedicated_max_tokens,
            "ladder_repeats": repeats,
            "ladder_max_tokens": args.ladder_max_tokens,
            "concurrency": concurrencies,
        },
        "warmups": warmups,
        "dedicated": dedicated,
        "ladder": ladder,
        "summary": {
            "dedicated_c1_median": dedicated_median,
            "ladder_medians": ladder_medians,
            "ladder_medians_true_concurrency": true_medians,
            "true_concurrency_levels": sorted(true_medians, key=int),
            "queued_only_levels": sorted(
                [k for k, v in levels.items() if not v["true_concurrency"]], key=int
            ),
            "occupancy": levels,
            "ladder_geomean": _geomean(
                [float(v) for v in ladder_medians.values() if v is not None]
            ),
            "errors": errors,
        },
        "status": "PASS" if errors == 0 and dedicated_median and ladder_medians else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
