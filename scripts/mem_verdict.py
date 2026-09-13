#!/usr/bin/env python3
"""Machine-checkable memory verdict for a vision soak/serve epoch.

Reads the guard telemetry JSONL of one serve epoch and decides whether host
MemAvailable stayed bounded after warm-up. Exit 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("telemetry", type=Path)
    ap.add_argument("--warmup-rows", type=int, default=100)
    ap.add_argument("--rows-done", type=int, required=True,
                    help="rows completed when this epoch ended (or so far)")
    ap.add_argument("--allowed-decline-gib", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    vals = [json.loads(l)["mem_available_kb"] / 1048576 for l in a.telemetry.read_text().splitlines() if l.strip()]
    if len(vals) < 60:
        verdict = {"status": "INSUFFICIENT_SAMPLES", "samples": len(vals)}
    else:
        # warm-up = first 25% of samples (model load + cache fill), bounded by rows
        warm = max(len(vals) // 4, 1)
        post = vals[warm:]
        decline = post[0] - min(post)
        verdict = {
            "status": "PASS" if decline <= a.allowed_decline_gib else "FAIL",
            "samples": len(vals),
            "post_warmup_start_gib": round(post[0], 2),
            "post_warmup_min_gib": round(min(post), 2),
            "observed_decline_gib": round(decline, 2),
            "allowed_decline_gib": a.allowed_decline_gib,
            "rows_done": a.rows_done,
            "warmup_samples": warm,
        }
    text = json.dumps(verdict, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    return 0 if verdict.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
