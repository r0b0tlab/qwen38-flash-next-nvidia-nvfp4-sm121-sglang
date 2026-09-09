#!/usr/bin/env python3
"""Small cold native-API profile screen; never stop a runtime on client failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import guard

MODEL = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"


def validate_native(meta, n_input, cap, forced, cold, done):
    if not done or not isinstance(meta, dict):
        raise ValueError("missing terminal native SSE evidence")
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
        if type(meta.get(key)) is not int:
            raise ValueError(f"missing/non-integer native {key}")
    if meta["prompt_tokens"] != n_input:
        raise ValueError("server prompt count mismatch")
    n = meta["completion_tokens"]
    if not 0 < n <= cap or (forced and n != cap):
        raise ValueError("server completion count mismatch")
    if meta["cached_tokens"] < 0 or (cold and meta["cached_tokens"] != 0):
        raise ValueError("cold cache policy violated")
    reason = meta.get("finish_reason", {})
    expected = "length" if forced else "stop"
    if not isinstance(reason, dict) or reason.get("type") != expected:
        raise ValueError("abort, truncated prose, or missing finish reason")


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fetch(base, path):
    with urllib.request.urlopen(base + path, timeout=15) as r:
        return r.read()


def idle(base, timeout=30.0):
    # The final SSE can precede the scheduler's gauge update. Wait for observed
    # zero running AND queued requests; never reinterpret stale/missing as zero.
    deadline = time.monotonic() + timeout
    while True:
        text = fetch(base, "/metrics").decode()
        blocked = []
        for metric in ("sglang:num_running_reqs", "sglang:num_queue_reqs"):
            values = [float(line.split()[-1]) for line in text.splitlines()
                      if line.startswith(metric + "{") or line.startswith(metric + " ")]
            if not values or any(v != 0 for v in values):
                blocked.append(metric)
        if not blocked:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("endpoint not proven idle: " + ", ".join(blocked))
        time.sleep(0.25)


def identity(args):
    record = json.loads(args.record.read_text())
    if record.get("model") != MODEL or record.get("model_sha") != REVISION:
        raise RuntimeError("unexpected checkpoint identity")
    if record.get("image") != args.image:
        raise RuntimeError("unexpected image identity")
    doc = guard.verify_ownership(
        guard.DockerCliTransport().inspect(record["cid"]), record)
    if not doc["State"]["Running"]:
        raise RuntimeError("owned container is not running")
    models = json.loads(fetch(args.base, "/v1/models"))
    if [x["id"] for x in models.get("data", [])] != [MODEL]:
        raise RuntimeError("served model mismatch")
    info = json.loads(fetch(args.base, "/get_server_info"))
    keys = ("status", "version", "context_length", "max_total_num_tokens",
            "max_req_input_len", "max_running_requests", "max_mamba_cache_size",
            "mem_fraction_static", "kv_cache_dtype", "chunked_prefill_size",
            "speculative_algorithm", "speculative_num_steps",
            "speculative_num_draft_tokens", "speculative_draft_model_quantization",
            "moe_runner_backend", "speculative_moe_runner_backend",
            "cuda_graph_config", "mamba_ssm_dtype", "mm_attention_backend")
    result = {k: info.get(k) for k in keys}
    result["workers"] = [
        {k: w.get(k) for k in ("memory_usage", "effective_max_running_requests_per_dp",
                               "avg_spec_accept_length")}
        for w in info.get("internal_states", [])]
    return {"launch": record, "resolved": result}


def generate(args, name, ids, cap, forced, cold):
    payload = {"rid": name, "input_ids": ids, "stream": True,
               "sampling_params": {"max_new_tokens": cap, "temperature": 0.0,
                                   "top_p": 1.0, "top_k": -1,
                                   "ignore_eos": forced}}
    save(args.output / (name + ".request.json"), payload)
    request = urllib.request.Request(
        args.base + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    first_count = last_count = 0
    meta, text, done = None, "", False
    with (args.output / (name + ".sse")).open("xb") as wire:
        with urllib.request.urlopen(request, timeout=900) as response:
            for line in response:
                wire.write(line)
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    continue
                item = json.loads(data)
                meta = item.get("meta_info")
                text = item.get("text", text)
                n = meta.get("completion_tokens") if isinstance(meta, dict) else None
                if type(n) is int and n > last_count:
                    now = time.perf_counter()
                    if first is None:
                        first, first_count = now, n
                    last, last_count = now, n
    end = time.perf_counter()
    validate_native(meta, len(ids), cap, forced, cold, done)
    if first is None or last is None or not isinstance(meta, dict):
        raise ValueError("no complete observed output-token timeline")
    if not forced:
        words = re.findall(r"\b[\w'-]+\b", text)
        if len(words) < 300 or len(set(w.lower() for w in words)) < 100:
            raise ValueError("prose is too short or pathologically repetitive")
    row = {"name": name, "input_sha256": hashlib.sha256(
               json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
           "input_tokens": len(ids), "output_tokens": meta["completion_tokens"],
           "start": start, "end": end, "wall_s": end - start,
           "ttft_s": first - start,
           "e2e_output_tok_s": meta["completion_tokens"] / (end - start),
           "client_decode_tok_s": ((meta["completion_tokens"] - first_count) /
                                    (last - first) if last > first else None),
           "meta_info": meta, "text": text}
    save(args.output / (name + ".json"), row)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lane", choices=("short", "medium", "prose"), required=True)
    p.add_argument("--concurrency", type=int, choices=(1, 2, 4), required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--prose-max-tokens", type=int, choices=(2048, 4096, 8192), default=2048)
    args = p.parse_args()
    args.base = args.base.rstrip("/")
    if args.base.endswith("/v1") or args.repeats < 1:
        p.error("use the native base without /v1 and positive repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        before = identity(args)
        save(args.output / "identity-before.json", before)
        idle(args.base)
        raw = args.corpus.read_bytes()
        corpus = json.loads(raw)
        arrays = corpus["token_arrays"]
        if args.lane == "prose":
            selected = sorted(arrays["prose"].items())
            cap, forced = args.prose_max_tokens, False
        else:
            selected = [(str(i), ids) for i, ids in
                        enumerate(arrays[args.lane][:4])]
            length, cap = (512, 256) if args.lane == "short" else (2048, 512)
            if len(selected) != 4 or any(len(ids) != length for _, ids in selected):
                raise ValueError("unexpected frozen throughput corpus")
            forced = True
        save(args.output / "inputs.json", {
            "corpus_sha256": hashlib.sha256(raw).hexdigest(),
            "selected": selected, "lane": args.lane, "cap": cap,
            "forced_length": forced, "client_concurrency": args.concurrency,
            "policy": "warm kernels; flush once per measured round; native cache=0"})
        prefix = hashlib.sha256(str(args.output).encode()).hexdigest()[:12]
        # Warm the actual requested client concurrency; this is not a scored row.
        def warm(i):
            return generate(args, f"{prefix}-warm-{i}", arrays["short"][i], 32, True, False)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(warm, range(args.concurrency)))
        rows, batches = [], []
        for repeat in range(args.repeats):
            identity(args)
            idle(args.base)
            request = urllib.request.Request(
                args.base + "/flush_cache", data=b"{}",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError("cache flush failed")
                flush_body = response.read().decode()
            save(args.output / f"flush-{repeat}.json", {"body": flush_body})
            idle(args.base)
            def run(case):
                key, ids = case
                return generate(args, f"{prefix}-r{repeat}-{key}", ids,
                                cap, forced, True)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                batch = list(pool.map(run, selected))
            rows.extend(batch)
            batches.append(sum(r["output_tokens"] for r in batch) /
                           (max(r["end"] for r in batch) -
                            min(r["start"] for r in batch)))
            idle(args.base)
        after = identity(args)
        if after["launch"] != before["launch"]:
            raise RuntimeError("launch identity changed during probe")
        save(args.output / "identity-after.json", after)
        if len(rows) != len(selected) * args.repeats:
            raise RuntimeError("measured row count mismatch")
        summary = {"status": "NATIVE_SCREEN_PASS", "lane": args.lane,
                   "client_concurrency": args.concurrency,
                   "measured_requests": len(rows), "errors": 0,
                   "median_e2e_output_tok_s": statistics.median(
                       r["e2e_output_tok_s"] for r in rows),
                   "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
                   "median_batch_e2e_output_tok_s": statistics.median(batches),
                   "scope": "native controlled screen, not a model-quality score"}
        save(args.output / "summary.json", summary)
        print(json.dumps(summary))
        return 0
    except Exception as exc:
        failure = {"status": "PROBE_FAILED_RUNTIME_NOT_STOPPED",
                   "error": f"{type(exc).__name__}: {exc}"}
        save(args.output / "failure.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
