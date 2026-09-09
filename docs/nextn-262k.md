# 262K native NEXTN candidate

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.

Profile: `profiles/nextn-262k-c2-s3.json`.
Existing local image: `sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56`.
Engine source: `84cf99860e3086ee0a458a71178343a8ac04fdab`.

## Implemented configuration

- 262144 context and physical token pool; engine prompt limit 262138.
- Effective maximum requests2; bundled native NEXTN/EAGLE steps3, draft tokens4.
- Native NVFP4 target/CUTLASS and native block-128 FP8 MTP/Triton, unchanged checkpoint.
- BF16 KV, FP32 recurrent SSM state, Mamba cache10, static fraction0.88.
- Prefill chunks4096; PLE resident budget4 GiB; full vision retained.
- Full target-verify, draft-decode and draft-extend graphs; decode batch buckets1,2.

This is a working, screened full-pool candidate, **not a claim of global optimality or long-context quality qualification**.

## Measured native screen

22 measured requests across six lanes, zero admitted request errors. Cold native cached-token counters were zero. SHORT is512/256 input/output; MEDIUM is2048/512. PROSE uses the three frozen natural-language prompts, thinking off, natural stop, and an explicit4096-token budget. Failed2048-budget prose captures remain excluded rather than relabeled successful.

| Lane | C1 median request E2E tok/s | C2 aggregate batch E2E tok/s | Measured requests C1 / C2 |
|---|---:|---:|---:|
| SHORT | 21.35 | 42.09 | 4 / 4 |
| MEDIUM | 23.83 | 26.32 | 4 / 4 |
| PROSE | 26.43 | 36.31 | 3 / 3 |

C1 request latency and C2 batch throughput are different metrics. They are labeled separately, not presented as interchangeable speedups. The C1 SHORT median is 6.74% below the earlier four-request32K NEXTN-S1 screen; C1 acceleration is still an open optimization goal.

Allocated target/index KV was reported as6.189 GiB, with14.630 GiB startup-available memory; this is not a peak-memory guarantee. Two post-ready generation canaries passed. Host regressions passed590 tests with5 declared skips. The fixed vision/video/multi-image suite passed11 measured cases plus11 warmups with no semantic or transport errors at this envelope. Full-window retrieval, Q200, and sustained stability remain unrun here.

## Run using the existing guard

Reuse the existing verified checkpoint receipt and local image. Set these paths for your machine:

```sh
export PROJECT="$PWD"
export MODEL_ROOT="/path/to/nvidia/Qwen3.8-Flash-Next-NVFP4"
export CACHE_DIR="/path/to/existing/verified/cache"
export RECEIPT="/path/to/verified/checkpoint-receipt.json"
export RECEIPT_SHA="<the verified receipt SHA256>"
export STATE="$PROJECT/.hermes/evidence/nextn-262k-$(date -u +%Y%m%dT%H%M%SZ)"
python3 scripts/guard.py \
  --image sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56 \
  --profile profiles/nextn-262k-c2-s3.json \
  --model-root "$MODEL_ROOT" --cache-dir "$CACHE_DIR" \
  --sources locks/sources.json --receipt "$RECEIPT" \
  --receipt-sha256 "$RECEIPT_SHA" --state-dir "$STATE" --max-watch-seconds 0
```

Run the guard in a durable tmux pane or service. The API is bound to `127.0.0.1:30080`; do not launch alongside another GPU owner. A fresh state directory is required for every attempt. Wait for an old guard's cache lease to close after its container stops; do not force-unlock or reuse an empty failed directory.

To stop only the selected owned epoch:

```sh
python3 scripts/guard_stop.py --record "$STATE/launch-record.json"
```

Do not stop/reload a healthy service because a client parser, output budget, or immediately stale scheduler gauge rejected a measurement. The native probe waits for observed idle and preserves the raw request/SSE evidence.

The original `profiles/nextn.json`32K/S1 and `profiles/ar.json` remain unchanged for explicit rollback. A shared262144-token pool does not provide two independent262144-token request budgets.

## NVFP4 KV investigation

The installed image's real `nvfp4_kv_quantize` and `nvfp4_kv_dequantize` primitives passed on SM121, including three CUDA graph replays into BF16 output. This is component proof, not a full-model KV-mode result.

The live profile still uses BF16 KV. Native NVFP4 storage would require scale-aware QSA prefix and selected-row reads into BF16 compute scratch. The checkpoint declares no KV quantization and provides no named k_scale/v_scale tensors, so per-layer calibration provenance and quality require validation. Estimated saving is approximately4.2 GiB with the existing dequant workspaces, potentially4.7 GiB with bounded QSA-specific scratch; these are sizing estimates, not observed allocation or speed gains.

Private local evidence: `.hermes/evidence/nextn-262k-c2-s3-fast/verified-screen-summary.json` and `.hermes/evidence/nvfp4-kv-research/`. No image or model upload is implied by this document.

The standalone native MTP tuning probe was not admitted: duplicating its weights and TP1 context beside the live full-window model crossed the probe's9 GiB headroom reserve. It exited without changing the service. No unmeasured tuning table was installed and no reserve was lowered; further full-MoE tuning needs an admitted idle-GPU window.
