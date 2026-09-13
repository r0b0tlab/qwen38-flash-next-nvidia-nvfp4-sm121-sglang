# Qwen3.8-Flash-Next NVIDIA NVFP4 — Single GB10 SGLang

Native **NVIDIA Qwen3.8-Flash-Next-NVFP4** (checkpoint revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, unmodified) served by SGLang on
**one NVIDIA GB10 (SM121, Linux ARM64, 128 GB unified memory)**: native NVFP4
W4A4 routed experts (FlashInfer CUTLASS), FP8 block-scaled integrated MTP
(NEXTN, steps=3), BF16 full vision (image / multi-image / video), bounded
file-backed PLE embedding offload, full decode CUDA graphs, 262,144-token
context and shared physical KV pool on one device.

Machine-readable verdict:
[`releases/recovery-stable-20260909/qualification-summary.json`](releases/recovery-stable-20260909/qualification-summary.json)
— status `RUNTIME_Q200_AND_RETRIEVAL_VERIFIED`.

## Click-run

Prerequisites: a GB10-class machine (SM121 / DGX Spark, Linux ARM64), NVIDIA
Docker, ~200 GB free NVMe for image + checkpoint + PLE + caches, no other
workload on the GPU. Model weights are **not** included (NVIDIA / Qwen license
terms); download them once from Hugging Face:

```bash
# 1. Image (public, anonymous pull verified; 33.9 GB compressed)
docker pull ghcr.io/r0b0tlab/qwen38-flash-next-nvidia-nvfp4-sm121-sglang:v1.0.0-sm121-nextn

# 2. Checkpoint (~132.7 GB, 25 files)
pip install -U "huggingface_hub[cli]"
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --local-dir /absolute/path/Qwen3.8-Flash-Next-NVFP4

# 3. Full-byte checkpoint verification (several minutes; produces the receipt
#    the launcher demands). Run from a clone of this repo:
python3 scripts/verify_files.py \
  /absolute/path/Qwen3.8-Flash-Next-NVFP4 locks/sources.json \
  --receipt /absolute/path/receipt.json

# 4. Launch (owned lifecycle: preflight, receipt gate, watchdog, scoped stop)
sh scripts/run.sh \
  --image sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56 \
  --profile profiles/nextn-262k-c2-s3.json \
  --sources locks/sources.json \
  --model-root /absolute/path/Qwen3.8-Flash-Next-NVFP4 \
  --cache-dir /absolute/path/cache \
  --state-dir /absolute/path/attempt-state \
  --receipt /absolute/path/receipt.json \
  --receipt-sha256 <SHA256-PRINTED-BY-STEP-3>

# 5. Wait for readiness (~9–10 min: weight load ~9 min, graph capture seconds),
#    then verify identity:
curl -s http://127.0.0.1:30080/v1/models
#   expected: "id":"nvidia/Qwen3.8-Flash-Next-NVFP4","max_model_len":262144

# 6. Chat (OpenAI-compatible; thinking is on by default for this model)
curl -s http://127.0.0.1:30080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
  "messages": [{"role": "user", "content": "What is 17*23? Reply with the number only."}],
  "max_tokens": 256, "temperature": 0}'

# 7. Stop exactly what you started:
python3 scripts/guard_stop.py --record /absolute/path/attempt-state/launch-record.json
```

Alternative distribution: the exact same image as a 14.6 GB zstd tar in the
[`v1.0.0-sm121-nextn` release](https://github.com/r0b0tlab/qwen38-flash-next-nvidia-nvfp4-sm121-sglang/releases/tag/v1.0.0-sm121-nextn)
(`container.tar.zst` + `container.tar.zst.sha256`;
`zstd -dc container.tar.zst | docker load`). Rebuild-from-source inputs
(`repro-source.tar`, pinned wheels) ship with the release.

Smoke expectation: the canary above returns `391`; a first-token sanity reply
returns cleanly with `finish_reason: stop`.

## Identity chain (what "qualified" binds to)

| Surface | Value |
|---|---|
| Image (config ID) | `sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56` (arm64, runs as UID/GID 1001:1001) |
| GHCR manifest digest | `sha256:df283f83acd9723d7cc9227b4d0536bf9a77026367083bf06a4fa4c6fb556397` (tag `v1.0.0-sm121-nextn`) |
| SGLang source | `84cf99860e3086ee0a458a71178343a8ac04fdab` (patched; base `20ca564b…`, v0.5.19-era nightly) |
| Launch wrapper commit | `818c913cbd23687d6ae5aae1e5ba5f1e0a50af89` |
| Model revision | `fc694b54fb0174e0913e6adf86691ef85a4ead47` (full-byte receipt-verified, 25 files) |
| Production profile | `profiles/nextn-262k-c2-s3.json` — repo bytes sha256 `090b4f101f154fc02f8ea49959819f0f6ffe2decfb81e4257508b5b804cf3b0f`; the packaged `production-profile.json` is the same profile with different key order (canonical-JSON digest `c8521cde67470848f608e79ee78ad74eca91d74dd85da50cd4ce3ed9db7166a3`) |
| Repo at qualification | main `7d265ca7373e7abbcbc798188e973a2dd8781847`, CI green at that SHA; anonymous clean clone: 671 passed / 5 skipped |

## Serving profile (exact resolved server configuration)

| Knob | Value |
|---|---|
| Context length / KV pool | 262,144 / 262,144 tokens (`max_req_input_len` 262,138) |
| Concurrency | `max_running_requests` 2 (shared pool — two requests share 262k tokens, not two independent 262k budgets) |
| Precision | BF16 activation dtype; NVFP4 W4A4 routed experts (`fp4_gemm_backend=flashinfer_cutlass`, `moe_runner=flashinfer_cutlass`); **BF16 KV** |
| Speculative | NEXTN (EAGLE-family) steps=3, top-k 1, draft tokens 4; draft is native FP8 block-scaled `modelopt_mixed`, draft KV BF16 |
| CUDA graphs | full decode graphs, batch sizes [1, 2]; prefill graphs disabled |
| Attention / linear attn / Mamba | triton (all); `mamba_ssm_dtype=float32`, mamba cache 10, track interval 256 |
| PLE embedding offload | file-backed, 4 GiB RSS budget (`/cache/ple/<model-sha>`) |
| Memory admission | `mem_fraction_static` 0.88, chunked prefill 4096 |
| Page size | 64 |
| Vision | triton mm attention, 1 mm processor worker; ViT CUDA graphs **bounded** (LRU cap `SGLANG_VIT_MAX_GRAPHS`, default 16; see v1.0.1 note) — vision benchmarking and sustained mixed-shape vision traffic run with graphs off (`profiles/vision-novitgraph.json`) because each retained large-image graph holds a ~1 GiB private pool on the unified GB10 memory |

## Measured host/GPU memory & startup (post-suite server telemetry)

| Metric | Value |
|---|---|
| Weights (device) | 83.68 GiB |
| KV cache (device, 262,144 tokens) | 6.19 GiB |
| CUDA graphs (target verify + draft) | 0.335 GiB (0.150 + 0.111 + 0.074) |
| Startup-available (device, after init) | 15.29 GiB |
| Host PLE table on NVMe | 47.68 GiB (4 GiB RSS budget at runtime) |
| Weight load | 528–546 s (mmap off); KV allocation 2.1 s; ready in ~9.5 min |
| Checkpoint on disk | 132.7 GB (123.6 GiB), 25 files, full-byte receipt-verified |
| Image on disk | 33.9 GB compressed manifest / 14.6 GB zstd export tar |

## Qualification results (this exact image + profile)

### Long-context retrieval — NIAH 9/9 PASS (exact-token transport)

All at temperature 0, `max_tokens` 4096, generous 43,200 s timeout, prompts
POSTed as exact token IDs against the real checkpoint tokenizer. One
full-window case per serve epoch (see "Known limits"): a second consecutive
258k-token prefill on the same epoch legitimately walks host memory toward the
watchdog floor.

| Case | Prompt tokens | Depth(s) | Verdict | Wall |
|---|---|---|---|---|
| single_8192_d50 | 8,192 | 50% | PASS | 5.8 s |
| single_32768_d50 | 32,768 | 50% | PASS | 13.6 s |
| single_131072_d50 | 131,072 | 50% | PASS | 51.7 s |
| single_258044_d5 | 258,044 | 5% | PASS | 130.1 s |
| single_258044_d25 | 258,044 | 25% | PASS | 130.9 s |
| single_258044_d50 | 258,044 | 50% | PASS | 130.6 s |
| single_258044_d75 | 258,044 | 75% | PASS | 132.4 s |
| single_258044_d95 | 258,044 | 95% | PASS | 131.9 s |
| multi_258044_d33_66 | 258,044 | 33% + 66% ordered | PASS | 134.8 s |

Pass = final answer is exactly the ordered pass-code list after thinking;
server-echoed `prompt_tokens` matches the constructed count exactly.

### Quality — Q200-v2: 185/200 = 92.5% (`SCORED_WITH_DISCLOSURES`)

Frozen text-180 corpus + official BFCL v4 `multi_turn_base` structural-hard20
(graded by `bfcl-eval==2025.12.17`). All 200 transported and graded, zero
grader errors.

| Family | Result |
|---|---|
| GSM8K (80) | **80/80** (100%) |
| HumanEval (40) | **40/40** (100%) |
| IFEval (40) | 36/40 (90%) |
| Hard reasoning (20) | 17/20 (85%) |
| BFCL-hard20 (20) | **12/20 (60%)** — weakest lane, disclosed up front |
| **Total (200)** | **185/200 (92.5%)** |

Disclosures: (1) two model nonterminations (`ifeval-023`, `hard-01`) burned the
full 8,192-token output ceiling on reasoning and emitted an empty final answer;
scored incorrect, not transport failures. (2) Hard-reasoning answers were
independently adjudicated; a known erroneous reference answer was not allowed
to mark correct math wrong. (3) Single-GPU C2 server, mixed quality cohort —
no throughput claim from this run.

### Vision — 11/11 PASS (semantic-checked, not just HTTP success)

8 image (color/geometry/OCR/counting), 1 multi-image, 2 video order-tracking;
median TTFT 2.88 s. Input/media hashes validated.

### Vision benchmark — r0b0bench-vision v1.0 (4,703 rows, 4 suites)

Frozen public contract (`r0b0tlab/r0b0bench` `scripts/vision/`; pinned dataset
revisions, deterministic graders, thinking-off, temperature 0, single image per
request). Two result sets:

| Suite | v1.0.0 image (graphs off, 3 serve epochs) | v1.0.1 image (graphs off, single epoch) |
|---|---|---|
| cvbench (2,638) | 87.9% | see `results/vision/r0b0bench-vision-v1-summary-v101-image.json` |
| mmvp (300) | 83.0% (paired 69.3%) | " |
| realworldqa (765) | 80.4% | " |
| ocrbench (1,000) | 85.9% | " |
| **total (4,703)** | **86.0%** | " |

Protocol notes: `--workers 2` (matches `max_running_requests` 2; the contract's
default 4 assumes ≥4), MMVP's paired metric is directional (±8 pp at p≈0.5),
RealWorldQA is CC-BY-ND (aggregates only, no images redistributed). The v1.0.0
numbers were assembled across three serve epochs with identical
protocol/loaders/graders — full disclosure in the summary JSON's `assembly`
and `defects` blocks (`results/vision/`).

**v1.0.1 vision-serving fixes** (why the image was rebuilt): upstream
`ViTCudaGraphRunner` retained an unbounded per-image-shape graph cache; real
benchmark traffic (185 distinct image sizes in the first 400 cvbench rows)
drove ~48 MiB/row of permanent unified-memory growth until the safety watchdog
stopped the serve. The shipped patch adds an LRU budget
(`SGLANG_VIT_MAX_GRAPHS`, default 16) with an in-flight replay refcount so a
live replay is never evicted. With graphs ON the budget now holds — but each
retained large-image graph keeps a ~1 GiB private pool, so bounded-16 still
costs ~20 GiB on a 128 GiB unified node; sustained mixed-shape vision work
therefore ships graphs OFF (`profiles/vision-novitgraph.json`, byte-identical
to the production profile except `vision.cuda_graph`). The 4 GiB
`MultiModalStaticCache` behaves as a configured bound, not a leak.

### Throughput & speculative decoding

| Measurement | This runtime (single GB10) | W4A16 TP=2 derivative (2× GB10, Marlin) |
|---|---|---|
| Dedicated c1 (2048 out, ×5 median) | **37.32 tok/s** | 62.10 tok/s |
| Ladder c1 / c2 / c4 (512 out, ×3 median) | 40.67 / 53.69 / 54.06 | 63.84 / 83.66 / 132.64 |
| NEXTN acceptance length (server gauge) | **3.3817** | 3.3830 |
| ms per verify step | ~54.5 ms implied | derived from 3.383 / 62.10 |

Identical frozen method on both sides (16 prompts, thinking off, temp 0,
streaming). Read honestly: **this runtime is 1.57–1.66× slower than the 2×GB10
W4A16 Marlin derivative at identical acceptance** — the entire gap is per-step
cost, and the often-quoted "3×" was a SHORT-lane method artifact. Short-lane
request-E2E screen (separate measurement, incl. prefill/TTFT): C1
SHORT/MED/PROSE 21.35 / 23.83 / 26.43 tok/s per request; C2 batch aggregate
42.09 / 26.32 / 36.31 tok/s. What this runtime buys: the whole 262k pool,
vision, and FP8-native MTP on **one** device.

### Test suites

Packaged source tests at build: 558 passed / 4 skipped. Repo test suite at
publication SHA: 671 passed / 5 skipped (CPU-only — fake Docker/probe
boundaries; explicitly not GPU qualification). CI runs the suite on every push.

## Known limits (disclosed, not hidden)

- One 258k-token prefill per serve epoch. During full-window prefill the host
  dips to ~7 GiB `MemAvailable`; a second consecutive full-window prefill walks
  it ~2 GiB lower and the owned watchdog will stop the server (guard rc=8).
  The watchdog floor is operator-tunable via
  `--mem-available-floor-gib` (default 8, clamp [4, 16]); any override is
  written into `launch-record.json`.
- Marlin FP4 model rollout failed closed on this checkpoint (oracle image):
  not used, not benchmarked, not shipped.
- NVFP4 KV / QSA: primitive + graph replay proven on SM121, but the checkpoint
  ships no KV quantization scales and QSA integration is **not** enabled —
  KV stays BF16.
- Not claimed: multi-node serving, globally optimal configuration, long-context
  *reasoning* quality (NIAH is functional retrieval evidence), redundancy/HA.
- No Windows/macOS; no SM120 consumer-GPU recipe for this NVFP4 checkpoint
  (the W4A16 sibling project covers dual-RTX PRO 6000).

## Repository layout

- `runtime/` — strict versioned launch-profile contract (unknown-field,
  enum- and range-checked) and the exact server argv compiler; no shell
  commands are ever generated.
- `profiles/` — frozen launch profiles (`nextn-262k-c2-s3.json` is production).
- `scripts/guard.py` + `run.sh` — owned launcher: preflight, receipt gate,
  hardened spawn (read-only mounts, caps dropped, UID 1001:1001), watchdog
  with claim-bound memory floor, scoped cleanup.
- `scripts/verify_files.py` — streamed full-byte checkpoint verification →
  `CHECKPOINT_VERIFIED` receipt; `scripts/audit_checkpoint.py` — structural
  header admission (explicitly **not** integrity verification).
- `scripts/preflight.py` — fail-closed host admission (idle GB10, ≥104 GiB
  host RAM available, per-filesystem disk budgets).
- `scripts/niah.py` — exact-token NIAH harness (fail-closed verdicts,
  bounded infra-only retries, per-attempt rows).
- `releases/recovery-stable-20260909/` — the qualified package: summary,
  evidence (performance/vision/q200 JSON), launch helper, checksums.
- `tests/` + `.github/workflows/ci.yml` — CPU-only pytest suite in CI.

## Launcher contract (short version)

`sh scripts/run.sh` requires `--image sha256:<config-id>`, `--profile`,
`--sources`, `--model-root`, `--cache-dir`, `--state-dir`, `--receipt`,
`--receipt-sha256` (last = SHA256 of the receipt from step 3 of click-run).
Checkpoint, cache and state dirs must be disjoint absolute canonical paths.
`--print`/`--dryrun` plans without touching the host. Exit classes: 2 usage,
3 preflight, 4 integrity/binding, 5 spawn, 7 cleanup, 8 watchdog, 9 lock;
server exit codes are preserved; TERM/INT → 128+signal. Cleanup retains the
CID on failure for `scripts/guard_stop.py --record <launch-record.json>`.
No automatic retry; no foreign-container deletion.

## Licenses

Code: MIT (see `LICENSE`). Upstream SGLang (Apache-2.0) attribution in
`third_party/` and `NOTICE`. The checkpoint remains subject to the NVIDIA Open
Model License and applicable Qwen terms; weights are not redistributed here.
