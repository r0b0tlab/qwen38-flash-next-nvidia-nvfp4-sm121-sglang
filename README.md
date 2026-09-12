# Qwen3.8-Flash-Next NVIDIA NVFP4 — Single GB10 SGLang

Status: **QUALIFIED single-GB10 release** — machine-readable verdict in
`releases/recovery-stable-20260909/qualification-summary.json`
(`RUNTIME_Q200_AND_RETRIEVAL_VERIFIED`).

Qualified on the exact pinned image `sha256:2ee545cf…427f56` + profile
`090b4f10…43b0f`:

- Full-window NIAH **9/9 PASS** (single-key 8K/32K/131K/258K at depths
  5–95% plus the ordered two-key 33/66 case at 258,044 prompt tokens),
  exact-token transport, one full-window case per serve epoch (host-memory
  admission), every attempt retained in per-epoch JSONL.
- Q200-v2 **185/200 = 92.5%** `SCORED_WITH_DISCLOSURES` — GSM8K 80/80,
  HumanEval 40/40, IFEval 36/40, hard reasoning 17/20,
  **BFCL-hard20 12/20 (60%)**, and two disclosed model nonterminations
  (ifeval-023, hard-01) that burned the 8,192-token ceiling with reasoning
  and emitted no final answer; scored as incorrect, not transport failures.
- Vision **11/11** (image / multi-image / video), semantic-checked.
- NEXTN acceptance gauge **3.3817** vs the W4A16 TP=2 derivative's 3.3830 —
  acceptance identical; the throughput difference is per-step cost.
- Honest performance attribution vs the W4A16 TP=2 derivative (identical
  method): dedicated c1 **37.32 vs 62.10** output tok/s (1.57–1.66×).
  Not claimed: multi-node, global optimality, NVFP4-KV serving.

Reproducible SGLang runtime for the unchanged `nvidia/Qwen3.8-Flash-Next-NVFP4`
checkpoint on exactly one NVIDIA GB10 (SM121, Linux ARM64): native W4A4 routed
experts, BF16 full vision, FP8 block-scaled integrated MTP, bounded file-backed
PLE on local NVMe, native decode CUDA graphs.

**No speed, quality, fit or correctness claim is made beyond the qualified
evidence in `releases/recovery-stable-20260909/qualification-summary.json`.**
Numbers on this page trace to that summary and the per-epoch JSONL evidence it
references; anything not present there remains unqualified.

Layout:

- `runtime/` — strict versioned launch profile contract and exact server argv
  compiler (`python -m`-style imports; no shell commands are ever generated).
- `scripts/audit_checkpoint.py` — structural header admission
  (`HEADERS_ADMITTED` is explicitly **not** full integrity verification).
- `scripts/verify_files.py` — streamed SHA-256 verification against the pinned
  inventory; emits a `CHECKPOINT_VERIFIED` receipt with per-file stat identity.
- `scripts/preflight.py` — fail-closed host admission (one idle GB10, RAM,
  per-filesystem disk with build/serve phases).
- `scripts/guard.py` + `scripts/run.sh` — owned launcher: receipt gate,
  hardened container spawn, watchdog, scoped cleanup.
- `tests/` — CPU-only pytest suite (`pytest` with a local venv; no GPU tests
  have been run — this is stated honestly rather than implied).

Model weights are not distributed with this repository or image and remain
subject to the NVIDIA Open Model License and applicable Qwen terms.

## Development launcher boundary (not serving qualification)

`sh scripts/run.sh` is a thin exec wrapper. Python owns TERM/INT and the
entire create/start/inspect/watch/stop lifecycle; it never kills a benchmark
client in place of stopping its server. Each launch uses a fresh private
`--state-dir`, a random ownership nonce, an immutable image config ID, and a
lifetime lock in the dedicated cache. State, cache and checkpoint directories
must be disjoint absolute canonical paths. Validated profile/source bytes
are copied into the attempt directory and mounted read-only.

Required arguments are `--image sha256:<64-hex-config-id>`, `--profile`,
`--sources`, `--model-root`, `--cache-dir`, `--state-dir`, `--receipt`, and
`--receipt-sha256`. Supply the last value from the successful full-byte
verifier's independently retained result. Do not calculate an approval hash
from an unknown or caller-edited receipt. Source locks and that expected
receipt digest are trusted operator inputs, not an authentication service
against another process running as the same host user. HF model revisions
are literal 40-hex commit IDs; file/image/receipt SHA256 values are 64 hex.

`--print` and `--dryrun` produce a plan without probing or mutating the host.
Real launches require host preflight plus pinned receipt/tree checks before
create and again before start. Container state, not a separate client poll,
determines exit. Unknown memory telemetry, <4 GiB available immediately, or
five consecutive <8 GiB samples stops the owned container. A healthy sample
resets the sustained counter. `--max-watch-seconds 0` means no deadline; a
positive value is a monotonic fail-closed deadline, not a success criterion.

Cleanup re-inspects CID, nonce, image and profile bindings before stop and
verifies the stopped state afterwards. Retain `launch-record.json`,
`container.final.json`, `preflight.json`, frozen input copies, telemetry,
and `server.log`. Docker logs rotate at 8 MiB × 3 files; final log capture
preserves both stdout and stderr. A benchmark owner must preserve startup
proof before rotation. Failed/uncertain cleanup retains the CID for explicit
recovery: `python3 scripts/guard_stop.py --record /absolute/attempt/launch-record.json`.
No automatic create/start retry or foreign-container deletion occurs.

Local exit classification: 2 usage, 3 preflight, 4 integrity/image binding,
5 spawn, 7 cleanup/evidence failure after otherwise successful exit, 8
watchdog/unknown running-state failure, 9 occupied cache lock. A real server
exit code is preserved; cleanup errors do not overwrite it. TERM/INT returns
128 + signal. These small integers can overlap server exit codes; inspect
the attempt record for provenance rather than inferring cause from the
number alone.

CPU tests exercise fake Docker/probe boundaries, including actual shell and
Python subprocess entrypoints. They do not constitute GPU, checkpoint
structural-audit, native-kernel, serving, or performance qualification.
