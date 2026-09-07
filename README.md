# Qwen3.8-Flash-Next NVIDIA NVFP4 — Single GB10 SGLang

Status: **NOT QUALIFIED — implementation in progress.**

Reproducible SGLang runtime for the unchanged `nvidia/Qwen3.8-Flash-Next-NVFP4`
checkpoint on exactly one NVIDIA GB10 (SM121, Linux ARM64): native W4A4 routed
experts, BF16 full vision, FP8 block-scaled integrated MTP, bounded file-backed
PLE on local NVMe, native decode CUDA graphs.

**No speed, quality, fit or correctness claim is made. Nothing here has been
qualified against the real image on real hardware. No performance numbers
exist for this repository.**

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
