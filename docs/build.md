# Development image build contract

Status: build inputs and the dependency preflight are verified. The candidate
image has NOT been built or GPU-qualified. Source/runtime quality approval is
required before the controller launches the native build. This document does
not authorize publishing or treating a build as an optimized result.

## Inputs

`locks/runtime.json` binds the official ARM64 parent, current-main-based SGLang
commit/tree and patch hash, all installed Python source hashes, custom package
version, native Rust module set, runtime UID/GID, and unchanged first-party
Q200 sandbox driver. `locks/dependencies.json` and `locks/python-overlay.lock`
bind the eight overlay/build wheels. The immutable parent pins everything else.

The source package is installed as a wheel, not an editable overlay. Its
Python path must resolve to site-packages, never the inherited
`/sgl-workspace` checkout. Native Rust extensions are built from the current
source with Cargo's locked manifests; they are not borrowed from the parent.
`SGLANG_RUST_BUILD_MODE=never` applies only to runtime rebuilds after those
extensions have been installed and import-checked.

## Prepare on the ARM64 build host

Use a separate upstream clone and wheel directory. Do not point either at a
model directory, another project, or an active worker checkout.

    git clone --filter=blob:none https://github.com/sgl-project/sglang.git /tmp/qwen38fn-upstream
    python3 -m pip download --only-binary=:all: --no-deps --require-hashes \
        -r locks/python-overlay.lock --dest /tmp/qwen38fn-wheels
    python3 scripts/prepare_build.py --upstream /tmp/qwen38fn-upstream \
        --wheel-dir /tmp/qwen38fn-wheels

The preparation command reconstructs the exact tree from locked main plus
`patches/sglang.patch`, verifies every tracked SGLang Python file and wheel,
and writes `build/preparation.json`. Existing `build/sglang` is deliberately
refused: use a fresh checkout/context rather than merging stale source files.
Model weights are never part of the Docker context.

## Packaging-only retry from a retained native wheel

The reconstructed source tree and patch retain all tracked source. The
SGLang package-source audit covers 3,625 Python files; the existing
`kernels/aot/` exclusion is the separately packaged sgl-kernel source subtree,
not an omission from source reconstruction. The installed-wheel audit omits
exactly two upstream `.claude` developer-skill helpers that setuptools does
not distribute; neither is SRT/runtime code.
The wheel contains all 3,623 required package source files byte-for-byte,
plus generated `_version.py` checked through installed package metadata.
Omitting any runtime source is still a hard failure.

If native compilation completed but a later image audit failed, preserve the
compiled wheel, its SHA256, source commit/tree, package version and builder
image evidence. `scripts/prepare_repackage.py` verifies those identities,
every packaged source hash and all four AArch64 Rust extensions, then derives
`build/Dockerfile.repackage` from the cold Dockerfile. Only the wheel COPY is
replaced and an exact checksum check added. BuildKit does not execute the
unreferenced native builder stage on this explicit packaging-only route:

    python3 scripts/prepare_repackage.py --wheel "$VERIFIED_WHEEL" --receipt "$COMPILED_ARTIFACT_RECEIPT"

Use the same admitted build arguments below, adding
`-f build/Dockerfile.repackage`, with a fresh attempt directory and current
wrapper revision. Do not reuse a wheel after changing its source, toolchain,
package version or native build inputs. The cold Dockerfile still rebuilds
the native extensions for the final clean-rebuild gate.

## Verified PLE startup

`locks/ple.json` records the exact checkpoint-derived FP8 table layout and
content hashes. The entrypoint binds it to `locks/sources.json` and the
fixed model geometry, then calls the installed SGLang preparation core before
starting the server. `--print` remains pure and neither imports the core nor
reads the PLE plan.

Preparation verifies the source and populated table with bounded reads.
An existing table is eligible for reuse only after full-content verification;
a reused filename or plausible size is insufficient. Cold population uses
bounded ordinary file I/O into a private temporary table, with space admission
for that complete temporary file and a reserve, followed by data/directory
synchronization and atomic publication. It does not overwrite the table via
an inference-oriented random-access mmap or disable the memory guard. Numeric
shard order is distinct from physical checkpoint order and must be preserved.

Native CUDA/HMM reads can update a shared table's modification/change times
without altering its bytes. For timestamp-only drift on the same device,
inode and size, preparation still verifies the full source and table contents
and requires unchanged source identity, then publishes a fresh receipt without
copying the table. Concurrent mutation fails closed. The native opener still
requires the exact fresh receipt-byte digest and current descriptor stat.

The fresh `receipt_path` and `receipt_sha256` returned by the core overwrite
both internal `R0B0TLAB_PLE_PREPARED_*` environment values before exec. Malformed
plans, source mismatches, preparation errors, stale caller-supplied handoffs,
and invalid return values cannot reach server exec. The native consumer must
verify that handoff and map the verified descriptor, retaining FP8 scale
loading, host-page-table checks, random-gather advice and RSS trimming.

The real legacy table was independently matched against all 128 canonical
checkpoint shards (table SHA256
`b070f9644adf93794d8a1030584ab705809387e64396a9327a68fa3a3a6666b3`).
This is integrity evidence, not optimized startup or serving qualification.
The earlier AR attempt remains failed on its multi-image semantic case; its
results cannot qualify a changed source tree or image. Real core/entrypoint
interoperability, cold/warm startup measurements, corrected ViT graph/cache
semantics and full model requalification are required before release.

## Initial matched memory profiles

The 32K/C1 AR and NEXTN profiles use `mem_fraction_static=0.83`, retaining
BF16 KV, FP32 recurrent state, Mamba16 capacity, native precision, graphs and
all host guards. The earlier NEXTN 0.80 epoch loaded target and FP8 draft but
failed KV sizing after reserving persistent and speculative recurrent state.
The 0.83 value is a candidate derived from that budget, not proof of readiness
or final-context capacity; require fresh live allocation and semantic gates.

## Native build, only after review/resource admission

Run under one durable build owner with a fresh log/evidence directory. The
following Python generates and executes the complete command from the locks;
it does not substitute moving tags or derive identities from image labels.

```python
import json
import subprocess
from pathlib import Path

runtime = json.loads(Path("locks/runtime.json").read_text())
model = json.loads(Path("locks/sources.json").read_text())["model"]
prepared = json.loads(Path("build/preparation.json").read_text())
assert prepared["status"] == "BUILD_INPUTS_VERIFIED"
assert prepared["source_tree"] == runtime["sglang"]["tree"]
assert not subprocess.check_output(["git", "status", "--porcelain"]).strip()
wrapper = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
values = {
    "BASE_IMAGE": runtime["base_image"],
    "WRAPPER_SHA": wrapper,
    "SGLANG_MAIN_SHA": runtime["sglang"]["main"],
    "SGLANG_TREE": runtime["sglang"]["tree"],
    "SGLANG_COMMIT": runtime["sglang"]["commit"],
    "SGLANG_VERSION": runtime["sglang"]["package_version"],
    "MODEL_SHA": model["sha"],
    "BUILD_MAX_JOBS": "2",
}
command = ["docker", "buildx", "build", "--platform", "linux/arm64", "--load", "--progress=plain"]
for key, value in values.items():
    command += ["--build-arg", f"{key}={value}"]
command += ["-t", "qwen38fn-nvidia-sm121:candidate", "."]
subprocess.run(command, check=True)
```

The native build is isolated in its own cacheable stage. Runtime JIT remains
at `MAX_JOBS=1`, independent of the two build jobs. CUDA headers, nvcc, ninja,
FFmpeg and a compiler stay available for the model's native JIT paths.

## Dependency exceptions and image checks

Keep security-fixed Pillow 12.3.0. The removed packages are unused optional
diffusion/video-editing roots, not the SRT image/video processor. Protobuf and
its generator are pinned to a mutually compatible release pair. The checker
requires every other dependency to satisfy installed metadata.

The upstream Dockerfile intentionally keeps NCCL 2.30.7 for DeepEP despite
PyTorch's 2.29.7 metadata pin, and uses NIXL's CUDA-13 binary despite its stub
requiring CUDA-12. Pip also does not recognize NVIDIA's `manylinux2014_sbsa`
cuSparseLT tag. The checker matches those exact three pip lines, verifies the
two dependency exceptions and the actual AArch64 ELF library separately, and
rejects additional failures. It does NOT claim a clean zero-exception
`pip check`, nor does an AArch64 ELF prove SM121 GPU execution.

The runtime account is UID/GID 1001, matching the admitted host. Its default
user must read the host's private control snapshots and write the dedicated
cache; do not solve a permission error with `--privileged`, broad chmod, or a
root override. `TMPDIR=/cache/tmp` is owned by the image/entrypoint and created
as a private runtime-owned directory before server exec. The general `/tmp`
tmpfs stays non-executable; TileLang/TVM and other native JITs use `/cache/tmp`
for generated libraries that must be executable-mapped. The runtime audit
compiles and loads a tiny shared library there rather than checking writability
alone. Exclude process-specific temp files from any reusable JIT cache seed.
The final build stage checks package version, source hashes,
all native Rust imports and writable cache locations as that default user.

After build: inspect exact image ID/labels; run native SM121 GPU microtests,
actual image CLI validation, model text/image/video and all NEXTN graph-phase
checks. Only then benchmark. The supplied 32K AR/NEXTN profiles are bring-up
controls, not selected optimal/final 262K profiles.
