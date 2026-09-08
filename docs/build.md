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
root override. The final build stage checks package version, source hashes,
all native Rust imports and writable cache locations as that default user.

After build: inspect exact image ID/labels; run native SM121 GPU microtests,
actual image CLI validation, model text/image/video and all NEXTN graph-phase
checks. Only then benchmark. The supplied 32K AR/NEXTN profiles are bring-up
controls, not selected optimal/final 262K profiles.
