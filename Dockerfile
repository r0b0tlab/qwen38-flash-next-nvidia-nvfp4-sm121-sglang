# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS dependencies
USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MAX_JOBS=1 \
    FLASHINFER_NVCC_THREADS=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    PYTHONPATH="" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
WORKDIR /opt/r0b0tlab
COPY locks/ /opt/r0b0tlab/locks/
COPY wheelhouse/ /opt/r0b0tlab/wheelhouse/
COPY docker/verify_environment.py /opt/r0b0tlab/docker/verify_environment.py
# Remove only unused diffusion roots; keep the security-fixed Pillow and headless cv2.
RUN python3 -m pip uninstall -y moviepy cosmos-guardrail retinaface-py && \
    python3 -m pip install --no-deps --no-index --require-hashes \
        --find-links /opt/r0b0tlab/wheelhouse -r locks/python-overlay.lock
RUN python3 docker/verify_environment.py --phase prebuild

FROM dependencies AS builder
COPY build/sglang/ /src/sglang/
ARG BUILD_MAX_JOBS=2
ARG SGLANG_VERSION
# Build current native Rust extensions, rather than inheriting the old editable tree.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/src/sglang/rust/target \
    test -n "$SGLANG_VERSION" && \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" \
    SGLANG_BUILD_RUST_EXTS=all CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=/src/sglang/rust/target MAX_JOBS="$BUILD_MAX_JOBS" \
    python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels /src/sglang/python

FROM dependencies AS runtime
COPY --from=builder /wheels/ /tmp/sglang-wheels/
RUN python3 -m pip uninstall -y sglang && \
    python3 -m pip install --no-deps --no-index /tmp/sglang-wheels/sglang-*.whl && \
    rm -rf /tmp/sglang-wheels /opt/r0b0tlab/wheelhouse
COPY runtime/ /opt/r0b0tlab/runtime/
COPY profiles/ /opt/r0b0tlab/profiles/
COPY kv-calibration/ /opt/r0b0tlab/kv-calibration/
RUN python3 -c "from pathlib import Path; r=Path('/opt/r0b0tlab/kv-calibration'); paths=[r,*r.rglob('*')]; [p.chmod(0o555 if p.is_dir() else 0o444) for p in paths]"
COPY third_party/ /opt/r0b0tlab/third_party/
COPY docker/q200_sandbox_driver.py /opt/r0b0tlab/q200_sandbox_driver.py
COPY scripts/audit_checkpoint.py /opt/r0b0tlab/audit_checkpoint.py
RUN groupadd --gid 1001 runtime && \
    useradd --uid 1001 --gid 1001 --no-create-home --home-dir /cache/home --shell /usr/sbin/nologin runtime && \
    mkdir -p /cache/home /cache/hf /cache/jit /cache/xdg /cache/triton /cache/torch /cache/flashinfer /cache/tmp && \
    chmod 0700 /cache/tmp && \
    chown -R 1001:1001 /cache
ENV HOME=/cache/home \
    TMPDIR=/cache/tmp \
    XDG_CACHE_HOME=/cache/xdg \
    HF_HOME=/cache/hf \
    SGLANG_CACHE_DIR=/cache/jit \
    TRITON_CACHE_DIR=/cache/triton \
    TORCH_EXTENSIONS_DIR=/cache/torch \
    FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
    SGLANG_RUST_BUILD_MODE=never
USER 1001:1001
# This is a real default-user import/source/dependency check; GPU gates are separate.
RUN command -v nvcc && command -v ninja && command -v ffmpeg && \
    python3 docker/verify_environment.py --phase runtime
ARG WRAPPER_SHA
ARG SGLANG_MAIN_SHA
ARG SGLANG_TREE
ARG SGLANG_COMMIT
ARG MODEL_SHA
LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen38-flash-next-nvidia-nvfp4-sm121-sglang" \
      org.opencontainers.image.revision="${WRAPPER_SHA}" \
      io.r0b0tlab.sglang.main="${SGLANG_MAIN_SHA}" \
      io.r0b0tlab.sglang.tree="${SGLANG_TREE}" \
      io.r0b0tlab.sglang.commit="${SGLANG_COMMIT}" \
      io.r0b0tlab.model.sha="${MODEL_SHA}" \
      io.r0b0tlab.target="single-gb10-sm121"
EXPOSE 30000
STOPSIGNAL SIGTERM
ENTRYPOINT ["python3", "-m", "runtime.entrypoint"]
CMD ["--profile", "/opt/r0b0tlab/profiles/nextn.json", "--sources", "/opt/r0b0tlab/locks/sources.json"]
