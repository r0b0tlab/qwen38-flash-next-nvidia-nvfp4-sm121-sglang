# Native single-GB10 runtime package

## Selected runtime

- Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`
- Checkpoint revision: `fc694b54fb0174e0913e6adf86691ef85a4ead47`
- Image: `sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56`
- Native NVFP4 target weights/FlashInfer CUTLASS, native FP8 block-scaled integrated MTP/Triton, BF16 KV.
- Context and shared physical KV pool: 262144 tokens. Two concurrent requests share that pool; this is not two independent full-window budgets.
- NEXTN depth 3; CUDA decode graphs enabled; full vision retained.
- Mamba cache limit 10, chunked prefill 4096, static memory fraction 0.88, bounded file-backed PLE residency 4 GiB.
- Default container identity: UID/GID 1001:1001. The owned launcher enforces memory admission, immutable model/image binding, scoped shutdown, read-only mounts, and a private executable JIT directory.

This is the best verified balanced C1 configuration selected from the completed native screens, not a claim of a globally optimal runtime. Experimental NVFP4-KV profiles did not produce a consistent low-concurrency improvement. The Marlin model rollout did not qualify and is not included in the selected deployment.

## Verification status

Use `qualification-summary.json` for the final machine-readable state. Missing or pending fields are not passing results. Q200-v2 means the frozen text-180 corpus plus official BFCL v4 multi_turn_base structural-hard20, not GSM8K-200. Source build provenance and runtime qualification are separate.

**Profile digest equivalence.** `production-profile.json` (this package, file SHA-256
`e8c754d0…`) and the repository's `profiles/nextn-262k-c2-s3.json` (file SHA-256
`090b4f10…`) are the same qualified profile and differ only in JSON key order. Both reduce to
the identical canonical JSON digest `c8521cde67470848f608e79ee78ad74eca91d74dd85da50cd4ce3ed9db7166a3`.
Retrieval (NIAH) evidence rows bind the repository file hash `090b4f10…`; this package and its
summary bind `e8c754d0…`. Use the canonical digest above to reconcile the evidence chain.

The README inside `repro-source/` is preserved from the historical image-build commit. Its historical development status is not the current qualification report.

## Load the container

From this directory:

    sha256sum -c container.tar.zst.sha256
    zstd -dc container.tar.zst | docker load
    docker image inspect sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56 --format '{{.Id}} {{.Architecture}} {{.Config.User}}'

Expected architecture/user: `arm64 1001:1001`. Model weights are not embedded or redistributed.

## Start safely

Prerequisites: Linux ARM64 GB10/SM121, NVIDIA-enabled Docker, Python 3.10+, the unchanged checkpoint tree, and enough local NVMe for the model, prepared PLE table and reserves. Do not run alongside an existing serving workload on the same GPU.

    python3 launch.py --model-root /absolute/checkpoint \
      --cache-dir /absolute/dedicated-cache \
      --state-dir /absolute/new-attempt-state

Without a receipt pair, the launcher performs full-byte verification against the pinned inventory and saves a local operator receipt. This can take several minutes. For subsequent starts, provide the trusted receipt and the SHA256 retained from that successful verification:

    python3 launch.py --model-root /absolute/checkpoint \
      --cache-dir /absolute/dedicated-cache \
      --state-dir /absolute/new-attempt-state \
      --receipt /absolute/trusted-receipt.json \
      --receipt-sha256 VERIFIED_RECEIPT_SHA256

Do not calculate an approval digest from an unknown or edited receipt. Checkpoint, cache and state directories must be disjoint. Keep the foreground guard alive, or run it under a durable supervisor. API readiness is checked at `http://127.0.0.1:30080/v1/models` and must return the exact model ID above.

Stop only the owned launch:

    python3 repro-source/scripts/guard_stop.py --record /absolute/attempt-state/launch-record.json

Do not kill long-context clients while their server request is still in flight.

## Rebuild inputs

`repro-source/` is the exact wrapper commit `818c913cbd23687d6ae5aae1e5ba5f1e0a50af89` used to build the selected image. It contains the Dockerfile, source/dependency locks and immutable SGLang patch. `wheels/` contains the pinned overlay/build wheels. Use `scripts/prepare_build.py --help` in that source tree; provide a read-only SGLang Git checkout containing its locked base commit and this wheel directory. Rebuilt image IDs can differ; verify installed bytes and repeat hardware/model admission rather than relabeling archived results.

## Licenses and publication boundary

Retain the source licenses and upstream NVIDIA, Qwen, SGLang, FlashInfer and kernel-project attribution. The checkpoint remains subject to its NVIDIA/Qwen terms. This local package does not constitute a registry upload or public release. No benchmark or quality result is claimed unless its completed evidence is present in the qualification summary.
