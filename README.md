# Qwen3.8-Flash-Next NVIDIA NVFP4 — Single GB10 SGLang

Status: NOT QUALIFIED — implementation in progress.

Reproducible SGLang runtime for the unchanged `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint on one NVIDIA GB10 (SM121, Linux ARM64).

The intended runtime preserves native W4A4 target experts, FP8 block-scaled MTP, full vision/video, native decode CUDA graphs and a bounded NVMe-backed PLE table. No speed, quality or fit claim is made until exact-artifact qualification is complete.

Model weights are not distributed with this repository or image and remain subject to NVIDIA Open Model License and applicable Qwen terms.
