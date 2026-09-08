# Checkpoint metadata fixtures

Source: `nvidia/Qwen3.8-Flash-Next-NVFP4` at
`fc694b54fb0174e0913e6adf86691ef85a4ead47`.

These are public model metadata, not tensor payloads. The repository's MIT
code license does not relicense NVIDIA model assets; the upstream NVIDIA
Open Model License and applicable Qwen terms remain authoritative.

- `config.json` and `hf_quant_config.json` are exact source bytes.
- `model.safetensors.index.json.gz` decompresses to the complete, byte-exact
  31,275,518-byte source index (299,545 weight-map entries). Deterministic
  gzip uses mtime=0; its compressed size is 859,340 bytes. No entries or
  whitespace were truncated.
- `mtp_ple.header.json` is a JSON-value-equivalent rendering of the 416,360-byte
  header in `model-fp8-mtp-ple.safetensors`, with 3,201 tensor entries.
- `shard10.header.json` is a JSON-value-equivalent rendering of the 2,312-byte
  header in `model-00010-of-00010.safetensors`, with 20 BF16 tensor entries.
- `provenance.json` records source scope, stored/decoded/source hashes and
  sizes, and exact-byte versus parsed-value equality for every capture.

The complete checkpoint references 11 safetensors files. Only the two
headers above are static fixtures; the other nine are not reconstructed or
claimed verified by a fabricated large checkpoint. CPU end-to-end tests use
small, explicitly synthetic two-shard checkpoints under a scaled test
contract. The production audit defaults to the pinned real contract and
must reject those small fixtures. Real on-disk header and whole-file hash
qualification are separate controller-owned evidence lanes.
