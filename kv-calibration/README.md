# QSA NVFP4 KV calibration

Checkpoint-bound scale sidecars are generated from BF16 forward-pass observations, never guessed from default scales. A frozen profile names an embedded sidecar and its SHA256. Sidecars must match the exact model revision, target/draft global layer IDs, KV-head geometry and nonzero observation counts.

Collection profiles use BF16 KV. Reset the collector only while the scheduler is idle, collect the declared calibration corpus, export `qsa_kv_state` from server info, then freeze global scales as per-layer amax/(6*448). The publication image embeds the validated sidecar under this directory.
