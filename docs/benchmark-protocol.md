# Bound qualification harness

Status: CPU/source/transport-fixture verification only. No results from this
harness are model qualification until the reviewed image is built, the owned
server passes native correctness/graph/vision gates, and these programs run
against that server. Unit-fixture throughput is never benchmark evidence.

## Protocol

- Same exact NVIDIA revision, image ID, source tree, tokenizer assets,
  context, physical token pool, C1 limit and request values on both sides.
- SHORT: eight distinct frozen 512-token arrays, 256 output tokens; five
  separate upstream rounds. MED: eight 2048-token arrays, 512 output tokens;
  five separate rounds. The first token differs across the eight arrays.
- Each upstream round warms once at 32 output tokens, flushes the owned
  server cache, then runs the eight requests. Every measured request must
  report zero cache hits. The upstream one-second settle sleep precedes its
  measured timer. Its measured duration includes the post-request server-info
  read. Keep this protocol identical on both sides.
- PROSE: three fixed cases, one separately recorded warmup per case, five
  measured repeats; warm-prefix policy. Do not call these cold-prefill rows.
  `--flush-cold` is diagnostic-only, not supported for bound promotion.
- VISION: all 43 cases, one recorded warmup per case, one measured envelope;
  includes changed-pixel, OCR, geometry, multi-image and video-order checks.
  Input/media hashes and semantic answers are validated, not just HTTP success.
- Prose and vision producers check the captured HTTP runtime identity before
  and after measured requests. Upstream rounds check it before and after the
  full round. Capture Docker identity on the host before and after each lane
  as well; a changed CID, owner nonce or start time invalidates that lane.
- Promotion requires all SHORT/MED/PROSE lanes to improve by at least 5%, no
  visual correctness loss, no more than 5% p95 visual TTFT regression, and an
  observed visual gain when a vision lever is selected. Changing an additional
  profile field invalidates a single-lever comparison.

## Runtime capacity observations

The configured context and physical token pool are separate from the maximum
prompt length. The pinned single-node SGLang worker reports
`max_req_input_len = min(context_length - 1, effective_pool - 1) - 5`.
Thus a full 32,768-token pool correctly reports a 32,762-token prompt limit.
The context capture requires the full configured context/pool, validates this
native reservation exactly, and records/rechecks the prompt limit through the
HTTP epoch binding. It does not lower serving capacity or waive an unexplained
prompt-limit shortfall. This host-side harness correction does not alter the
model image, profiles, kernels or sampling contract.

## Why the upstream client adapter exists

The pinned upstream OAI parser indexes `choices[0]` and otherwise defaults
output length to the requested cap. Its usage-only final SSE event has an
empty `choices` list. `scripts/upstream_capture.py` applies an exact-anchor,
client-process-only correction to that function: record the actual request
and SSE stream, accept the usage-only event, and use observed usage for native
aggregation. Missing usage, wrong model, wrong input, wrong finish, nonzero
cache hits or incomplete `[DONE]` fail the round. No model/server/kernel code
is changed. Raw wire files and upstream JSONL remain alongside bound rows.

The adapter uses the original upstream argument parser, dataset dispatch,
semaphore, warmup/flush, request timer and metrics aggregator. The output
contains both observed-token and upstream-retokenized totals; only observed
usage counts are the throughput numerator. Do not publish chunk-derived ITL
as token rate. `asyncio.gather` keeps input order; the adapter additionally
binds each actual payload by hash, not by equal lengths or assumed arrival order.

## Required operator inputs

Use a completed owned-launch record and the exact build's runtime lock. The
following variables must be set explicitly; no script selects an endpoint or
starts/stops a server on the operator's behalf:

- `ROOT`: this checkout, containing the reviewed runtime and harness code.
- `RECORD`: the owned server's launch record.
- `RUNTIME_LOCK`: the exact build's `locks/runtime.json`.
- `MODEL_DIR`: the unchanged local NVIDIA checkpoint.
- `EVIDENCE`: a fresh, private, UID/GID-writable evidence directory.
- `IMAGE_ID`: the exact qualified image config ID, `sha256:<64 hex>`.

Prepare the first AR side after the owner's correctness gates pass:

```sh
set -eu
: "${ROOT:?}" "${RECORD:?}" "${RUNTIME_LOCK:?}" "${MODEL_DIR:?}" "${EVIDENCE:?}" "${IMAGE_ID:?}"
mkdir -p "$EVIDENCE"
python3 "$ROOT/scripts/runtime_context.py" --record "$RECORD" \
  --runtime-lock "$RUNTIME_LOCK" --base http://127.0.0.1:30080 \
  --output "$EVIDENCE/context-before.json"
```

Run client programs in a CPU-only container of that image, with no GPU and no
Docker socket. `--network host` is needed to reach the owned loopback port;
the client does not bind a listening port. Do not use the serving cache for
client temporary files:

```sh
client() {
  docker run --rm --runtime=runc --network host --cpus 4 --memory 8g --memory-swap 8g \
    -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/client-hf -e XDG_CACHE_HOME=/tmp/client-cache \
    -e QUAL_HARNESS_VISION_ENABLED=1 \
    --mount "type=bind,src=$ROOT,dst=/harness,readonly" \
    --mount "type=bind,src=$RUNTIME_LOCK,dst=/runtime-lock.json,readonly" \
    --mount "type=bind,src=$MODEL_DIR,dst=/model,readonly" \
    --mount "type=bind,src=$EVIDENCE,dst=/evidence" \
    --entrypoint python3 "$IMAGE_ID" "$@"
}
client /harness/scripts/make_vision_fixtures.py --outdir /evidence/fixtures
client /harness/scripts/freeze_benchmark.py --context /evidence/context-before.json \
  --runtime-lock /runtime-lock.json --tokenizer-dir /model \
  --fixtures /evidence/fixtures --output /evidence/manifest.json
for lane in short medium; do
  for repeat in 0 1 2 3 4; do
    client /harness/scripts/run_upstream_round.py --manifest /evidence/manifest.json \
      --lane "$lane" --repeat "$repeat" --tokenizer-dir /model --allow-cold-flush \
      --output "/evidence/$lane-$repeat.jsonl"
  done
done
client /harness/scripts/bench_real.py --base http://127.0.0.1:30080 \
  --output /evidence/prose.jsonl --promotion-manifest /evidence/manifest.json
client /harness/scripts/vision_bench.py --base http://127.0.0.1:30080 \
  --output /evidence/vision.jsonl --promotion-manifest /evidence/manifest.json --variant AR
python3 "$ROOT/scripts/runtime_context.py" --record "$RECORD" \
  --runtime-lock "$RUNTIME_LOCK" --base http://127.0.0.1:30080 \
  --output "$EVIDENCE/context-after.json"
```

Compare the full before/after runtime context values; invalidate the lane on
any difference. After the owner cleanly replaces AR with the matched NEXTN
profile, use another fresh evidence directory and capture its own context and
manifest. Reuse the same fixture bytes. Do not change the source/image/tokenizer,
input arrays, pool or any unrelated profile setting. For a vision-only A/B,
use an explicit supported `--lever`, such as `vision.cuda_graph`, on both sides.

The reducer takes one JSONL file per upstream lane. Concatenate the five bound
round files in Python, preserving all records; do not concatenate their raw
`.upstream.jsonl` files instead. Run `scripts/compare.py --help` for the explicit
baseline/candidate, upstream, medium, vision and manifest arguments. Missing
lanes, duplicate repeat IDs and unbound rows cannot yield promotion PASS.

## Verification boundaries

- Host suite: all CPU contract tests, with explicitly skipped native-client
  tests when SGLang is absent and explicitly skipped live vision tests.
- Pinned CPU image: `test_upstream_native_integration.py` exercises the real
  upstream coroutine; `test_upstream_round_integration.py` exercises its real
  CLI, dataset path and metrics reducer. Only transport/tokenizer/runtime
  observations are unit doubles. These tests send no real model requests.
- Real-tokenizer construction: all 16 synthetic arrays, three post-template
  prose arrays and 43 visual request values, repeated for deterministic equality.
- Live qualification, GPU correctness and benchmark numbers remain separate,
  mandatory gates. An exact-SHA review is not approval for another tree.
