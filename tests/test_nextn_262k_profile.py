import json
from pathlib import Path
from runtime import validate_payload
from runtime.entrypoint import build_argv

ROOT = Path(__file__).resolve().parents[1]


def test_full_window_candidate_is_explicit_and_native():
    data = validate_payload((ROOT / 'profiles/nextn-262k-c2-s3.json').read_text())
    assert data == {
        'schema': 1, 'mode': 'nextn', 'context_length': 262144,
        'max_total_tokens': 262144, 'max_running_requests': 2,
        'chunked_prefill_size': 4096, 'max_mamba_cache_size': 10,
        'mem_fraction_static': .88, 'kv_cache_dtype': 'bf16',
        'vision': {'backend': 'triton_attn', 'cuda_graph': True},
        'ple_rss_gib': 4, 'mm_processor_worker_num': 1,
        'speculative': {'steps': 3},
    }


def test_full_window_argv_keeps_graphs_and_native_mixed_draft():
    from runtime import load_profile
    profile = load_profile(str(ROOT / 'profiles/nextn-262k-c2-s3.json'))
    sources = json.loads((ROOT / 'locks/sources.json').read_text())
    argv = build_argv(profile, sources)
    assert argv[argv.index('--context-length') + 1] == '262144'
    assert argv[argv.index('--max-total-tokens') + 1] == '262144'
    assert argv[argv.index('--speculative-num-steps') + 1] == '3'
    assert argv[argv.index('--speculative-num-draft-tokens') + 1] == '4'
    assert argv[argv.index('--cuda-graph-backend-decode') + 1] == 'full'
    assert argv[argv.index('--speculative-draft-model-quantization') + 1] == 'modelopt_mixed'
    assert argv[argv.index('--speculative-moe-runner-backend') + 1] == 'triton'
    assert argv[argv.index('--mamba-ssm-dtype') + 1] == 'float32'
