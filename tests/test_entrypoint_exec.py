"""Public entrypoint regression tests; subprocess uses a fake executable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from runtime import ProfileError, profile_from_dict
from runtime import entrypoint as ep

SHA = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'
SOURCES = {'model': {'id': ep.MODEL_ID, 'sha': SHA}}
BASE = {'schema': 1, 'mode': 'ar', 'context_length': 32768,
        'max_total_tokens': 32768, 'vision': {'backend': 'triton_attn', 'cuda_graph': True}}


@pytest.mark.parametrize('concurrency', [1, 2, 4, 8])
def test_graph_buckets_are_real_nargs_integers_bounded_by_scheduler(concurrency):
    profile = profile_from_dict({**BASE, 'max_running_requests': concurrency})
    argv = ep.build_argv(profile, SOURCES)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cuda-graph-bs-decode', type=int, nargs='+')
    values, unused = parser.parse_known_args(argv)
    assert values.cuda_graph_bs_decode == [n for n in (1, 2, 4, 8) if n <= concurrency]


def test_explicit_bf16_kv_is_not_auto():
    argv = ep.build_argv(profile_from_dict(BASE), SOURCES)
    assert argv[argv.index('--kv-cache-dtype')+1] == 'bf16'


@pytest.mark.parametrize('sha', ['short', '../outside', 'a'*39, 'A'*40, 'a'*41, '', None])
def test_full_literal_model_revision_required(sha):
    with pytest.raises(ProfileError):
        ep.build_argv(profile_from_dict(BASE), {'model': {'id': ep.MODEL_ID, 'sha': sha}})


def test_source_env_cannot_disable_required_features():
    with pytest.raises(ProfileError):
        ep.build_env(profile_from_dict(BASE), {**SOURCES, 'env': {'SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK': '1'}})


def test_inconsistent_frozen_profile_is_rejected():
    profile = profile_from_dict(BASE)
    profile.raw['context_length'] = 262144
    profile.raw['max_total_tokens'] = 262144
    with pytest.raises(ProfileError):
        ep.build_argv(profile, SOURCES)


def test_module_really_executes_server_with_safe_environment(tmp_path):
    profile = tmp_path/'profile.json'
    sources = tmp_path/'sources.json'
    profile.write_text(json.dumps(BASE))
    sources.write_text(json.dumps(SOURCES))
    capture = tmp_path/'captured.json'
    executable = tmp_path/'sglang'
    executable.write_text('#!'+sys.executable+'\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE_PATH"]).write_text(json.dumps({"argv":sys.argv[1:],"env":{k:os.environ.get(k) for k in ["MAX_JOBS","FLASHINFER_NVCC_THREADS","SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK","SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH"]}}))\n')
    executable.chmod(0o700)
    env = {**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH'],
           'CAPTURE_PATH': str(capture), 'SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK': '1'}
    result = subprocess.run([sys.executable, '-m', 'runtime.entrypoint', '--profile', str(profile), '--sources', str(sources)], cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert capture.exists(), 'module returned without execing the server'
    data = json.loads(capture.read_text())
    assert data['argv'][0] == 'serve'
    assert data['argv'][data['argv'].index('--tp-size')+1] == '1'
    assert data['env'] == {'MAX_JOBS':'1','FLASHINFER_NVCC_THREADS':'1','SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK':'0','SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH':'0'}


def test_duplicate_source_keys_fail_before_exec(tmp_path):
    profile = tmp_path/'profile.json'; profile.write_text(json.dumps(BASE))
    sources = tmp_path/'sources.json'
    sources.write_text('{"model":{},"model":'+json.dumps(SOURCES['model'])+'}')
    result = subprocess.run([sys.executable, '-m', 'runtime.entrypoint', '--profile', str(profile), '--sources', str(sources)], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
