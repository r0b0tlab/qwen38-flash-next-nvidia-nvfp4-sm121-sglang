import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from runtime import ProfileError, profile_from_dict, profile_from_json
from runtime.entrypoint import build_env, build_argv, ENV_VARS_OWNED

ROOT = Path(__file__).resolve().parents[1]
REV = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'
SOURCES = {'model': {'id': 'nvidia/Qwen3.8-Flash-Next-NVFP4', 'sha': REV}}


def base():
    return json.loads((ROOT / 'profiles/nextn-262k-c2-s3.json').read_text())


def frozen():
    data = base(); data['kv_cache_dtype'] = 'nvfp4'
    data['kv_calibration'] = {'mode': 'frozen', 'file': 'kv-scales-v1.json', 'sha256': 'a' * 64}
    return data


def test_legacy_profile_hash_and_constructor_state_preserved():
    data = base(); profile = profile_from_dict(data)
    assert profile.kv_calibration_mode is None
    expected = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    assert profile.digest() == expected
    assert profile_from_json(profile.to_json()) == profile


def test_collect_is_explicit_bf16_and_owned_env():
    data = base(); data['kv_calibration'] = {'mode': 'collect'}
    p = profile_from_dict(data)
    assert p.kv_calibration_mode == 'collect'
    env = build_env(p, SOURCES)
    assert env['SGLANG_QSA_KV_AMAX_CALIB'] == '1'
    assert env['SGLANG_QSA_KV_MODEL_REVISION'] == REV
    assert env['SGLANG_QSA_KV_SCALE_SIDECAR'] == ''
    assert env['SGLANG_QSA_KV_SCALE_SHA256'] == ''


def test_frozen_nvfp4_is_bound_and_roundtrips():
    p = profile_from_dict(frozen())
    assert p.kv_calibration_mode == 'frozen'
    assert p.kv_scale_file == 'kv-scales-v1.json'
    assert p.kv_scale_sha256 == 'a' * 64
    assert profile_from_json(p.to_json()) == p
    env = build_env(p, SOURCES)
    assert env['SGLANG_QSA_KV_AMAX_CALIB'] == '0'
    assert env['SGLANG_QSA_KV_SCALE_SIDECAR'] == f'/opt/r0b0tlab/kv-calibration/{REV}/kv-scales-v1.json'
    assert env['SGLANG_QSA_KV_SCALE_SHA256'] == 'a' * 64
    assert set(k for k in env if k.startswith('SGLANG_QSA_KV_')) <= set(ENV_VARS_OWNED)
    argv = build_argv(p, SOURCES)
    assert argv[argv.index('--kv-cache-dtype') + 1] == 'nvfp4'
    assert argv[argv.index('--speculative-draft-model-quantization') + 1] == 'modelopt_mixed'


def test_default_env_clears_stale_calibration():
    env = build_env(profile_from_dict(base()), SOURCES)
    assert env['SGLANG_QSA_KV_AMAX_CALIB'] == '0'
    assert env['SGLANG_QSA_KV_SCALE_SIDECAR'] == env['SGLANG_QSA_KV_SCALE_SHA256'] == ''


@pytest.mark.parametrize('value', [None, True, 1, [], '', {'mode': 'wrong'}, {'mode': True}, {'mode': 'collect', 'extra': 1}])
def test_bad_block_rejected(value):
    data = base(); data['kv_calibration'] = value
    with pytest.raises(ProfileError): profile_from_dict(data)


@pytest.mark.parametrize('filename', ['../x', '/tmp/x', '.', '..', 'a/b', 'a\\b', '', 'has space', 'a\nb'])
def test_frozen_sidecar_is_basename_only(filename):
    data = frozen(); data['kv_calibration']['file'] = filename
    with pytest.raises(ProfileError): profile_from_dict(data)


@pytest.mark.parametrize('sha', [None, True, '', 'A' * 64, 'a' * 63, 'a' * 65, ' a' * 32])
def test_frozen_hash_is_literal_lowercase_sha256(sha):
    data = frozen(); data['kv_calibration']['sha256'] = sha
    with pytest.raises(ProfileError): profile_from_dict(data)


def test_modes_and_dtype_cannot_misrepresent_calibration():
    for data in (dict(base(), kv_cache_dtype='nvfp4'),
                 dict(frozen(), kv_cache_dtype='bf16'),
                 dict(frozen(), kv_calibration={'mode': 'collect'})):
        with pytest.raises(ProfileError): profile_from_dict(data)


def test_tampered_dataclass_is_rejected_by_runtime():
    p = profile_from_dict(frozen())
    changed = dataclasses.replace(p, kv_scale_sha256='b' * 64)
    with pytest.raises(ProfileError): build_env(changed, SOURCES)
