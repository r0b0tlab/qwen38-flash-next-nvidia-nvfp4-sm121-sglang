import copy,json
from pathlib import Path
import pytest
from runtime import profile_from_dict,profile_from_json,ProfileError
from runtime.entrypoint import build_argv
ROOT=Path(__file__).resolve().parents[1]
SOURCES=json.loads((ROOT/'locks/sources.json').read_text())

def base():return json.loads((ROOT/'profiles/nextn-262k-nvfp4-c2-s2.json').read_text())

def test_draft_kv_inherits_without_override():
    p=profile_from_dict(base());assert p.draft_kv_cache_dtype is None
    args=build_argv(p,SOURCES)
    assert args[args.index('--speculative-draft-kv-cache-dtype')+1]=='nvfp4'

def test_explicit_bf16_draft_preserves_native_weights_and_target_kv():
    d=base();d['speculative']['kv_cache_dtype']='bf16';p=profile_from_dict(d)
    assert p.draft_kv_cache_dtype=='bf16' and profile_from_json(p.to_json())==p
    args=build_argv(p,SOURCES)
    assert args[args.index('--kv-cache-dtype')+1]=='nvfp4'
    assert args[args.index('--speculative-draft-kv-cache-dtype')+1]=='bf16'
    assert args[args.index('--speculative-draft-model-quantization')+1]=='modelopt_mixed'

@pytest.mark.parametrize('bad',[None,True,1,'fp8_e4m3','unknown'])
def test_invalid_draft_override(bad):
    d=base();d['speculative']['kv_cache_dtype']=bad
    with pytest.raises(ProfileError):profile_from_dict(d)
