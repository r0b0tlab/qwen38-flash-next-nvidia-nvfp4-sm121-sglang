import json
from pathlib import Path
import pytest
from runtime import profile_from_dict,ProfileError
from runtime.entrypoint import build_argv
ROOT=Path(__file__).resolve().parents[1]
SOURCES=json.loads((ROOT/'locks/sources.json').read_text())

def test_explicit_native_marlin_target_keeps_draft_fp8_triton():
    d=json.loads((ROOT/'profiles/nextn-262k-hybridkv-c2-s3.json').read_text())
    d['moe_backend']='marlin';p=profile_from_dict(d);a=build_argv(p,SOURCES)
    assert a[a.index('--moe-runner-backend')+1]=='marlin'
    assert a[a.index('--speculative-moe-runner-backend')+1]=='triton'
    assert a[a.index('--quantization')+1]=='modelopt_mixed'

@pytest.mark.parametrize('value',['auto','cutlass',None,True])
def test_unknown_target_backend_rejected(value):
    d=json.loads((ROOT/'profiles/nextn-262k-hybridkv-c2-s3.json').read_text());d['moe_backend']=value
    with pytest.raises(ProfileError):profile_from_dict(d)
