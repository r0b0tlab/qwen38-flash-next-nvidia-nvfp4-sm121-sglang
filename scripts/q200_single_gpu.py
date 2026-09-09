#!/usr/bin/env python3
"""Run the frozen text-180 Q200 component on one verified native finalist.

The immutable toolkit supplies grading; this module supplies real one-node
admission, not a two-rank lease emulation. BFCL-hard20 closes separately.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.guard import DockerCliTransport,verify_ownership
from scripts.native_profile_probe import idle

MODEL='nvidia/Qwen3.8-Flash-Next-NVFP4'
REVISION='fc694b54fb0174e0913e6adf86691ef85a4ead47'
TEXT_SHA='74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6'


def checked_finalist(record,selection):
    if selection.get('status')!='OPTIMIZATION_FINALIST_SELECTED':raise ValueError('optimization finalist not selected')
    if record.get('model')!=MODEL or record.get('model_sha')!=REVISION:raise ValueError('checkpoint identity mismatch')
    for key in ('image','profile_sha256'):
        if record.get(key)!=selection.get(key):raise ValueError('finalist '+key+' mismatch')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',str(record.get('image',''))):raise ValueError('immutable image required')
    if not re.fullmatch(r'[0-9a-f]{64}',str(record.get('profile_sha256',''))):raise ValueError('profile hash required')


def read_json_url(base,path):
    with urllib.request.urlopen(base+path,timeout=20) as response:return json.load(response)


class SingleGpuAdmission:
    def __init__(self,record,base,output,not_granted,*,snapshot=None,idle_fn=None):
        url=urllib.parse.urlsplit(base)
        if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost') or url.path not in ('','/'):
            raise ValueError('single-node native HTTP root required, without /v1')
        self.record=record;self.base=base.rstrip('/');self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.not_granted=not_granted;self.snapshot=snapshot or self._snapshot
        self.idle_fn=idle_fn or (lambda:idle(self.base));self.failed=None

    def _snapshot(self):
        doc=verify_ownership(DockerCliTransport().inspect(self.record['cid']),self.record)
        if not doc['State']['Running'] or doc['State'].get('OOMKilled'):raise RuntimeError('owned runtime not healthy')
        info=read_json_url(self.base,'/get_server_info')
        models=read_json_url(self.base,'/v1/models')
        if info.get('status')!='ready' or [x['id'] for x in models.get('data',[])]!=[MODEL]:
            raise RuntimeError('native endpoint identity/readiness mismatch')
        available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        if available<10*2**30:raise RuntimeError('Q200 generation/sandbox memory reserve')
        return {'cid':doc['Id'],'image':doc['Image'],'profile_sha256':self.record['profile_sha256'],
                'model':MODEL,'version':info.get('version'),'host_available_bytes':available,
                'context_length':info.get('context_length'),'gpu_count':1,'nodes':1}

    @contextmanager
    def request(self,row_id):
        if self.failed:raise self.not_granted(self.failed)
        if not isinstance(row_id,str) or not row_id or len(row_id)>500:raise ValueError('invalid admission row ID')
        lease={'schema':'single-gb10-q200-admission-v1','row_id':row_id,'owner_pid':os.getpid()}
        fd=os.open(self.output/'.admission.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.idle_fn();lease['before']=self.snapshot()
            except Exception as exc:raise self.not_granted(str(exc)) from exc
            lease['admitted_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:yield lease
            finally:
                try:
                    self.idle_fn();lease['after']=self.snapshot();lease['release_status']='VERIFIED'
                except Exception as exc:
                    # Preserve completed responses; block every subsequent batch.
                    self.failed=type(exc).__name__+': '+str(exc)
                    lease['release_status']='FAILED';lease['release_error']=self.failed
                lease['released_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                name=hashlib.sha256(row_id.encode()).hexdigest()+'.json'
                (self.output/name).write_text(json.dumps(lease,indent=2)+'\n')
        finally:
            fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def native_sandbox_command(command,image):
    """Keep all frozen limits, but override the serving image's ENTRYPOINT."""
    command=list(command)
    index=command.index(image)
    if command[index+1:]!=['python3','/opt/r0b0tlab/q200_sandbox_driver.py']:
        raise ValueError('unexpected frozen sandbox command')
    return command[:index]+['--entrypoint=/usr/bin/python3','--runtime=runc',
        '--env=NVIDIA_VISIBLE_DEVICES=void','--env=CUDA_VISIBLE_DEVICES=',image,
        '/opt/r0b0tlab/q200_sandbox_driver.py']


def load_kit(path):
    path=Path(path).resolve(strict=True)
    manifest=json.loads((path/'SOURCE_MANIFEST.json').read_text())
    for item in manifest['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('invalid toolkit manifest path')
        if hashlib.sha256((path/rel).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('toolkit hash mismatch: '+str(rel))
    if hashlib.sha256((path/'artifacts/quality-text-180-v2.jsonl').read_bytes()).hexdigest()!=TEXT_SHA:
        raise ValueError('wrong frozen text-180 corpus')
    sys.path.insert(0,str(path/'scripts'))
    spec=importlib.util.spec_from_file_location('frozen_q200_text180',path/'scripts/run_quality_set.py')
    if spec is None or spec.loader is None:raise ValueError('toolkit module cannot be loaded')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    upstream_command=module._docker_command
    setattr(module,"_docker_command",lambda name,image:native_sandbox_command(upstream_command(name,image),image))
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit',type=Path,required=True)
    parser.add_argument('--launch-record',type=Path,required=True)
    parser.add_argument('--finalist',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:30080')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--manual-evidence',type=Path)
    args=parser.parse_args(argv)
    if args.max_tokens<8192:parser.error('native-thinking Q200 requires a generous 8192-token ceiling or higher')
    record=json.loads(args.launch_record.read_text());selection=json.loads(args.finalist.read_text())
    checked_finalist(record,selection);core=load_kit(args.kit)
    admission=SingleGpuAdmission(record,args.base_url,args.output/'admission',core.AdmissionNotGranted)
    summary=core.run_quality(base_url=args.base_url,run_id=str(args.output/'text180'),
        dataset_path=args.kit/'artifacts/quality-text-180-v2.jsonl',admission=admission,model=MODEL,
        max_tokens=args.max_tokens,timeout=1200,workers=args.workers,human_eval_timeout=8,
        image_id=record['image'],profile_id=record['profile_sha256'],candidate_id=selection.get('candidate_id','selected'),
        chat_kwargs={'thinking':True,'enable_thinking':True,'reasoning_effort':'low'},manual_evidence_path=args.manual_evidence)
    summary['single_gpu_admission_error']=admission.failed
    (args.output/'single-gpu-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('status','rows','transport_complete','grade_complete','correct_count','ungraded_count','single_gpu_admission_error')}))
    if not summary['transport_complete'] or admission.failed is not None:return 1
    # Twenty response-bound hard-reasoning grades are a separate mandatory gate.
    return 0 if summary['status']=='SCORED' else 2

if __name__=='__main__':raise SystemExit(main())
