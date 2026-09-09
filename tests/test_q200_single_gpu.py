import json
import fcntl
import os
import pytest
from scripts.q200_single_gpu import checked_finalist,SingleGpuAdmission,MODEL,REVISION

class NotGranted(RuntimeError):pass

def record():
    return {'image':'sha256:'+'1'*64,'profile_sha256':'2'*64,'model':MODEL,'model_sha':REVISION}

def test_matching_selected_finalist():
    r=record();checked_finalist(r,{**r,'status':'OPTIMIZATION_FINALIST_SELECTED'})

@pytest.mark.parametrize('field,value',[('status','READY'),('image','sha256:'+'3'*64),('profile_sha256','4'*64)])
def test_wrong_finalist_rejected(field,value):
    r=record();s={**r,'status':'OPTIMIZATION_FINALIST_SELECTED'};s[field]=value
    with pytest.raises(ValueError):checked_finalist(r,s)

@pytest.mark.parametrize('url',['https://127.0.0.1:30080','http://127.0.0.1:30080/v1','http://peer:30080'])
def test_rejects_other_topology_or_wrong_api_root(tmp_path,url):
    with pytest.raises(ValueError):SingleGpuAdmission(record(),url,tmp_path,NotGranted)

def test_one_node_lease_has_real_checks_before_and_after(tmp_path):
    calls=[]
    def snap():calls.append('snapshot');return {'gpu_count':1,'nodes':1}
    a=SingleGpuAdmission(record(),'http://127.0.0.1:30080',tmp_path,NotGranted,snapshot=snap,idle_fn=lambda:calls.append('idle'))
    with a.request('x') as lease:assert 'before' in lease
    assert calls==['idle','snapshot','idle','snapshot']
    assert lease['release_status']=='VERIFIED' and len(list(tmp_path.glob('*.json')))==1

def test_release_failure_preserves_lease_and_blocks_next_batch(tmp_path):
    n=0
    def snap():
        nonlocal n;n+=1
        if n>1:raise RuntimeError('runtime changed')
        return {'gpu_count':1}
    a=SingleGpuAdmission(record(),'http://127.0.0.1:30080',tmp_path,NotGranted,snapshot=snap,idle_fn=lambda:None)
    with a.request('first') as lease:pass
    assert lease['release_status']=='FAILED'
    with pytest.raises(NotGranted):
        with a.request('second'):pytest.fail('must not admit')

def test_competing_owner_and_symlink_lock_rejected(tmp_path):
    a=SingleGpuAdmission(record(),'http://127.0.0.1:30080',tmp_path,NotGranted,snapshot=lambda:{},idle_fn=lambda:None)
    fd=os.open(tmp_path/'.admission.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX)
    try:
        with pytest.raises(NotGranted):
            with a.request('x'):pytest.fail('lock conflict')
    finally:os.close(fd)
    (tmp_path/'.admission.lock').unlink();(tmp_path/'.admission.lock').symlink_to(tmp_path/'other')
    with pytest.raises(OSError):
        with a.request('x'):pytest.fail('symlink lock')


def test_sandbox_overrides_serve_entrypoint_without_weakening_limits():
    from scripts.q200_single_gpu import native_sandbox_command
    image='sha256:'+'1'*64
    prefix=['docker','run','--read-only','--network=none','--memory=256m']
    cmd=native_sandbox_command(prefix+[image,'python3','/opt/r0b0tlab/q200_sandbox_driver.py'],image)
    assert cmd[:len(prefix)]==prefix
    assert '--entrypoint=/usr/bin/python3' in cmd and '--runtime=runc' in cmd
    assert '--env=NVIDIA_VISIBLE_DEVICES=void' in cmd
    assert cmd[-2:]==[image,'/opt/r0b0tlab/q200_sandbox_driver.py']
    with pytest.raises(ValueError):native_sandbox_command(prefix+[image,'unexpected'],image)
