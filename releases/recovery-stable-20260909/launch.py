#!/usr/bin/env python3
"""Launch the packaged native runtime through its owned memory-safe guard."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'repro-source'
IMAGE='sha256:2ee545cf877ae8497c123637e061b6e6313c624e30018f975969b1d554e27f56'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--receipt',type=Path)
    p.add_argument('--receipt-sha256')
    p.add_argument('--print',action='store_true')
    a=p.parse_args()
    if bool(a.receipt)!=bool(a.receipt_sha256):p.error('provide both trusted receipt and its retained SHA256, or neither')
    if a.receipt is None:
        if a.print:p.error('--print needs an existing trusted receipt pair')
        sys.path.insert(0,str(SOURCE))
        from scripts.verify_files import verify_files,write_receipt
        receipt=ROOT/'operator-checkpoint-receipt.json'
        if receipt.exists():p.error('receipt file already exists; use the trusted receipt pair or choose a clean bundle directory')
        verified=verify_files(str(a.model_root.resolve(strict=True)),str(SOURCE/'locks/sources.json'))
        sha=write_receipt(verified,str(receipt))
        print(json.dumps({'checkpoint_receipt':str(receipt),'verified_receipt_sha256':sha}),flush=True)
        a.receipt=receipt;a.receipt_sha256=sha
    cmd=[sys.executable,'-B',str(SOURCE/'scripts/guard.py'),'--image',IMAGE,
         '--profile',str(ROOT/'production-profile.json'),'--sources',str(SOURCE/'locks/sources.json'),
         '--model-root',str(a.model_root.resolve()),'--cache-dir',str(a.cache_dir.resolve()),
         '--state-dir',str(a.state_dir.resolve()),'--receipt',str(a.receipt.resolve()),
         '--receipt-sha256',a.receipt_sha256,'--max-watch-seconds','0']
    if a.print:cmd.append('--print')
    os.execv(sys.executable,cmd)
if __name__=='__main__':main()
