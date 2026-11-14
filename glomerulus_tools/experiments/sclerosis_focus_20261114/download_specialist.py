"""Fetch only the previously verified MESCnn internal-validation task checkpoint."""
from pathlib import Path
import hashlib,json,time,urllib.request,sys

HERE=Path(__file__).resolve().parent
ASSETS=HERE/'specialist_assets';ASSETS.mkdir(exist_ok=True)
FILE=ASSETS/'mescnn_cascade_r50_internal_model_final.pth'
URL='https://huggingface.co/MESCnn/MESCnn/resolve/a38cc99aa5fea432743fdeaf5f3f4113b2057aa4/detection/logs/cascade_mask_rcnn_R_50_FPN_1x/internal-validation/output/model_final.pth?download=true'
if '--mirror' in sys.argv:URL=URL.replace('huggingface.co','hf-mirror.com')
EXPECTED_SIZE=573457459
EXPECTED_SHA='007181ac33b5a4a0e6f13d5d6dfef9ee06a45fa0e8d33494ff7c885f98d812d4'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    start=time.monotonic();record=dict(source_url=URL,expected_bytes=EXPECTED_SIZE,expected_sha256=EXPECTED_SHA)
    try:
        if not FILE.exists():
            part=FILE.with_suffix('.pth.partial')
            if part.exists():raise RuntimeError('Partial exists; review prior receipt before retry')
            req=urllib.request.Request(URL,headers={'User-Agent':'glomerulus-research/20261114'})
            with urllib.request.urlopen(req,timeout=60) as r,part.open('wb') as f:
                total=0;next_print=32*2**20
                record['http_status']=r.status
                while True:
                    b=r.read(2**20)
                    if not b:break
                    f.write(b);total+=len(b)
                    if total>=next_print:
                        print('downloaded_MiB',round(total/2**20),flush=True);next_print+=32*2**20
                    if time.monotonic()-start>1200:raise TimeoutError('20 minute download bound exceeded')
            assert part.stat().st_size==EXPECTED_SIZE,part.stat().st_size
            assert digest(part)==EXPECTED_SHA,'checkpoint digest mismatch'
            part.rename(FILE)
        assert FILE.stat().st_size==EXPECTED_SIZE and digest(FILE)==EXPECTED_SHA
        record.update(status='downloaded_and_verified',actual_bytes=FILE.stat().st_size,actual_sha256=digest(FILE),path=str(FILE))
    except Exception as e:
        record.update(status='failed',error=repr(e))
    record['seconds']=time.monotonic()-start
    receipt='download_mirror_receipt.json' if '--mirror' in sys.argv else 'download_receipt.json'
    (ASSETS/receipt).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    if record['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
