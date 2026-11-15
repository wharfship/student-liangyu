"""Resolve and download pinned author assets, never a gated checkpoint."""
import requests,concurrent.futures,zipfile,base64,time
from common import *

def get(url):
 r=requests.get(url,timeout=(8,30));r.raise_for_status();return r

def download(url,name,expected=None):
 p=HERE/'assets'/name;p.parent.mkdir(exist_ok=True)
 if p.exists() and (not expected or sha(p)==expected):return dict(url=url,file=name,bytes=p.stat().st_size,sha256=sha(p),cached=True)
 start=time.monotonic();part=p.with_suffix(p.suffix+'.part')
 with requests.get(url,stream=True,timeout=(12,60)) as r:
  r.raise_for_status()
  with part.open('wb') as f:
   for b in r.iter_content(2**20):f.write(b)
 digest=sha(part)
 if expected:assert digest==expected,(name,'hash differs')
 part.replace(p);print('DOWNLOADED',name,p.stat().st_size,flush=True)
 return dict(url=url,file=name,bytes=p.stat().st_size,sha256=digest,upstream_sha_matched=bool(expected),seconds=time.monotonic()-start)

def metadata(item):
 name,url=item
 try:
  r=get(url);(HERE/'sources'/(name+'.txt')).write_bytes(r.content);print(name,r.status_code,len(r.content),flush=True);return dict(name=name,url=url,status=r.status_code)
 except Exception as e:return dict(name=name,url=url,error=str(e))

def main():
 queries={
  'grandqc_zenodo_tissue':'https://zenodo.org/api/records/14507273',
  'grandqc_zenodo_artifacts':'https://zenodo.org/api/records/14041538',
  'grandqc_zenodo_benchmark':'https://zenodo.org/api/records/14041717',
  'hibou_api':'https://hf-mirror.com/api/models/histai/hibou-b?blobs=true',
  'hibou_card':'https://hf-mirror.com/histai/hibou-b/raw/main/README.md',
 }
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(metadata,queries.items()))
 save(HERE/'asset_metadata_requests.json',records)
 # The InstanSeg source tree and the release are archived separately.
 itree=read(HERE/'sources/instanseg_tree.txt');commit=itree['sha']
 receipts=[download('https://github.com/instanseg/instanseg/releases/download/instanseg_models_v0.1.1/brightfield_nuclei.zip','instanseg_brightfield_v0.1.1.zip'),
  download('https://codeload.github.com/instanseg/instanseg/zip/'+commit,'instanseg_source.zip'),
  download('https://codeload.github.com/cpath-ukk/grandqc/zip/'+read(HERE/'sources/grandqc_tree.txt')['sha'],'grandqc_source.zip')]
 save(HERE/'download_receipts.json',receipts)
 for receipt in receipts:
  if 'source' in receipt['file'] or 'instanseg_brightfield' in receipt['file']:
   with zipfile.ZipFile(HERE/'assets'/receipt['file']) as z:
    assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    dest=HERE/'assets'/('instanseg_model' if 'brightfield' in receipt['file'] else 'source');z.extractall(dest)
 print('Source and model archives extracted',flush=True)

if __name__=='__main__':main()
