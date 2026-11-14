"""Inspect already downloaded source, download bounded official specialist assets."""
import zipfile,requests,time,concurrent.futures
from common import *

def download(url,p,expected=None):
 tick=time.monotonic();rec={'url':url,'path':str(p),'expected_bytes':expected}
 try:
  if p.exists() and (expected is None or p.stat().st_size==expected):rec['status']='reuse'
  else:
   p.parent.mkdir(parents=True,exist_ok=True)
   with requests.get(url,stream=True,timeout=(20,35)) as r:
    rec['http_status']=r.status_code;r.raise_for_status()
    with p.with_suffix(p.suffix+'.part').open('wb') as f:
     for chunk in r.iter_content(2**20):
      if time.monotonic()-tick>600:raise TimeoutError('bounded600s download')
      f.write(chunk)
   p.with_suffix(p.suffix+'.part').replace(p);rec['status']='downloaded'
  rec.update(bytes=p.stat().st_size,sha256=sha(p))
  if expected:assert rec['bytes']==expected
 except Exception as e:rec.update(status='failed',error=str(e))
 rec['seconds']=time.monotonic()-tick
 return rec

def main():
 REPORT.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(OLD/'specialist_assets/mescnn_284e83781ae8e83a1362d896c2ea11fa95f82a9e.zip') as z:
  texts={name:z.read(name).decode('utf8') for name in z.namelist() if name.endswith('.py')}
  selected={name:text for name,text in texts.items() if any(k in name for k in ['reader','tiling','tile.py','segment.py','utils.py','config.py'])}
 save(HERE/'mescnn_source_selected.json',selected)
 for name,text in selected.items():
  print('\nSOURCE',name);print(text[:8000])
 url='https://api.github.com/repos/SkinetTeam/Skinet-MEST-C/releases/tags/v1.0'
 rec=requests.get(url,timeout=30);rec.raise_for_status();release=rec.json();save(HERE/'assets/skinet_release.json',release)
 jobs=[]
 for a in release['assets']:
  if a['name'] in ['skinet_mest_main.zip','skinet_cortex.zip']:
   jobs.append((a['browser_download_url'],HERE/'assets'/a['name'],a['size']))
 jobs.append(('https://codeload.github.com/SkinetTeam/Skinet-MEST-C/zip/refs/heads/main',HERE/'assets/skinet_source.zip',None))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  receipts=list(pool.map(lambda args:download(*args),jobs))
 save(HERE/'asset_receipts.json',receipts)
 print(json.dumps(receipts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
