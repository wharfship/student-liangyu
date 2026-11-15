"""Retrieve completed results and exact execution/environment receipts only."""
import subprocess,tarfile,json
from common import *

def main():
 dest=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115'
 code=f'''from pathlib import Path
import json,tarfile,hashlib,subprocess
p=Path('{dest}');assert json.loads((p/'pilot_complete.json').read_text())['status']=='complete'
names=['results','smoke','smoke_complete.json','smoke_runs.json','pilot_complete.json','pilot_runs.json','smoke.log','pilot.log','setup.log','fix_dependencies.log','requirements-requested.txt','requirements-executed-freeze.txt','reference_prediction.npy']
archive=p/'results.tgz'
with tarfile.open(archive,'w:gz') as t:
 for n in names:t.add(p/n,arcname=n)
snapshot=p/'execution_snapshot'
digests={{f.relative_to(snapshot).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(snapshot.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}}
print(json.dumps(dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),execution_files=digests,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''
 d=json.loads(remote(code));r=subprocess.run(['scp','-q','chenliangyu@10.103.238.124:'+dest+'/results.tgz',str(HERE/'results.tgz')],capture_output=True,text=True,timeout=180);assert r.returncode==0,r.stderr
 assert sha(HERE/'results.tgz')==d['sha256']
 local={f.relative_to(HERE/'execution_snapshot').as_posix():sha(f) for f in sorted((HERE/'execution_snapshot').rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
 assert local==d['execution_files'],'Execution snapshot changed'
 with tarfile.open(HERE/'results.tgz') as t:
  assert all(not n.startswith('/') and '..' not in Path(n).parts for n in t.getnames());t.extractall(HERE)
 save(HERE/'retrieval_verification.json',d);print('Retrieved completed5 batches; exact execution SHA verified; bytes',d['bytes'])
if __name__=='__main__':main()
