"""Read-only final remote verification. Read the lab manual before running."""
from common import *


def main():
 code='''from pathlib import Path
import json,hashlib,subprocess,os,socket
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115')
assert __import__('getpass').getuser()=='chenliangyu'
assert socket.gethostname()=='netxge'
snapshot=p/'execution_snapshot'
files={f.relative_to(snapshot).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(snapshot.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
runs=json.loads((p/'pyramid_runs.json').read_text())
for r in runs:
 assert r['exit_code']==0
 complete=json.loads((p/('pyramid_results_'+r['mode'])/'completed.json').read_text())
 assert complete['status']=='complete' and complete['cases']==80
 try:os.kill(r['pid'],0);r['process_exists_now']=True
 except ProcessLookupError:r['process_exists_now']=False
print(json.dumps(dict(host=socket.gethostname(),user=__import__('getpass').getuser(),files=files,runs=runs,
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 claims='read-only verification; PIDs could be reused; no process termination'),ensure_ascii=False))
'''
 r=remote_python(code);assert r.returncode==0,r.stderr;data=json.loads(r.stdout)
 local={p.relative_to(HERE/'execution_snapshot').as_posix():sha(p) for p in sorted((HERE/'execution_snapshot').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
 assert data['files']==local,'Remote execution snapshot differs from staged bytes'
 data['local_snapshot_equal_remote']=True;save(HERE/'executed_source_verification.json',data)
 print('Remote snapshot verified:',len(local),'files; both80 runs exit0. GPU:',data['gpu'])


if __name__=='__main__':main()
