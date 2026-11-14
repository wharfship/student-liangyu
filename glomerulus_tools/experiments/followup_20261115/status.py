from common import *
code='''from pathlib import Path
import subprocess,json
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115')
for name in ['pyramid_preparation.log','skinet_smoke.log','skinet_results.log','pyramid_run.log','pyramid_PNG.log','pyramid_JPG.log']:
 f=p/name
 if f.exists():print(name+'\\n'+'\\n'.join(f.read_text(errors='replace').splitlines()[-6:]))
for name in ['pyramid_preparation.json','skinet_smoke_complete.json','skinet_results/completed.json','pyramid_runs.json']:
 f=p/name
 if f.exists():
  d=json.loads(f.read_text());print(name,json.dumps({k:v for k,v in d.items() if k!='records'}) if isinstance(d,dict) else d)
for mode in ['PNG','JPG']:
 f=p/('pyramid_results_'+mode)/'progress.json'
 if f.exists():print(mode,f.read_text())
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
'''
if __name__=='__main__':
 r=remote_python(code);assert r.returncode==0,r.stderr;print(r.stdout)
