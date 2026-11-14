from common import *
CODE='''from pathlib import Path
import json,subprocess,datetime,hashlib
p=Path('/home/chenliangyu/project/new_agent');old=p/'glomerulus_tools/experiments/sclerosis_focus_20261114'
assert subprocess.check_output(['hostname'],text=True).strip()=='netxge'
assert subprocess.check_output(['whoami'],text=True).strip()=='chenliangyu'
commands=[['pwd'],['hostname'],['whoami'],['nvidia-smi'],['/home/chenliangyu/miniconda3/bin/conda','env','list'],
 [str(old/'specialist_venv/bin/python'),'-m','pip','freeze']]
data={'utc':str(datetime.datetime.now(datetime.timezone.utc)),'commands':[]}
for cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);data['commands'].append({'command':cmd,'exit_code':r.returncode,'output':r.stdout+r.stderr})
inv=Path('/home/chenliangyu/project/_server_inventory')/('followup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
inv.write_text(json.dumps(data,indent=2));data['snapshot_path']=str(inv)
# Read metadata from original paths recorded in the previous preparation.
members=json.loads((old/'specialist_membership_EVAL_ONLY.json').read_text()) if (old/'specialist_membership_EVAL_ONLY.json').exists() else []
data['metadata_available']=bool(members)
data['openslide_probe']=subprocess.run(['/home/chenliangyu/miniconda3/envs/omni_seg/bin/python','-c','import openslide;print(openslide.__version__)'],capture_output=True,text=True).stdout
print(json.dumps(data))
'''
if __name__=='__main__':
 r=remote_python(CODE,timeout=150);assert r.returncode==0,r.stderr
 d=__import__('json').loads(r.stdout);save(HERE/'environment_snapshot.json',d)
 for c in d['commands']:
  if c['command'][0] in ['nvidia-smi','hostname','whoami']:print(c['output'])
 print('openslide',d['openslide_probe'])
