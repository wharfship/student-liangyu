from inspect_sources import remote_python
from prepare import HERE, save
import json

code='''from pathlib import Path
import subprocess, json, datetime
p=Path('/home/chenliangyu/project/new_agent')
commands=[['hostname'],['whoami'],['pwd'],['uname','-a'],['nvidia-smi'],['/home/chenliangyu/miniconda3/bin/conda','env','list']]
results=[]
for cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 results.append(dict(command=cmd,exit_code=r.returncode,output=r.stdout+r.stderr))
for exe in [Path('/home/chenliangyu/miniconda3/envs/sam/bin/python'),p/'glomerulus_tools/experiments/candidate_classifier_20261108/venv/bin/python']:
 if exe.exists():
  cmd=[str(exe),'-c',"import sys,importlib.util,json;print(sys.version);print({n:importlib.util.find_spec(n) is not None for n in ['detectron2','tensorflow','torch']});import torch;print(torch.__version__,torch.version.cuda)"]
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);results.append(dict(command=cmd,exit_code=r.returncode,output=r.stdout+r.stderr))
record=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=results)
out=Path('/home/chenliangyu/project/_server_inventory')/('sclerosis_focus_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
out.write_text(json.dumps(record,indent=2));record['snapshot_path']=str(out)
print(json.dumps(record,indent=2))
'''
r=remote_python(code,timeout=180)
assert r.returncode==0,r.stderr
data=json.loads(r.stdout);save(HERE/'environment_snapshot.json',data)
print(r.stdout)
