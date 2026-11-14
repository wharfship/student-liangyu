from inspect_sources import remote_python
from prepare import HERE,save
import json

code='''from pathlib import Path
import subprocess,json
p=Path('/home/chenliangyu/project/new_agent')
commands=[['find',str(p),'-maxdepth','7','-type','d','-name','detectron2'],['which','nvcc'],['gcc','--version']]
records=[]
for cmd in commands:
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);records.append(dict(command=cmd,exit_code=r.returncode,output=(r.stdout+r.stderr)[:12000]))
vendor=p/'glomerulus_tools/experiments/domain_tools_execution_20261109/vendor'
if vendor.exists():records.append(dict(vendor=str(vendor),contents=[x.name for x in vendor.iterdir()]))
print(json.dumps(records,indent=2))
'''
r=remote_python(code,timeout=120);assert r.returncode==0,r.stderr
save(HERE/'specialist_remote_probe.json',json.loads(r.stdout));print(r.stdout)
