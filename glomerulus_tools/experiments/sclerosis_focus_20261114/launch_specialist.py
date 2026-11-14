import json,subprocess,tarfile
from prepare import HERE,ROOT,save,sha
from inspect_sources import remote_python

REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
snap=HERE/'specialist_execution_snapshot/glomerulus_tools';snap.mkdir(parents=True,exist_ok=False)
for p in (ROOT/'glomerulus_tools').glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
for name in ['catalog.csv','runtime_registry.json']:(snap/name).write_bytes((ROOT/'glomerulus_tools'/name).read_bytes())
save(snap.parent/'manifest.json',{p.name:sha(p) for p in snap.iterdir()})
with tarfile.open(HERE/'specialist_stage.tgz','w:gz') as a:
    for name in ['specialist_online','specialist_execution_snapshot','specialist_batch_NO_GT.json','run_specialist_remote.py']:
        a.add(HERE/name,arcname=name)
subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'specialist_stage.tgz'),'chenliangyu@10.103.238.124:'+REMOTE+'/'],check=True)
code='''from pathlib import Path
import tarfile,subprocess,json
p=Path(REMOTE)
assert not (p/'specialist_start.json').exists()
with tarfile.open(p/'specialist_stage.tgz') as a:a.extractall(p,filter='data')
with (p/'specialist_controller.log').open('w') as log:
 proc=subprocess.Popen([str(p/'specialist_venv/bin/python'),'-u',str(p/'run_specialist_remote.py')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=proc.pid,log=str(p/'specialist_controller.log'));(p/'specialist_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'specialist_start.json',json.loads(r.stdout));print(r.stdout)
