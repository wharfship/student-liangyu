import subprocess,json
from prepare import HERE,ROOT,save
from inspect_sources import remote_python
REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
for source,name in [(HERE/'resume_refiners_remote.py','resume_refiners_remote.py'),(ROOT/'glomerulus_tools/mask_refinement.py','mask_refinement_memoryfix.py')]:
    subprocess.run(['scp','-o','BatchMode=yes',str(source),'chenliangyu@10.103.238.124:'+REMOTE+'/'+name],check=True)
code='''from pathlib import Path
import subprocess,sys,json
p=Path(REMOTE)
assert not (p/'resume_start.json').exists()
with (p/'resume_controller.log').open('w') as log:
 proc=subprocess.Popen([sys.executable,'-u',str(p/'resume_refiners_remote.py')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=proc.pid,log=str(p/'resume_controller.log'));(p/'resume_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'resume_start.json',json.loads(r.stdout));print(r.stdout)
