import json,subprocess,tarfile
from common import *
R='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115'
snap=HERE/'execution_snapshot/glomerulus_tools';snap.mkdir(parents=True,exist_ok=False)
for p in (ROOT/'glomerulus_tools').glob('*.py'):__import__('shutil').copy2(p,snap/p.name)
for name in ['catalog.csv','runtime_registry.json']:__import__('shutil').copy2(ROOT/'glomerulus_tools'/name,snap/name)
save(snap.parent/'manifest.json',{p.name:sha(p) for p in snap.iterdir()})
with tarfile.open(HERE/'code_stage.tgz','w:gz') as a:
 a.add(HERE/'execution_snapshot',arcname='execution_snapshot');a.add(HERE/'run_pyramid_remote.py',arcname='run_pyramid_remote.py')
subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'code_stage.tgz'),'chenliangyu@10.103.238.124:'+R+'/'],check=True)
code='''from pathlib import Path
import tarfile,subprocess,json
p=Path(REMOTE);assert (p/'pyramid_preparation.json').exists();assert not (p/'pyramid_run_start.json').exists()
with tarfile.open(p/'code_stage.tgz') as a:a.extractall(p,filter='data')
with (p/'pyramid_run.log').open('w') as log:
 proc=subprocess.Popen(['/home/chenliangyu/miniconda3/envs/sam/bin/python','-u',str(p/'run_pyramid_remote.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':proc.pid,'log':str(p/'pyramid_run.log')};(p/'pyramid_run_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(R))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'pyramid_run_start.json',json.loads(r.stdout));print(r.stdout)
