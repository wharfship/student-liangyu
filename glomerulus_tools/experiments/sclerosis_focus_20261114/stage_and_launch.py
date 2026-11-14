from pathlib import Path
import json,subprocess,tarfile,time
from prepare import ROOT,HERE,save,sha
from inspect_sources import remote_python

REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
assert not (HERE/'launch.json').exists(),'Review previous run before relaunch'
snap=HERE/'execution_snapshot/glomerulus_tools';snap.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'glomerulus_tools').glob('*.py'):
    (snap/p.name).write_bytes(p.read_bytes())
for name in ['catalog.csv','runtime_registry.json']:(snap/name).write_bytes((ROOT/'glomerulus_tools'/name).read_bytes())
save(snap.parent/'manifest.json',{p.name:sha(p) for p in sorted(snap.iterdir())})
tasks=json.loads((HERE/'batch_NO_GT.json').read_text(encoding='utf-8'))
for task in tasks['cases']:
    for k in ['image','baseline','target']:task[k]=REMOTE+'/online/'+task['case_id']+'/'+Path(task[k]).name
save(HERE/'batch_remote_NO_GT.json',tasks)
with tarfile.open(HERE/'online_stage.tar.gz','w:gz') as archive:
    archive.add(HERE/'online',arcname='online')
    archive.add(HERE/'execution_snapshot',arcname='execution_snapshot')
    archive.add(HERE/'run_remote.py',arcname='run_remote.py')
    archive.add(HERE/'protocol.json',arcname='protocol.json')
    archive.add(HERE/'batch_remote_NO_GT.json',arcname='batch_NO_GT.json')
r=remote_python("from pathlib import Path;p=Path("+repr(REMOTE)+");assert not p.exists(),str(p);p.mkdir(parents=True);print(p)")
assert r.returncode==0,r.stderr
subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'online_stage.tar.gz'),'chenliangyu@10.103.238.124:'+REMOTE+'/online_stage.tar.gz'],check=True)
code='''from pathlib import Path
import tarfile,subprocess,json,sys
p=Path(REMOTE)
with tarfile.open(p/'online_stage.tar.gz') as a:a.extractall(p,filter='data')
with (p/'controller.log').open('w') as log:
 process=subprocess.Popen([sys.executable,'-u',str(p/'run_remote.py')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,directory=str(p),log=str(p/'controller.log'),completion=str(p/'inference_complete.json'))
(p/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr
record=json.loads(r.stdout);record['online_archive_sha256']=sha(HERE/'online_stage.tar.gz');save(HERE/'launch.json',record)
print(json.dumps(record,indent=2))
