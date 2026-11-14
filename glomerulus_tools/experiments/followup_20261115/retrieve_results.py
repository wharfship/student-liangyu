import subprocess,tarfile,json
from common import *
R='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115'
code='''from pathlib import Path
import json,tarfile,subprocess
p=Path(REMOTE);assert (p/'pyramid_complete.json').exists()
names=['pyramid_preparation.json','pyramid_complete.json','pyramid_runs.json','pyramid_results_PNG','pyramid_results_JPG',
 'pyramid_inputs','pyramid_PNG.log','pyramid_JPG.log','setup.log']
with tarfile.open(p/'results.tgz','w:gz') as a:
 for name in names:a.add(p/name,arcname=name)
freeze=subprocess.check_output([str(p/'skinet_venv/bin/python'),'-m','pip','freeze'],text=True)
(p/'skinet_freeze.txt').write_text(freeze)
print(json.dumps({'bytes':(p/'results.tgz').stat().st_size}))
'''.replace('REMOTE',repr(R))
r=remote_python(code,timeout=180);assert r.returncode==0,r.stderr;print(r.stdout)
subprocess.run(['scp','-o','BatchMode=yes','chenliangyu@10.103.238.124:'+R+'/results.tgz',str(HERE/'results.tgz')],check=True)
with tarfile.open(HERE/'results.tgz') as a:a.extractall(HERE,filter='data')
subprocess.run(['scp','-o','BatchMode=yes','chenliangyu@10.103.238.124:'+R+'/skinet_freeze.txt',str(HERE/'skinet_freeze.txt')],check=True)
save(HERE/'retrieval.json',json.loads(r.stdout))
