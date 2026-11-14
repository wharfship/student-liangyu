import subprocess,tarfile,json
from prepare import HERE,save
from inspect_sources import remote_python
REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
code='''from pathlib import Path
import tarfile,json
p=Path(REMOTE);assert (p/'specialist_inference_complete.json').exists()
names=['specialist_results','specialist_smoke','specialist_runs.json','specialist_exit.json','specialist_inference_complete.json',
       'specialist_controller.log','specialist_results.log','specialist_smoke.log']
with tarfile.open(p/'specialist_outputs.tgz','w:gz') as a:
 for name in names:a.add(p/name,arcname=name)
print(json.dumps(dict(bytes=(p/'specialist_outputs.tgz').stat().st_size)))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;print(r.stdout)
subprocess.run(['scp','-o','BatchMode=yes','chenliangyu@10.103.238.124:'+REMOTE+'/specialist_outputs.tgz',str(HERE/'specialist_outputs.tgz')],check=True)
with tarfile.open(HERE/'specialist_outputs.tgz') as a:a.extractall(HERE,filter='data')
save(HERE/'specialist_retrieval.json',json.loads(r.stdout))
