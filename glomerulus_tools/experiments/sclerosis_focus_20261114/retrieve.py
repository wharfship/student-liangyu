from pathlib import Path
import json,subprocess,tarfile
from prepare import HERE,save
from inspect_sources import remote_python

REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
code='''from pathlib import Path
import tarfile,json
p=Path(REMOTE);assert (p/'inference_complete.json').exists()
names=['results','smoke','inference_complete.json','resume_runs.json','runs.json','execution_snapshot_memoryfix',
       'resume_controller.log','samrefiner_vitb_paper_fp16_resume.log','ritm_hrnet32_external_mask_point1_resume.log',
       'specialist_setup_complete.json','specialist_setup_runs.json','specialist_setup.log']
if (p/'seam_validation/completed.json').exists():names+=['seam_validation','seam_controller.log']
with tarfile.open(p/'completed_outputs.tgz','w:gz') as a:
 for name in names:
  if (p/name).exists():a.add(p/name,arcname=name)
print(json.dumps(dict(bytes=(p/'completed_outputs.tgz').stat().st_size,files=names)))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;print(r.stdout)
subprocess.run(['scp','-o','BatchMode=yes','chenliangyu@10.103.238.124:'+REMOTE+'/completed_outputs.tgz',str(HERE/'completed_outputs.tgz')],check=True)
with tarfile.open(HERE/'completed_outputs.tgz') as a:a.extractall(HERE,filter='data')
save(HERE/'retrieval.json',json.loads(r.stdout))
