import subprocess,json
from common import *
R='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115'
catalog={c['task']['task_id']:c['task'] for c in read(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
plan=[]
for s in read(OLD/'specialist_membership_EVAL_ONLY.json'):
 t=catalog[s['task_id']]
 plan.append(dict(case_id=s['case_id'],stain=s['stain'],source_image=t['image'],native_mpp=t['source_mpp'],
   box_native=s['box_native'],scope=s['scope'],previous_image=R.replace('followup_20261115','sclerosis_focus_20261114')+'/specialist_online/'+s['case_id']+'/image.png'))
save(HERE/'pyramid_plan_EVAL_ONLY.json',plan)
for name in ['pyramid_plan_EVAL_ONLY.json','prepare_pyramid_remote.py']:
 subprocess.run(['scp','-o','BatchMode=yes',str(HERE/name),'chenliangyu@10.103.238.124:'+R+'/'],check=True)
code='''from pathlib import Path
import subprocess,json,time
p=Path(REMOTE)
with (p/'pyramid_preparation.log').open('w') as log:
 proc=subprocess.Popen(['/home/chenliangyu/miniconda3/envs/omni_seg/bin/python','-u',str(p/'prepare_pyramid_remote.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'start':time.time(),'log':str(p/'pyramid_preparation.log')}))
'''.replace('REMOTE',repr(R))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'pyramid_start.json',json.loads(r.stdout));print(r.stdout)
