import json,subprocess
from prepare import HERE,ROOT,read,csvout,save
from inspect_sources import remote_python

OLD=HERE.parent/'problem_to_repair_20261111'
cases=read(OLD/'benchmark_EVAL_ONLY.json')
seams=['C004','C022','C030','C031','C032','C034']
controls=['C021','C023','C024','C025','C049','C050']
catalog={x['task']['task_id']:x['task'] for x in read(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
selected=[c for c in cases if c['case_id'] in seams+controls]
assert len(selected)==12
plan=dict(schema='local_grid_phase_diagnostic_v1',phase=256,GT_read=False,
          scope='previous GT-stratified developmental cohort; candidates from frozen prediction geometry, no prevalence claim',cases=[])
for c in selected:
    plan['cases'].append(dict(case_id=c['case_id'],task_id=c['task_id'],box=c['box'],cache_key=catalog[c['task_id']]['cache_key']))
save(HERE/'seam_plan_NO_GT.json',plan)
save(HERE/'seam_evaluation_membership.json',{c['case_id']:'seam_candidate' if c['case_id'] in seams else 'normal_geometry_control' for c in selected})
REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
for name in ['seam_validation_remote.py','seam_plan_NO_GT.json','environment_snapshot.json']:
    subprocess.run(['scp','-o','BatchMode=yes',str(HERE/name),'chenliangyu@10.103.238.124:'+REMOTE+'/'],check=True)
code='''from pathlib import Path
import subprocess,json
p=Path(REMOTE)
assert not (p/'seam_start.json').exists()
with (p/'seam_controller.log').open('w') as log:
 proc=subprocess.Popen(['/home/chenliangyu/miniconda3/envs/segformer/bin/python','-u',str(p/'seam_validation_remote.py')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=proc.pid,log=str(p/'seam_controller.log'));(p/'seam_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(REMOTE))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'seam_start.json',json.loads(r.stdout));print(r.stdout)
