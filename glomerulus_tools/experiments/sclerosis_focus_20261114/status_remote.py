from inspect_sources import remote_python
code='''from pathlib import Path
import json,subprocess
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114')
for name in ['specialist_runs.json','specialist_exit.json','specialist_inference_complete.json','seam_validation/completed.json']:
 f=p/name
 if f.exists():print(name,f.read_text()[-5000:])
for f in sorted(p.glob('*resume.log')):
 print(f.name,f.read_text()[-600:])
for name in ['specialist_controller.log','specialist_smoke.log','specialist_results.log','seam_controller.log']:
 f=p/name
 if f.exists():print(name,f.read_text()[-1800:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True))
'''
r=remote_python(code);print(r.stdout,r.stderr);assert r.returncode==0
