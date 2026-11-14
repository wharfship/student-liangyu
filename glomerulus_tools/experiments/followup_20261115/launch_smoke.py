import subprocess,json
from common import *
R='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115'
subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'skinet_smoke_remote.py'),'chenliangyu@10.103.238.124:'+R+'/'],check=True)
code='''from pathlib import Path
import subprocess,json,os
p=Path(REMOTE);python=p/'skinet_venv/bin/python'
r=subprocess.run([str(python),'-c','import tensorflow as tf;print(tf.__version__)'],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TF_NUM_INTEROP_THREADS='2',TF_NUM_INTRAOP_THREADS='2'))
assert r.returncode==0,repr((r.returncode,r.stdout,r.stderr))
with (p/'skinet_smoke.log').open('w') as log:
 proc=subprocess.Popen([str(python),'-u',str(p/'skinet_smoke_remote.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'log':str(p/'skinet_smoke.log')}))
'''.replace('REMOTE',repr(R))
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'skinet_smoke_start.json',json.loads(r.stdout));print(r.stdout)
