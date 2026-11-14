from common import *
code='''from pathlib import Path
import subprocess,json,os
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115')
records=[]
for args in [['lscpu'],['bash','-c','ulimit -c'],[str(p/'skinet_venv/bin/python'),'-X','faulthandler','-c','import numpy;print(numpy.__version__,flush=True);import tensorflow;print(tensorflow.__version__)']]:
 r=subprocess.run(args,capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='2',TF_ENABLE_ONEDNN_OPTS='0'))
 records.append({'command':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps(records))
'''
r=remote_python(code);assert r.returncode==0,r.stderr;save(HERE/'tf_probe.json',__import__('json').loads(r.stdout));print(r.stdout)
