import subprocess
from common import *
REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/followup_20261115'
CODE='''from pathlib import Path
import subprocess,sys,json,zipfile,time
p=Path(REMOTE);assert not (p/'setup_start.json').exists()
env=p/'skinet_venv'
subprocess.run([sys.executable,'-m','venv',str(env)],check=True)
for name in ['skinet_source.zip','skinet_cortex.zip','skinet_mest_main.zip']:
 with zipfile.ZipFile(p/'assets'/name) as z:z.extractall(p/'assets')
cmd=[str(env/'bin/python'),'-m','pip','install','--disable-pip-version-check','--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',
 'tensorflow[and-cuda]==2.15.1','pillow','opencv-python-headless<4.12','scipy','scikit-image','matplotlib','tqdm','psutil','shapely','seaborn','numba']
with (p/'setup.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec={'pid':proc.pid,'command':cmd,'start':time.time(),'completion':'pip process exit and GPU SavedModel smoke required'}
(p/'setup_start.json').write_text(json.dumps(rec));print(json.dumps(rec))
'''.replace('REMOTE',repr(REMOTE))
if __name__=='__main__':
 r=remote_python('from pathlib import Path;p=Path('+repr(REMOTE)+');p.mkdir(parents=True,exist_ok=False);(p/"assets").mkdir();print(p)');assert r.returncode==0,r.stderr
 for name in ['skinet_source.zip','skinet_cortex.zip','skinet_mest_main.zip']:
  subprocess.run(['scp','-o','BatchMode=yes',str(HERE/'assets'/name),'chenliangyu@10.103.238.124:'+REMOTE+'/assets/'],check=True)
 r=remote_python(CODE,timeout=120);assert r.returncode==0,r.stderr;save(HERE/'setup_start.json',__import__('json').loads(r.stdout));print(r.stdout)
