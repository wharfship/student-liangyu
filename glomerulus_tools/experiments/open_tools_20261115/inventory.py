"""Read remote identity/resources and snapshot before any new environment."""
from common import *
code='''import subprocess,json,os,socket,importlib.util
from pathlib import Path
assert __import__('getpass').getuser()=='chenliangyu' and socket.gethostname()=='netxge'
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115');p.mkdir(parents=True,exist_ok=True)
data=dict(host=socket.gethostname(),user=__import__('getpass').getuser(),cwd=os.getcwd(),
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),
 disk=subprocess.check_output(['df','-h',str(p)],text=True),
 environments=subprocess.check_output(['/home/chenliangyu/miniconda3/bin/conda','env','list'],text=True))
data['python_probes']={}
for py in ['/home/chenliangyu/miniconda3/envs/sam/bin/python','/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114/specialist_venv/bin/python']:
 r=subprocess.run([py,'-c','import torch,importlib.util,json;print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.is_available(),modules={x:importlib.util.find_spec(x)is not None for x in ["transformers","timm","sklearn","instanseg","segmentation_models_pytorch"]})))'],capture_output=True,text=True)
 data['python_probes'][py]=dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
snapshot=Path('/home/chenliangyu/project/_server_inventory/open_tools_20261115.json');snapshot.write_text(json.dumps(data,indent=2));data['snapshot']=str(snapshot)
print(json.dumps(data))
'''
if __name__=='__main__':
 data=__import__('json').loads(remote(code));save(HERE/'environment_inventory.json',data);print(__import__('json').dumps(data,indent=2))
