from common import *
code='''from pathlib import Path
import subprocess,json
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115');py=str(p/'tool_venv/bin/python')
test=subprocess.run([py,'-c','import torch,cv2,instanseg,segmentation_models_pytorch,numpy;print(torch.__version__,cv2.__version__,numpy.__version__);assert torch.cuda.is_available()'],capture_output=True,text=True)
assert test.returncode==0,test.stderr
log=(p/'smoke.log').open('w');r=subprocess.Popen([py,'-u',str(p/'smoke_remote.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=p)
print(json.dumps(dict(pid=r.pid,log=str(p/'smoke.log'),probe=test.stdout)))
'''
if __name__=='__main__':
 d=__import__('json').loads(remote(code));save(HERE/'smoke_start.json',d);print(d)
