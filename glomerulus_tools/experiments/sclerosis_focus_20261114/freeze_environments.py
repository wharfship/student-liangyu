"""Read exact environments after completed jobs; no package mutations."""
from inspect_sources import remote_python
from prepare import HERE,save
code='''import subprocess,json
from pathlib import Path
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114')
envs={'sam':'/home/chenliangyu/miniconda3/envs/sam/bin/python',
 'segformer':'/home/chenliangyu/miniconda3/envs/segformer/bin/python',
 'mescnn':str(p/'specialist_venv/bin/python')}
data={}
for name,exe in envs.items():
 r=subprocess.run([exe,'-m','pip','freeze'],capture_output=True,text=True,check=True)
 data[name]={'python':exe,'freeze':r.stdout}
 data[name]['versions']=subprocess.check_output([exe,'-c','import sys,torch;print(sys.version);print(torch.__version__,torch.version.cuda)'],text=True)
print(json.dumps(data))
'''
if __name__=='__main__':
    import json
    r=remote_python(code,timeout=180);assert r.returncode==0,r.stderr
    save(HERE/'final_environment_freeze.json',json.loads(r.stdout))
    print('three environments frozen')
