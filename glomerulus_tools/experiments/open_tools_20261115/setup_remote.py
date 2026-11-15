"""Create a private package layer, preserving the existing GPU torch stack."""
from common import *
code='''from pathlib import Path
import subprocess,json,os
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115')
env=p/'tool_venv';base='/home/chenliangyu/miniconda3/envs/sam/bin/python'
assert not env.exists(),'Preserve existing environment; inspect rather than recreating'
subprocess.run([base,'-m','venv','--system-site-packages',str(env)],check=True)
requirements='numpy==1.26.4\\nscipy==1.13.1\\nscikit-image==0.24.0\\nscikit-learn==1.5.2\\nmatplotlib==3.9.4\\nsegmentation-models-pytorch==0.3.1\\ntransformers==4.48.3\\nfastremap==1.17.7\\neinops==0.8.1\\ninstanseg-torch==0.1.2\\n'
(p/'requirements-requested.txt').write_text(requirements)
log=(p/'setup.log').open('w')
cmd=[str(env/'bin/python'),'-m','pip','install','-r',str(p/'requirements-requested.txt')]
proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=p)
print(json.dumps(dict(pid=proc.pid,command=cmd,log=str(p/'setup.log'),environment=str(env),base_environment_unchanged=True)))
'''
if __name__=='__main__':
 d=__import__('json').loads(remote(code));save(HERE/'setup_start.json',d);print(d)
