from common import *
code='''from pathlib import Path
import subprocess
p=Path('/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/open_tools_20261115')
for name in ['setup.log','setup_encoder.log','smoke.log','pilot.log']:
 f=p/name
 if f.exists():print(name+'\\n'+'\\n'.join(f.read_text(errors='replace').splitlines()[-16:]))
for name in ['smoke_complete.json','pilot_complete.json']:
 f=p/name
 if f.exists():print(name,f.read_text())
'''
if __name__=='__main__':print(remote(code))
