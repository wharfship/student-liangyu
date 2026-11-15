from common import *
import subprocess

def main():
 dest=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115'
 r=subprocess.run(['scp','-q',str(HERE/'pilot_remote.py'),'chenliangyu@10.103.238.124:'+dest+'/pilot_remote.py'],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
 code=f'''from pathlib import Path
import json,subprocess
p=Path('{dest}')
assert json.loads((p/'smoke_complete.json').read_text())['status']=='complete'
assert not (p/'pilot_start.json').exists(),'Prevent duplicate batch'
log=(p/'pilot.log').open('w');r=subprocess.Popen([str(p/'tool_venv/bin/python'),'-u',str(p/'pilot_remote.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=p)
data=dict(pid=r.pid,log=str(p/'pilot.log'));(p/'pilot_start.json').write_text(json.dumps(data));print(json.dumps(data))
'''
 d=__import__('json').loads(remote(code));save(HERE/'pilot_start.json',d);print(d)
if __name__=='__main__':main()
