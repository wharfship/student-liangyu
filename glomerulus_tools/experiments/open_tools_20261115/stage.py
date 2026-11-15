"""Stage an explicit execution snapshot and bounded assets; no original data writes."""
import tarfile,subprocess,shutil
from common import *

def main():
 dest=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115';snapshot=HERE/'execution_snapshot/glomerulus_tools';snapshot.mkdir(parents=True,exist_ok=True)
 for p in (ROOT/'glomerulus_tools').glob('*.py'):shutil.copy2(p,snapshot/p.name)
 for name in ['catalog.csv','runtime_registry.json']:shutil.copy2(ROOT/'glomerulus_tools'/name,snapshot/name)
 save(HERE/'execution_snapshot/manifest.json',{p.name:sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()})
 for name in ['feature','nuclei','qc','qc_paired','tnbc']:
  task=read(HERE/(name+'_tasks_NO_GT.json'))
  for s in task['samples']:s['image']=s['image'].replace(str(HERE),dest).replace('\\','/')
  save(HERE/(name+'_remote_NO_GT.json'),task)
 archive=HERE/'stage.tgz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for folder in ['execution_snapshot','inputs','assets/instanseg_model']:
   tar.add(HERE/folder,arcname=folder)
  for name in ['Tissue_Detection_MPP10.pth','GrandQC_MPP15.pth','dinov2_vits14_pretrain.pth']:
   tar.add(HERE/'assets'/name,arcname='assets/'+name)
  for prefix in ['instanseg-','dinov2-','grandqc-']:
   p=next((HERE/'assets/source').glob(prefix+'*'));tar.add(p,arcname='assets/source/'+p.name)
  for p in HERE.glob('*remote.py'):tar.add(p,arcname=p.name)
  for p in HERE.glob('*_remote_NO_GT.json'):tar.add(p,arcname=p.name)
 result=subprocess.run(['scp','-q',str(archive),'chenliangyu@10.103.238.124:'+dest+'/stage.tgz'],capture_output=True,text=True,timeout=180)
 assert result.returncode==0,result.stderr
 code=f'''from pathlib import Path
import tarfile,hashlib,json,subprocess
p=Path('{dest}');archive=p/'stage.tgz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='{sha(archive)}'
with tarfile.open(archive) as t:
 assert all(not n.startswith('/') and '..' not in Path(n).parts for n in t.getnames())
 t.extractall(p)
py=str(p/'tool_venv/bin/python')
log=(p/'fix_dependencies.log').open('w')
r=subprocess.Popen([py,'-m','pip','install','opencv-python-headless==4.10.0.84'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(archive_sha256='{sha(archive)}',bytes=archive.stat().st_size,private_opencv_install_pid=r.pid)))
'''
 data=__import__('json').loads(remote(code));save(HERE/'staging.json',data);print(data)
if __name__=='__main__':main()
