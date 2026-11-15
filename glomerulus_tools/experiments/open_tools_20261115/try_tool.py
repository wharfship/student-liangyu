"""One-command local client for the already accepted lab GPU installation.

Reads the required server manual before SSH. Uses SSH key authentication only.
Default: a recorded example. Custom patch: --image --mpp --scale-evidence --stain.
Never sends images to a VLM, changes original data, or updates the tool catalog.
"""
import argparse,subprocess,uuid,tarfile,shutil,json
from common import *

ALIASES={'nuclei':('instanseg_brightfield_nuclei_v011','nuclei'),
 'features':('dinov2_vits14_features','feature'),'tissue':('grandqc_tissue_mpp10','qc'),
 'qc':('grandqc_artifact_mpp15','qc')}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',choices=ALIASES,required=True)
 p.add_argument('--image',type=Path);p.add_argument('--mpp',type=float);p.add_argument('--scale-evidence');p.add_argument('--stain',choices=['HE','PAS'],default='HE')
 a=p.parse_args();tid,taskname=ALIASES[a.tool]
 manual=ROOT/'docs/知识库/remote_lab_server_ai_manual.md'
 text=manual.read_text(encoding='utf8');assert text.strip(),'Required server manual unavailable'
 run_id='manual_'+uuid.uuid4().hex[:12];remote_dir=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115';dest=HERE/'manual_tests'/run_id;dest.mkdir(parents=True)
 original=read(HERE/(taskname+'_remote_NO_GT.json'))['samples'][0]
 if a.image:
  if not a.image.is_file() or not a.mpp or not a.scale_evidence:p.error('Custom patch requires existing --image, positive --mpp and explicit --scale-evidence')
  from PIL import Image
  with Image.open(a.image) as im:
   if max(im.size)>8192:p.error('Use a bounded patch, max side8192; full WSI is not this protocol')
  remote_image=remote_dir+'/manual_inputs/'+run_id+a.image.suffix.lower()
  remote(f"from pathlib import Path;Path('{remote_dir}/manual_inputs').mkdir(exist_ok=True)")
  r=subprocess.run(['scp','-q',str(a.image.resolve()),'chenliangyu@10.103.238.124:'+remote_image],capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
  original=dict(sample_id=run_id,image=remote_image,image_sha256=sha(a.image),source_mpp=a.mpp,scale_evidence=a.scale_evidence,stain=a.stain)
  shutil.copy2(a.image,dest/('input'+a.image.suffix.lower()))
 else:
  local=read(HERE/(taskname+'_tasks_NO_GT.json'))['samples'][0];shutil.copy2(local['image'],dest/'input.png')
 task=dict(schema_version=1,samples=[original]);save(dest/'task_NO_GT.json',task)
 data=json.dumps(task,ensure_ascii=False)
 code=f'''from pathlib import Path
import subprocess,os,json,hashlib,tarfile,time
p=Path('{remote_dir}');out=p/'manual_outputs'/'{run_id}';out.parent.mkdir(exist_ok=True)
task=json.loads({data!r});taskpath=p/'manual_inputs'/'{run_id}.json';taskpath.parent.mkdir(exist_ok=True);taskpath.write_text(json.dumps(task))
assert hashlib.sha256(Path(task['samples'][0]['image']).read_bytes()).hexdigest()==task['samples'][0]['image_sha256']
live=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
gpus=[int(s.split(',')[0]) for s in live.splitlines() if int(s.split(',')[1])>12000];assert gpus,'No free GPU; no CPU fallback'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpus[0]),HF_HUB_OFFLINE='1')
cmd=[str(p/'tool_venv/bin/python'),'-u','-m','glomerulus_tools','run','--tool','{tid}','--task',str(taskpath),'--output',str(out),'--research']
t=time.monotonic();r=subprocess.run(cmd,cwd=p/'execution_snapshot',env=env,capture_output=True,text=True,timeout=300)
(out.parent/'{run_id}.log').write_text(r.stdout+r.stderr)
receipt=dict(run_id='{run_id}',exit_code=r.returncode,seconds=time.monotonic()-t,gpu=gpus[0],command=cmd,log=str(out.parent/'{run_id}.log'))
if r.returncode==0:
 assert json.loads((out/'completed.json').read_text())['status']=='complete'
 archive=out.with_suffix('.tgz')
 with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='results')
 receipt.update(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
print(json.dumps(receipt))
'''
 rec=json.loads(remote(code,timeout=340));save(dest/'execution_receipt.json',rec)
 if rec['exit_code']:raise RuntimeError('Test failed; see preserved lab log '+rec['log'])
 r=subprocess.run(['scp','-q','chenliangyu@10.103.238.124:'+rec['archive'],str(dest/'results.tgz')],capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
 assert sha(dest/'results.tgz')==rec['archive_sha256']
 with tarfile.open(dest/'results.tgz') as t:
  assert all(not n.startswith('/') and '..' not in Path(n).parts for n in t.getnames());t.extractall(dest)
 print('SUCCESS',tid,'GPU',rec['gpu'],'seconds',round(rec['seconds'],2));print('Local results:',dest)
 for f in (dest/'results').rglob('record.json'):print(f.read_text(encoding='utf8'))

if __name__=='__main__':main()
