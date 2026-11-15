"""Inspect leakage/shortcuts and run a fixed-physical-FOV retrieval control."""
from common import *
import numpy as np,cv2,hashlib,subprocess,tarfile,csv
from PIL import Image

def main():
 old=HERE.parent/'sclerosis_focus_20261114';links=read(HERE/'feature_labels_EVAL_ONLY.json');bb={r['case_id']:r for r in read(old/'benchmark_EVAL_ONLY.json')}
 original=read(HERE/'feature_tasks_NO_GT.json')['samples'];source={s['sample_id']:s for s in original};hashes={};tasks=[];details=[]
 for link in links:
  sid=link['sample_id'];s=source[sid];im=np.array(Image.open(s['image']).convert('RGB'));digest=hashlib.sha256(im.tobytes()).hexdigest();hashes.setdefault(digest,[]).append(sid)
  meta=bb[link['case_id']];p=old/'online'/link['case_id'];raw=np.array(Image.open(p/'image.png').convert('RGB'));mask=np.asarray(Image.open(p/'target.png'))>0
  y,x=np.where(mask);cx=int((x.min()+x.max())/2);cy=int((y.min()+y.max())/2);side=round(224/meta['source_mpp']);assert side<=min(raw.shape[:2])
  left=max(0,min(cx-side//2,raw.shape[1]-side));top=max(0,min(cy-side//2,raw.shape[0]-side))
  out=HERE/'fixed_fov_inputs'/sid/'image.png';out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(raw[top:top+side,left:left+side]).save(out)
  tasks.append(dict(s,image=str(out),image_sha256=sha(out)));details.append(dict(sample_id=sid,physical_FOV_um=224,native_side=side,MPP=meta['source_mpp'],group_id=link['group_id']))
 duplicates=[ids for ids in hashes.values() if len(ids)>1];assert not duplicates,duplicates
 audit=dict(exact_RGB_duplicates=duplicates,raw_image_SHA_unique=77,cohort_labels_only_offline=True,selection='all77 already located exposed development targets',
  crop_basis='predicted support only; noGT mask overlays',fixed_FOV_control_um=224,not_independent_validation=True)
 save(HERE/'feature_input_audit.json',audit);save(HERE/'fixed_fov_geometry_EVAL_ONLY.json',details)
 save(HERE/'fixed_fov_tasks_NO_GT.json',dict(schema_version=1,samples=tasks))
 dest=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115'
 for s in tasks:s['image']=s['image'].replace(str(HERE),dest).replace('\\','/')
 save(HERE/'fixed_fov_remote_NO_GT.json',dict(schema_version=1,samples=tasks))
 archive=HERE/'fixed_fov.tgz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:t.add(HERE/'fixed_fov_inputs',arcname='fixed_fov_inputs');t.add(HERE/'fixed_fov_remote_NO_GT.json',arcname='fixed_fov_remote_NO_GT.json')
 r=subprocess.run(['scp','-q',str(archive),'chenliangyu@10.103.238.124:'+dest+'/fixed_fov.tgz'],capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
 code=f'''from pathlib import Path
import json,tarfile,hashlib,subprocess,os
p=Path('{dest}')
assert hashlib.sha256((p/'fixed_fov.tgz').read_bytes()).hexdigest()=='{sha(archive)}'
with tarfile.open(p/'fixed_fov.tgz') as t:t.extractall(p)
free=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
cards=[int(s.split(',')[0]) for s in free.splitlines() if int(s.split(',')[1])>12000];assert cards
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(cards[0]),HF_HUB_OFFLINE='1')
cmd=[str(p/'tool_venv/bin/python'),'-u','-m','glomerulus_tools','run','--tool','dinov2_vits14_features','--task',str(p/'fixed_fov_remote_NO_GT.json'),'--output',str(p/'results/fixed_fov_features77'),'--research']
log=(p/'fixed_fov.log').open('w');r=subprocess.Popen(cmd,cwd=p/'execution_snapshot',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=r.pid,command=cmd,gpu=cards[0])))
'''
 d=__import__('json').loads(remote(code));save(HERE/'fixed_fov_start.json',d);print(d)

if __name__=='__main__':main()
