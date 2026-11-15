"""Retrieve and score the fixed224um control without changing chosen tool weights."""
from common import *
import subprocess,tarfile,numpy as np
from PIL import Image

def main():
 dest=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115'
 code=f'''from pathlib import Path
import tarfile,hashlib,json
p=Path('{dest}');assert json.loads((p/'results/fixed_fov_features77/completed.json').read_text())['samples']==77
with tarfile.open(p/'fixed_results.tgz','w:gz') as t:t.add(p/'results/fixed_fov_features77',arcname='fixed_fov_features77');t.add(p/'fixed_fov.log',arcname='fixed_fov.log')
print(json.dumps(dict(sha256=hashlib.sha256((p/'fixed_results.tgz').read_bytes()).hexdigest())))
'''
 rec=__import__('json').loads(remote(code));r=subprocess.run(['scp','-q','chenliangyu@10.103.238.124:'+dest+'/fixed_results.tgz',str(HERE/'fixed_results.tgz')],capture_output=True,text=True,timeout=60);assert r.returncode==0,r.stderr;assert sha(HERE/'fixed_results.tgz')==rec['sha256']
 with tarfile.open(HERE/'fixed_results.tgz') as t:t.extractall(HERE/'results')
 links=read(HERE/'feature_labels_EVAL_ONLY.json');ids=[x['sample_id'] for x in links];byid={r['sample_id']:r for r in links};X=np.stack([np.load(HERE/'results/fixed_fov_features77'/sid/'embedding_l2.npy') for sid in ids])
 tasks=read(HERE/'fixed_fov_tasks_NO_GT.json')['samples'];H=[]
 for s in tasks:
  r=read(HERE/'results/fixed_fov_features77'/s['sample_id']/'record.json');assert r['input_sha256']==sha(s['image']) and r['GT_read'] is False
  im=np.asarray(Image.open(s['image']).convert('RGB'));v=np.concatenate([np.histogram(im[:,:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(float);H.append(v/np.linalg.norm(v))
 rows=[];summaries=[]
 for model,features in [('DINOv2',X),('RGB_histogram',np.stack(H))]:
  sim=features@features.T
  for i,sid in enumerate(ids):
   allowed=[j for j,other in enumerate(ids) if byid[other]['group_id']!=byid[sid]['group_id'] and byid[other]['stain']==byid[sid]['stain']]
   rank=sorted(allowed,key=lambda j:-sim[i,j])[:5]
   rows.append(dict(model=model,sample_id=sid,group_id=byid[sid]['group_id'],cohort=byid[sid]['cohort'],stain=byid[sid]['stain'],neighbor=ids[rank[0]],same_class_top1=byid[sid]['cohort']==byid[ids[rank[0]]]['cohort'],
    top5_precision=float(np.mean([byid[sid]['cohort']==byid[ids[j]]['cohort'] for j in rank]))))
  for stain in ['ALL','HE','PAS']:
   rr=[r for r in rows if r['model']==model and (stain=='ALL' or r['stain']==stain)]
   summaries.append(dict(model=model,stain=stain,n=len(rr),top1_same_class=float(np.mean([r['same_class_top1'] for r in rr])),top5_same_class_precision=float(np.mean([r['top5_precision'] for r in rr]))))
 from evaluate import csvout
 csvout(REPORT/'固定224微米_同染色跨组检索_EVAL_ONLY.csv',rows)
 save(HERE/'fixed_fov_control_summary.json',dict(fixed_FOV_um=224,summaries=summaries,GT_only_offline=True,training_not_performed=True,independent_test=False))
 print(__import__('json').dumps(summaries,indent=2))

if __name__=='__main__':main()
