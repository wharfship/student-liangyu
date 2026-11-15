"""Create image-only acceptance inputs; offline cohort membership stays separate."""
import numpy as np,cv2,hashlib
from PIL import Image
from common import *

def main():
 old=HERE.parent/'sclerosis_focus_20261114';cohort=read(old/'cohort_EVAL_ONLY.json')
 samples=[];links=[];nuclei=[];qc=[];cv2.setNumThreads(1)
 benchmark={r['case_id']:r for r in read(old/'benchmark_EVAL_ONLY.json')}
 for i,o in enumerate(cohort,1):
  cid=o['case_id'];dest=HERE/'inputs'/f'P{i:03d}';dest.mkdir(parents=True,exist_ok=True)
  if o['refinement_available']:
   src=old/'online'/cid;im=np.asarray(Image.open(src/'image.png').convert('RGB'));mask=np.asarray(Image.open(src/'target.png'))>0
   yy,xx=np.where(mask);cx=(xx.min()+xx.max())//2;cy=(yy.min()+yy.max())//2
   side=max(int(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)*1.2),64)
   # The benchmark has measured original MPP and predicted supports; no GT is read to crop.
   meta=benchmark[cid]
   mpp=meta['source_mpp'];pad=side//2
   x=max(0,int(cx-pad));y=max(0,int(cy-pad));right=min(im.shape[1],int(cx+pad));bottom=min(im.shape[0],int(cy+pad))
   crop=im[y:bottom,x:right];Image.fromarray(crop).save(dest/'object.png')
   samples.append(dict(sample_id=f'P{i:03d}',image=str(dest/'object.png'),image_sha256=sha(dest/'object.png'),source_mpp=mpp,scale_evidence=meta['scale_basis'],stain=o['stain']))
   links.append(dict(sample_id=f'P{i:03d}',case_id=cid,cohort=o['cohort'],group_id=o['group_id'],stain=o['stain'],selection='predicted component crop; cohort GT-informed development'))
 # 12 HE +4 exploratory PAS crops, fixed evenly across original groups, no output-score selection.
 selected=[r for stain,n in [('HE',6),('PAS',2)] for group in ['sclerosis','normal_control']
  for r in sorted([x for x in links if x['stain']==stain and x['cohort']==group],key=lambda x:hashlib.sha256(x['sample_id'].encode()).hexdigest())[:n]]
 featuremap={r['sample_id']:r for r in samples}
 for r in selected:
  sample=dict(featuremap[r['sample_id']]);src=old/'online'/r['case_id']
  im=np.asarray(Image.open(src/'image.png').convert('RGB'));mask=np.asarray(Image.open(src/'target.png'))>0
  yy,xx=np.where(mask);cx=int((xx.min()+xx.max())/2);cy=int((yy.min()+yy.max())/2)
  # Nuclei are evaluated at a native 512px subwindow around a predicted object.
  x=max(0,min(cx-256,im.shape[1]-512));y=max(0,min(cy-256,im.shape[0]-512))
  path=HERE/'inputs'/r['sample_id']/'nuclei_native.png';Image.fromarray(im[y:y+512,x:x+512]).save(path)
  sample.update(image=str(path),image_sha256=sha(path));nuclei.append(sample)
 tasks={r['task_id']:r for r in read(old/'specialist_batch_NO_GT.json')['cases']}
 for r in selected:
  if r['stain']!='HE':continue
  cid=r['case_id'];t=tasks[cid];path=HERE/'inputs'/r['sample_id']/'context.png'
  __import__('shutil').copy2(old/'specialist_online'/cid/'image.png',path)
  qc.append(dict(sample_id=r['sample_id'],image=str(path),image_sha256=sha(path),source_mpp=t['source_mpp'],scale_evidence=t['scale_evidence'],stain='HE'))
 save(HERE/'feature_tasks_NO_GT.json',dict(schema_version=1,samples=samples))
 save(HERE/'feature_labels_EVAL_ONLY.json',links)
 save(HERE/'nuclei_tasks_NO_GT.json',dict(schema_version=1,samples=nuclei))
 save(HERE/'qc_tasks_NO_GT.json',dict(schema_version=1,samples=qc))
 save(HERE/'pilot_preparation.json',dict(feature_samples=len(samples),excluded_complete_omissions=3,GT_not_used_for_crop=True,
  cohort_membership_GT_informed=True,selection='all77 located targets,20% predicted bbox padding',nuclei=len(nuclei),qc=len(qc)))
 print('Prepared feature crops',len(samples),flush=True)

if __name__=='__main__':main()
