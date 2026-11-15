"""Freeze small acceptance examples before looking at predictions."""
from common import *
from PIL import Image
import numpy as np,cv2

def main():
 base=HERE/'assets/acceptance/TNBC_and_Brain_dataset';tasks=[];labels=[]
 # First2 lexicographic fields from each of first4 slides, fixed pre-inference.
 slides=sorted(x for x in base.iterdir() if x.is_dir() and x.name.startswith('Slide_'))[:4]
 for slide in slides:
  files=sorted(x for x in slide.iterdir() if x.suffix.lower() in {'.png','.tif','.tiff'})[:2]
  for f in files:
   gt=base/slide.name.replace('Slide_','GT_')/f.name;assert gt.is_file(),gt
   sid=f'TNBC{len(tasks)+1:02d}';dest=HERE/'inputs'/sid;dest.mkdir(parents=True,exist_ok=True)
   with Image.open(f) as im:im.convert('RGB').save(dest/'image.png')
   tasks.append(dict(sample_id=sid,image=str(dest/'image.png'),image_sha256=sha(dest/'image.png'),source_mpp=.25,scale_evidence='author InstanSeg TNBC loader specifies0.25um/px; dataset protocol assumption, not remeasured',stain='HE'))
   labels.append(dict(sample_id=sid,gt=str(gt),slide=slide.name,source=str(f),training_overlap='source corpus used to train InstanSeg; exact field split unknown; not independent test'))
 save(HERE/'tnbc_tasks_NO_GT.json',dict(schema_version=1,samples=tasks));save(HERE/'tnbc_labels_EVAL_ONLY.json',labels)
 # Artificial defocus sanity check is labelled synthetic, never real artifact GT.
 qc=read(HERE/'qc_tasks_NO_GT.json');paired=[];links=[]
 for s in qc['samples']:
  image=np.asarray(Image.open(s['image']).convert('RGB'));blur=cv2.GaussianBlur(image,(0,0),8/s['source_mpp'])
  f=Path(s['image']).with_name('context_severe_blur.png');Image.fromarray(blur).save(f)
  b=dict(s,sample_id=s['sample_id']+'BLUR',image=str(f),image_sha256=sha(f));paired.extend([s,b]);links.append(dict(original=s['sample_id'],blur=b['sample_id'],sigma_um=8.,synthetic=True))
 save(HERE/'qc_paired_tasks_NO_GT.json',dict(schema_version=1,samples=paired));save(HERE/'qc_blur_linkage_EVAL_ONLY.json',links)
 print('Frozen TNBC',len(tasks),'and quality sanity',len(paired),'inputs')
if __name__=='__main__':main()
