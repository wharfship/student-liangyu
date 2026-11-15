"""Offline task-specific acceptance; GT never goes to model inference."""
import csv,numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from common import *

def csvout(path,rr):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)

def instance_score(pred,truth):
 _,gt=np.unique(truth,return_inverse=True);gt=gt.reshape(truth.shape)
 _,pd=np.unique(pred,return_inverse=True);pd=pd.reshape(pred.shape)
 g,p=int(gt.max()),int(pd.max());table=np.bincount((gt*(p+1)+pd).ravel(),minlength=(g+1)*(p+1)).reshape(g+1,p+1)
 ga=table.sum(1)[1:];pa=table.sum(0)[1:];inter=table[1:,1:];union=ga[:,None]+pa[None,:]-inter
 iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
 matched=0
 if g and p:
  # Maximize number of valid IoU>=.5 matches; break ties by total IoU.
  a,b=linear_sum_assignment(-(iou>=.5).astype(float)*10000-iou);matched=int((iou[a,b]>=.5).sum())
 tp=int(((pred>0)&(truth>0)).sum());dice=2*tp/max(1,int((pred>0).sum()+(truth>0).sum()))
 return dict(binary_dice=dice,GT_components=g,pred_instances=p,matched_iou05=matched,
  instance_precision=matched/p if p else 0.,instance_recall=matched/g if g else None,instance_f1=2*matched/(p+g) if p+g else 1.)

def main():
 REPORT.mkdir(parents=True,exist_ok=True);tasks={}
 for name in ['feature','nuclei','qc_paired','tnbc']:
  tasks[name]={s['sample_id']:s for s in read(HERE/(name+'_tasks_NO_GT.json'))['samples']}
 verification=[]
 for folder,name,count in [('nuclei_hospital16','nuclei',16),('nuclei_TNBC8','tnbc',8),('features77','feature',77),('tissue24','qc_paired',24),('artifact24','qc_paired',24)]:
  completed=read(HERE/'results'/folder/'completed.json');assert completed['status']=='complete' and completed['samples']==count
  for sid,s in tasks[name].items():
   rec=read(HERE/'results'/folder/sid/'record.json');assert rec['input_sha256']==sha(s['image'])==s['image_sha256'] and rec['GT_read'] is False
   if folder.startswith('nuclei'):
    a=np.asarray(Image.open(HERE/'results'/folder/sid/'instances.tif'));assert list(a.shape[1::-1])==rec['native_size'] and a.dtype.kind in 'ui'
   if folder=='features77':
    a=np.load(HERE/'results'/folder/sid/'embedding_l2.npy');assert a.shape==(384,) and np.isfinite(a).all() and abs(np.linalg.norm(a)-1)<1e-5
  verification.append(dict(folder=folder,samples=count,input_SHA=True,GT_inference=False))
 tnbc=[]
 for r in read(HERE/'tnbc_labels_EVAL_ONLY.json'):
  sid=r['sample_id'];pred=np.asarray(Image.open(HERE/'results/nuclei_TNBC8'/sid/'instances.tif'))
  raw=np.asarray(Image.open(r['gt']));truth=ndimage.label(raw>0)[0]
  assert pred.shape==truth.shape
  tnbc.append(dict(sample_id=sid,slide=r['slide'],**instance_score(pred,truth),
    GT_unit='binary annotated mask connected components; touching nuclei not guaranteed separate IDs',training_overlap=r['training_overlap']))
 csvout(REPORT/'TNBC8_细胞核验收_EVAL_ONLY.csv',tnbc)
 ts=dict(n=8,slides=len({r['slide'] for r in tnbc}),mean_binary_dice=float(np.mean([r['binary_dice'] for r in tnbc])),
  mean_instance_f1=float(np.mean([r['instance_f1'] for r in tnbc])),GT_components=sum(r['GT_components'] for r in tnbc),pred_instances=sum(r['pred_instances'] for r in tnbc),
  matched_iou05=sum(r['matched_iou05'] for r in tnbc),not_independent_test=True,reason='known training-source corpus; exact field split unknown;8 acceptance images only')
 links=read(HERE/'feature_labels_EVAL_ONLY.json');index={r['sample_id']:r for r in links};ids=list(index);X=np.stack([np.load(HERE/'results/features77'/sid/'embedding_l2.npy') for sid in ids])
 H=[]
 for sid in ids:
  im=np.asarray(Image.open(tasks['feature'][sid]['image']).convert('RGB'))
  v=np.concatenate([np.histogram(im[:,:,c],bins=16,range=(0,256))[0] for c in range(3)]).astype(float);H.append(v/np.linalg.norm(v))
 H=np.stack(H);retrieval=[]
 for model,features in [('DINOv2',X),('RGB_histogram',H)]:
  sim=features@features.T
  for i,sid in enumerate(ids):
   allowed=[j for j,other in enumerate(ids) if index[other]['group_id']!=index[sid]['group_id'] and index[other]['stain']==index[sid]['stain']]
   if not allowed:continue
   ranked=sorted(allowed,key=lambda j:-sim[i,j]);top=ranked[:5];target=index[sid]['cohort']
   retrieval.append(dict(model=model,sample_id=sid,group_id=index[sid]['group_id'],stain=index[sid]['stain'],cohort=target,
    neighbor_id=ids[top[0]],cosine=float(sim[i,top[0]]),same_class_top1=index[ids[top[0]]]['cohort']==target,
    same_class_precision_top5=float(np.mean([index[ids[j]]['cohort']==target for j in top])),reference_count=len(allowed),
    neighbors=json.dumps([ids[j] for j in top]),
    meaning='label-aware offline assay of image retrieval; not a trained or zero-shot disease classifier; GT-informed exposed cohort'))
 csvout(REPORT/'77候选_跨来源组同染色检索_EVAL_ONLY.csv',retrieval)
 rs=[]
 for model in ['DINOv2','RGB_histogram']:
  for stain in ['ALL','HE','PAS']:
   r=[x for x in retrieval if x['model']==model and (stain=='ALL' or x['stain']==stain)]
   balanced=np.mean([np.mean([x['same_class_top1'] for x in r if x['cohort']==group]) for group in ['sclerosis','normal_control']])
   rs.append(dict(model=model,stain=stain,n=len(r),top1_same_class=float(np.mean([x['same_class_top1'] for x in r])),balanced_top1_same_class=float(balanced),
     top5_same_class_precision=float(np.mean([x['same_class_precision_top5'] for x in r])),same_group_excluded=True,same_stain_only=True))
 quality=[]
 for pair in read(HERE/'qc_blur_linkage_EVAL_ONLY.json'):
  a=read(HERE/'results/artifact24'/pair['original']/'record.json');b=read(HERE/'results/artifact24'/pair['blur']/'record.json')
  quality.append(dict(sample_id=pair['original'],original_tissue=a['tissue_fraction'],blur_tissue=b['tissue_fraction'],original_OOF=a['class_fractions']['6'],blur_OOF=b['class_fractions']['6'],
   delta_OOF=b['class_fractions']['6']-a['class_fractions']['6'],original_artifact=a['artifact_fraction'],blur_artifact=b['artifact_fraction'],
   original_unmapped0=a['unmapped_class0_fraction'],blur_unmapped0=b['unmapped_class0_fraction'],synthetic=True,sigma_um=8.))
 csvout(REPORT/'GrandQC12对_人工严重模糊响应.csv',quality)
 qs=dict(pairs=12,mean_original_OOF=float(np.mean([r['original_OOF'] for r in quality])),mean_blur_OOF=float(np.mean([r['blur_OOF'] for r in quality])),
  increased=sum(r['delta_OOF']>0 for r in quality),increase_gt001=sum(r['delta_OOF']>.01 for r in quality),
  real_artifact_GT=False,not_realworld_sensitivity=True)
 # Independent smoke/full batches should reproduce without hiding numerical differences.
 repeats=[]
 for tid,folder,name in [('instanseg_brightfield_nuclei_v011','nuclei_hospital16','nuclei'),('dinov2_vits14_features','features77','feature'),('grandqc_tissue_mpp10','tissue24','qc_paired'),('grandqc_artifact_mpp15','artifact24','qc_paired')]:
  sid=next(iter(tasks[name]));filename={'instanseg_brightfield_nuclei_v011':'instances.tif','dinov2_vits14_features':'embedding.npy','grandqc_tissue_mpp10':'tissue_classes.png','grandqc_artifact_mpp15':'artifact_classes.png'}[tid]
  a=HERE/'smoke'/tid/sid/filename;b=HERE/'results'/folder/sid/filename
  repeats.append(dict(tool=tid,sample_id=sid,byte_equal=sha(a)==sha(b)))
 save(HERE/'acceptance_summary.json',dict(nuclei_TNBC=ts,retrieval=rs,quality_sanity=qs,repeatability=repeats,formal_samples=149))
 save(HERE/'evaluation_verification.json',dict(status='passed',batches=verification,samples=149,repeatability=repeats,GT_only_offline=True,reference_model_prediction_not_GT=True))
 print(json.dumps(dict(nuclei_TNBC=ts,retrieval=rs,quality_sanity=qs,repeatability=repeats),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
