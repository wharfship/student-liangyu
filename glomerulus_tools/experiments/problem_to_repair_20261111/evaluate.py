"""Offline paired evaluation; never imported by inference workers."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,json,time
import numpy as np
import cv2
from scipy.ndimage import binary_erosion,distance_transform_edt
from scipy.optimize import linear_sum_assignment
from PIL import Image
HERE=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def binary(p):return np.asarray(Image.open(p))>0
def counts(pred,gt):
 tp=int(np.count_nonzero(pred&gt));fp=int(np.count_nonzero(pred&~gt));fn=int(np.count_nonzero(gt&~pred))
 return dict(tp=tp,fp=fp,fn=fn,dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)
def boundary(pred,gt,mpp):
 ys,xs=np.where(pred|gt)
 if len(ys):
  sl=(slice(int(ys.min()),int(ys.max())+1),slice(int(xs.min()),int(xs.max())+1))
  pred=pred[sl];gt=gt[sl]
 a=pred&~binary_erosion(pred);b=gt&~binary_erosion(gt)
 if not a.any() or not b.any():return dict(hd95_um=None,bf1_1um=0.,bf1_2um=0.,bf1_5um=0.)
 da=distance_transform_edt(~b,sampling=mpp)[a];db=distance_transform_edt(~a,sampling=mpp)[b]
 out=dict(hd95_um=float(np.percentile(np.concatenate([da,db]),95)))
 for tol in [1,2,5]:
  precision=float(np.mean(da<=tol));recall=float(np.mean(db<=tol));out[f'bf1_{tol}um']=2*precision*recall/(precision+recall) if precision+recall else 0.
 return out
def matches(pred,instances,folder):
 n,labels,stats,_=cv2.connectedComponentsWithStats(pred.astype(np.uint8),8)
 mat=np.zeros((len(instances),n-1),np.float64)
 for i,g in enumerate(instances):
  gt=binary(folder/g['mask']);ids,inter=np.unique(labels[gt],return_counts=True)
  for label,value in zip(ids,inter):
   if label:mat[i,label-1]=value/(g['gt_pixels_full']+stats[label,cv2.CC_STAT_AREA]-value)
 out={}
 for threshold in [.5,.75]:
  # Match cardinality first, IoU second, using original IDs in the window.
  weights=(mat>=threshold).astype(float)*(len(instances)+1)+mat
  ii,jj=linear_sum_assignment(weights,maximize=True)
  out[str(threshold)]={instances[i]['id'] for i,j in zip(ii,jj) if mat[i,j]>=threshold}
 return out
def csvout(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def bootstrap(values,seed=20261111):
 a=np.asarray(values,float)
 if len(a)<2:return [None,None]
 rng=np.random.default_rng(seed);means=np.mean(a[rng.integers(0,len(a),(2000,len(a)))],axis=1)
 return np.quantile(means,[.025,.975]).tolist()
def main():
 p=argparse.ArgumentParser();p.add_argument('--methods',nargs='*');args=p.parse_args()
 cv2.setNumThreads(1);start=time.monotonic();cases=load(HERE/'benchmark_EVAL_ONLY.json');sources=load(HERE/'source_records_EVAL_ONLY.json')
 methods=args.methods or [x.name for x in sorted((HERE/'results').iterdir()) if (x/'completed.json').exists()]
 rows=[];all_images=[];summaries=[]
 for method in methods:
  out=HERE/'results'/method;complete=load(out/'completed.json');assert complete['cases']==len(cases)
  image_deltas=defaultdict(lambda:dict(tp=0,fp=0));method_rows=[]
  for s in cases:
   cid=s['case_id'];on=HERE/'online'/cid;ev=HERE/'evaluation'/cid;pred=out/cid
   base=binary(on/'baseline.png');target=binary(on/'target.png');gt=binary(ev/'gt.png')
   proposal=binary(pred/'proposal.png');edited=binary(pred/'mask.png');assert edited.shape==base.shape
   assert np.array_equal(edited,(base&~target)|proposal)
   original=counts(base,gt);new=counts(edited,gt);g=load(ev/'instances.json');target_g=next(x for x in g if x['id']==s['source_gt_id']);target_gt=binary(ev/target_g['mask'])
   original_target=counts(target,target_gt);new_target=counts(proposal,target_gt)
   # Original target entirely in window, and proposal is a target proposal, not global-instance matching.
   assert target_g['visible_pixels']==target_g['gt_pixels_full']
   before=matches(base,g,ev);after=matches(edited,g,ev)
   old_boundary=boundary(target,target_gt,s['source_mpp']);new_boundary=boundary(proposal,target_gt,s['source_mpp'])
   rec=load(pred/'record.json')
   delta_tp=new['tp']-original['tp'];delta_fp=new['fp']-original['fp']
   image_deltas[s['task_id']]['tp']+=delta_tp;image_deltas[s['task_id']]['fp']+=delta_fp
   changed=edited!=base;touches=bool(np.any(changed[[0,-1],:]) or np.any(changed[:,[0,-1]]))
   row=dict(method=method,case_id=cid,task_id=s['task_id'],dataset=s['dataset'],stain=s['stain'],stratum=s['stratum'],source_mpp=s['source_mpp'],
    local_dice_before=original['dice'],local_dice_after=new['dice'],delta_local_dice=new['dice']-original['dice'],
    target_dice_before=original_target['dice'],target_dice_after=new_target['dice'],delta_target_dice=new_target['dice']-original_target['dice'],
    delta_tp=delta_tp,delta_fp=delta_fp,delta_fn=-delta_tp,removed_old_tp=int(np.count_nonzero(base&~edited&gt)),
    added_fp=int(np.count_nonzero(edited&~base&~gt)),changed_pixels=int(changed.sum()),changes_touch_window_edge=touches,
    bf1_2um_before=old_boundary['bf1_2um'],bf1_2um_after=new_boundary['bf1_2um'],delta_bf1_2um=new_boundary['bf1_2um']-old_boundary['bf1_2um'],
    hd95_before_um=old_boundary['hd95_um'],hd95_after_um=new_boundary['hd95_um'],
    bf1_1um_after=new_boundary['bf1_1um'],bf1_5um_after=new_boundary['bf1_5um'],
    seconds=rec['seconds'],peak_allocated_mb=rec.get('peak_allocated_mb'),inference_gt_read=rec['gt_read'])
   for th in ['0.5','0.75']:
    row['window_ID_rescued_'+th]=len(after[th]-before[th]);row['window_ID_lost_'+th]=len(before[th]-after[th])
   method_rows.append(row)
  image_rows=[]
  for tid in sorted({s['task_id'] for s in cases}):
   b=sources[tid]['baseline'];dt=image_deltas[tid];tp=b['tp']+dt['tp'];fp=b['fp']+dt['fp'];fn=b['fn']-dt['tp']
   assert min(tp,fp,fn)>=0
   dice=2*tp/(2*tp+fp+fn)
   image_rows.append(dict(method=method,task_id=tid,dataset=sources[tid]['task']['dataset'],stain=sources[tid]['task']['stain'],
    baseline_dice=b['dice'],edited_dice=dice,delta_dice=dice-b['dice'],tp=tp,fp=fp,fn=fn,
    delta_tp=dt['tp'],delta_fp=dt['fp'],delta_fn=-dt['tp']))
  for dataset in ['ALL','hospital','hubmap2021']:
   ss=[x for x in image_rows if dataset=='ALL' or x['dataset']==dataset]
   cc=[x for x in method_rows if dataset=='ALL' or x['dataset']==dataset]
   ci=bootstrap([x['delta_dice'] for x in ss])
   summaries.append(dict(method=method,dataset=dataset,images=len(ss),windows=len(cc),baseline_dice=float(np.mean([x['baseline_dice'] for x in ss])),
    edited_dice=float(np.mean([x['edited_dice'] for x in ss])),delta_dice=float(np.mean([x['delta_dice'] for x in ss])),
    delta_dice_ci_low=ci[0],delta_dice_ci_high=ci[1],improved_images=sum(x['delta_dice']>1e-9 for x in ss),
    worsened_images=sum(x['delta_dice']<-1e-9 for x in ss),delta_tp=sum(x['delta_tp'] for x in cc),delta_fp=sum(x['delta_fp'] for x in cc),
    removed_old_tp=sum(x['removed_old_tp'] for x in cc),window_ID_rescued_05=sum(x['window_ID_rescued_0.5'] for x in cc),
    window_ID_lost_05=sum(x['window_ID_lost_0.5'] for x in cc),
    improved_windows=sum(x['delta_local_dice']>1e-6 for x in cc),worsened_windows=sum(x['delta_local_dice']<-1e-6 for x in cc),
    good_controls=len([x for x in cc if x['stratum']=='good_control']),
    harmed_good_controls=sum(x['stratum']=='good_control' and x['delta_target_dice']<-.01 for x in cc),
    mean_delta_target_bf1_2um=float(np.mean([x['delta_bf1_2um'] for x in cc])),seconds=sum(x['seconds'] for x in cc),
    max_peak_allocated_mb=max([x['peak_allocated_mb'] or 0 for x in cc])))
  rows+=method_rows;all_images+=image_rows
  print('EVALUATED',method,[(r['dataset'],round(r['delta_dice'],6)) for r in summaries[-3:]],flush=True)
 (HERE/'evaluation_results').mkdir(exist_ok=True)
 csvout(HERE/'evaluation_results/per_window.csv',rows);csvout(HERE/'evaluation_results/per_image.csv',all_images)
 csvout(HERE/'evaluation_results/summary.csv',summaries)
 details=[]
 for method in methods:
  for stratum in sorted({r['stratum'] for r in rows}):
   cc=[r for r in rows if r['method']==method and r['stratum']==stratum]
   details.append(dict(method=method,stratum=stratum,n=len(cc),mean_delta_target_dice=float(np.mean([x['delta_target_dice'] for x in cc])),
    mean_delta_local_dice=float(np.mean([x['delta_local_dice'] for x in cc])),mean_delta_bf1_2um=float(np.mean([x['delta_bf1_2um'] for x in cc])),
    target_improved_001=sum(x['delta_target_dice']>.01 for x in cc),target_worsened_001=sum(x['delta_target_dice']<-.01 for x in cc)))
 csvout(HERE/'evaluation_results/by_problem.csv',details)
 (HERE/'evaluation_results/verification.json').write_text(json.dumps(dict(methods=methods,cases=len(cases),source_images=len(sources),
  all_target_GT_fully_in_prediction_windows=True,all_proposals_equal_native_geometry=True,all_edit_composition_exact=True,
  full_image_dice_method='exact original TP/FP/FN plus changes in nonoverlapping native windows',
  instance_scope='original GT IDs matched against window-clipped components; global outside-window connectivity not reconstructed; not full-image instance safety',
  dataset_status='GT-stratified exposed development cohort; no independent generalization claim',seconds=time.monotonic()-start),indent=2))
if __name__=='__main__':main()
