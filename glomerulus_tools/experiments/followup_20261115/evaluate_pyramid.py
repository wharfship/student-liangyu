"""Paired input-chain intervention; same checkpoint, boxes and offline targets."""
import numpy as np,cv2
from PIL import Image
from common import *
from evaluate import E,cluster_interval

def main():
 cv2.setNumThreads(1);cohort={r['case_id']:r for r in read(OLD/'cohort_EVAL_ONLY.json')}
 previous={r['case_id']:r for r in rows(ROOT/'已经整理好的内容/医院分割结果分析/硬化专项验证_20261114/MESCnn逐对象_EVAL_ONLY.csv') if r['area_rule']=='raw'}
 prep={r['case_id']:r for r in read(HERE/'pyramid_preparation.json')['records']};allrows=[];summaries=[]
 for mode in ['PNG','JPG']:
  completed=read(HERE/('pyramid_results_'+mode)/'completed.json');assert completed['status']=='complete' and completed['cases']==80
  for cid,o in cohort.items():
   dest=HERE/('pyramid_results_'+mode)/(cid+mode);rec=read(dest/'record.json')
   image=HERE/'pyramid_inputs'/cid/('image.png' if mode=='PNG' else 'image.jpg')
   assert sha(image)==rec['input_sha256'] and rec['gt_read'] is False and rec['task_checkpoint_strict_load']
   labels=np.asarray(Image.open(dest/'instances.tif'));mask=np.asarray(Image.open(dest/'mask.png'))
   assert np.array_equal(mask>0,labels>0) and list(labels.shape[1::-1])==prep[cid]['size']
   pred=cv2.resize(labels,(4096,4096),interpolation=cv2.INTER_NEAREST)
   with np.load(OLD/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:truth=f['target'];gt=f['gt'];base=f['baseline']
   ids=np.unique(pred[truth]);ids=ids[ids>0];scores=[]
   for label in ids:
    m=pred==label;inter=int(np.count_nonzero(m&truth));scores.append((inter/(int(m.sum())+int(truth.sum())-inter),int(label)))
   best=max(scores,default=(0.,0));pick=pred==best[1] if best[1] else np.zeros_like(truth)
   dice=E.counts(pick,truth)['dice'];before=float(previous[cid]['baseline_target_dice']);old=float(previous[cid]['specialist_target_dice'])
   window=E.counts(pred>0,gt);bwin=E.counts(base,gt)
   allrows.append(dict(mode=mode,case_id=cid,group_id=o['group_id'],cohort=o['cohort'],stain=o['stain'],scope=prep[cid]['scope'],
      baseline_target_dice=before,previous_native_AREA_dice=old,new_target_dice=dice,delta_vs_baseline=dice-before,delta_vs_previous=dice-old,
      matched05=best[0]>=.5,baseline_matched05=o['matched_05']=='True',best_instance_iou=best[0],
      window_dice=window['dice'],delta_window_vs_baseline=window['dice']-bwin['dice'],
      delta_fp=window['fp']-bwin['fp'],delta_tp=window['tp']-bwin['tp'],
      instance_count=rec['instance_count'],seconds=rec['network_inference_seconds'],
      original_vs_pyramid_rgb_MAE=prep[cid]['rgb_mean_abs_difference'],GT_inference=False))
  for group in ['sclerosis','normal_control']:
   for stain in ['ALL','HE','PAS']:
    rr=[r for r in allrows if r['mode']==mode and r['cohort']==group and (stain=='ALL' or r['stain']==stain)]
    ci=cluster_interval(rr,'delta_vs_baseline');cii=cluster_interval(rr,'delta_vs_previous')
    summaries.append(dict(mode=mode,cohort=group,stain=stain,n=len(rr),
       baseline=float(np.mean([r['baseline_target_dice'] for r in rr])),previous=float(np.mean([r['previous_native_AREA_dice'] for r in rr])),
       new=float(np.mean([r['new_target_dice'] for r in rr])),delta_vs_baseline=float(np.mean([r['delta_vs_baseline'] for r in rr])),
       ci95=ci,delta_vs_previous=float(np.mean([r['delta_vs_previous'] for r in rr])),previous_ci95=cii,
       improved001=sum(r['delta_vs_baseline']>.01 for r in rr),worsened001=sum(r['delta_vs_baseline']<-.01 for r in rr),
       matched05=sum(r['matched05'] for r in rr),baseline_matched05=sum(r['baseline_matched05'] for r in rr),
       rescued05=sum(r['matched05'] and not r['baseline_matched05'] for r in rr),
       gross_lost05=sum(not r['matched05'] and r['baseline_matched05'] for r in rr)))
 csvout(REPORT/'金字塔配对逐目标_EVAL_ONLY.csv',allrows);save(HERE/'paired_summary.json',summaries)
 # JPEG effect evaluated on precisely the same pyramid pixels.
 pairs={r['case_id']:r for r in allrows if r['mode']=='PNG'}
 for r in allrows:
  if r['mode']=='JPG':r['delta_JPEG_vs_PNG']=r['new_target_dice']-pairs[r['case_id']]['new_target_dice']
 csvout(REPORT/'金字塔配对逐目标_EVAL_ONLY.csv',allrows)
 save(HERE/'evaluation_verification.json',dict(cases=80,recipes=2,outputs=160,input_SHA_equal=True,
    instance_union_binary_equal=True,GT_inference=False,all_author_level_checks_pass=True,
    target_comparison='best predicted instance by offline target IoU; not online selection or global matching',
    same_native_reference_box=True,rounding='native4096 / rounded factor2 or4; actual pyramid dimension ratios recorded; subpixel differences not removed',
    full_WSI_search=False,three_GT_centered_references_not_deployment=True))
 for s in summaries:
  if s['stain']=='ALL':print(s)

if __name__=='__main__':main()
