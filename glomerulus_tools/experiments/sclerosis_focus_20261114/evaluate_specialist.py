"""Independent instance output versus SegFormer in shared native reference windows."""
import numpy as np,cv2
from PIL import Image
from prepare import HERE,REPORT,read,save,csvout,sha
from evaluate import E,cluster_interval

def main():
    cv2.setNumThreads(1)
    complete=read(HERE/'specialist_results/completed.json');assert complete['cases']==80 and complete['strict_checkpoint_load']
    cohort={r['case_id']:r for r in read(HERE/'cohort_EVAL_ONLY.json')}
    members=read(HERE/'specialist_membership_EVAL_ONLY.json');rows=[]
    for s in members:
        cid=s['case_id'];o=cohort[cid];out=HERE/'specialist_results'/cid
        rec=read(out/'record.json');assert rec['gt_read'] is False and rec['input_sha256']==sha(HERE/'specialist_online'/cid/'image.png')
        labels=np.asarray(Image.open(out/'instances.tif'))
        assert list(labels.shape[1::-1])==s['tile_size']
        native=cv2.resize(labels.astype('uint16'),(4096,4096),interpolation=cv2.INTER_NEAREST)
        with np.load(HERE/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:
            gt=f['gt'];target=f['target'];baseline=f['baseline']
        before=E.counts(baseline,gt)
        # Baseline counterpart is selected for evaluation only; no GT feeds inference.
        n,oldlabels,stats,_=cv2.connectedComponentsWithStats(baseline.astype('uint8'),8)
        overlap=np.bincount(oldlabels[target],minlength=n);overlap[0]=0
        oldpick=oldlabels==int(overlap.argmax()) if overlap.max() else np.zeros_like(target)
        oldtarget=E.counts(oldpick,target)
        source_mpp=float(o['area_um2'])**.5/float(o['gt_pixels'])**.5
        # Author area output is polygon-based; this is pixel-area sensitivity, not
        # an exact reimplementation of QuPath polygon filtering.
        areas=np.bincount(native.ravel())*source_mpp**2
        for filter_name in ['raw','pixel_area_ge5000um2_sensitivity']:
            pred=native>0
            if filter_name!='raw':pred&=areas[native]>=5000
            ids=np.unique(native[pred]);ids=ids[ids>0]
            scores=[]
            for i in ids:
                m=native==i;inter=int(np.count_nonzero(m&target))
                scores.append((inter/(int(m.sum())+int(target.sum())-inter),int(i)))
            best=max(scores,default=(0.,0));pick=native==best[1] if best[1] else np.zeros_like(target)
            after=E.counts(pred,gt);t=E.counts(pick,target)
            rows.append(dict(method='mescnn_cascade_r50',case_id=cid,task_id=s['task_id'],group_id=o['group_id'],
                cohort=o['cohort'],stain=o['stain'],scope=s['scope'],area_rule=filter_name,
                source_class=o['source_class'],instance_id=o['instance_id'],baseline_matched_05=o['matched_05']=='True',
                baseline_target_dice=oldtarget['dice'],specialist_target_dice=t['dice'],delta_target_dice=t['dice']-oldtarget['dice'],
                specialist_best_instance_iou=best[0],specialist_matched05=best[0]>=.5,
                baseline_window_dice=before['dice'],specialist_window_dice=after['dice'],delta_window_dice=after['dice']-before['dice'],
                delta_tp=after['tp']-before['tp'],delta_fp=after['fp']-before['fp'],
                instance_count_raw=rec['instance_count'],instances_retained=len(ids),
                inference_seconds=rec['network_inference_seconds'],strict_load=True,GT_read=False))
    summaries=[]
    for cohort_name in ['sclerosis','normal_control']:
        for scope in ['ALL','prediction_centered','GT_centered_capability_reference_only']:
            for area in ['raw','pixel_area_ge5000um2_sensitivity']:
                cc=[r for r in rows if r['cohort']==cohort_name and r['area_rule']==area and (scope=='ALL' or r['scope']==scope)]
                if not cc:continue
                ci=cluster_interval(cc,'delta_target_dice')
                summaries.append(dict(cohort=cohort_name,scope=scope,area_rule=area,n=len(cc),
                    baseline_matched05=sum(r['baseline_matched_05'] for r in cc),specialist_matched05=sum(r['specialist_matched05'] for r in cc),
                    rescued05=sum(not r['baseline_matched_05'] and r['specialist_matched05'] for r in cc),
                    gross_lost05=sum(r['baseline_matched_05'] and not r['specialist_matched05'] for r in cc),
                    mean_baseline_target_dice=float(np.mean([r['baseline_target_dice'] for r in cc])),
                    mean_specialist_target_dice=float(np.mean([r['specialist_target_dice'] for r in cc])),
                    mean_delta_target_dice=float(np.mean([r['delta_target_dice'] for r in cc])),
                    cluster95_low=ci[0],cluster95_high=ci[1],
                    target_improved001=sum(r['delta_target_dice']>.01 for r in cc),target_worsened001=sum(r['delta_target_dice']<-.01 for r in cc),
                    mean_delta_window_dice=float(np.mean([r['delta_window_dice'] for r in cc]))))
    csvout(REPORT/'MESCnn逐对象_EVAL_ONLY.csv',rows);csvout(REPORT/'MESCnn分组_EVAL_ONLY.csv',summaries)
    save(HERE/'specialist_evaluation_summary.json',summaries)
    save(HERE/'specialist_evaluation_verification.json',dict(cases=80,input_shas_equal=True,strict_load=True,
      target_matching='best IoU over predicted instances, offline target diagnostic, not online identity selector',
      fully_omitted='3 GT-centered windows are capability-only references, not deployment rescue',
      area_filter='pixel-area5000 sensitivity, not exact polygon QuPath implementation',
      no_WSI_policy_or_global_matching=True,source_status='exposed development, training overlap unknown'))
    for r in summaries:
        if r['area_rule']=='raw':print(r)

if __name__=='__main__':main()
