"""Offline per-action evaluation; overlapping windows are never added as one policy."""
from collections import defaultdict
import importlib.util,json,time
import cv2,numpy as np
from prepare import ROOT,HERE,REPORT,read,csvout,save,sha

spec=importlib.util.spec_from_file_location('old_evaluation',HERE.parent/'problem_to_repair_20261111/evaluate.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)

def cluster_interval(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['group_id']].append(float(r[field]))
    values=list(groups.values())
    if len(values)<2:return [None,None]
    rng=np.random.default_rng(20261114);means=[]
    for _ in range(2000):
        pick=rng.integers(0,len(values),len(values));means.append(np.mean([v for i in pick for v in values[i]]))
    return np.quantile(means,[.025,.975]).tolist()

def main():
    cv2.setNumThreads(1);tick=time.monotonic()
    cases=read(HERE/'benchmark_EVAL_ONLY.json');cohort=read(HERE/'cohort_EVAL_ONLY.json')
    protocol=read(HERE/'protocol.json');allrows=[];summary=[]
    for method in protocol['actions']:
        out=HERE/'results'/method;completed=read(out/'completed.json');assert completed['cases']==len(cases)
        rows=[]
        for s in cases:
            cid=s['case_id'];on=HERE/'online'/cid;ev=HERE/'evaluation'/cid;dest=out/cid
            base=E.binary(on/'baseline.png');target=E.binary(on/'target.png');gt=E.binary(ev/'gt.png')
            proposal=E.binary(dest/'proposal.png');edited=E.binary(dest/'mask.png')
            assert base.shape==edited.shape and np.array_equal(edited,(base&~target)|proposal)
            rec=read(dest/'record.json');assert rec['gt_read'] is False
            for k in ['image','baseline','target']:assert sha(on/(k+'.png'))==rec['input_sha256'][k]
            inst=read(ev/'instances.json');g=next(g for g in inst if g['id']==s['source_gt_id'])
            assert g['visible_pixels']==g['gt_pixels_full']
            target_gt=E.binary(ev/g['mask']);old=E.counts(base,gt);new=E.counts(edited,gt)
            a=E.counts(target,target_gt);b=E.counts(proposal,target_gt)
            before=E.matches(base,inst,ev);after=E.matches(edited,inst,ev)
            ba=E.boundary(target,target_gt,s['source_mpp']);bb=E.boundary(proposal,target_gt,s['source_mpp'])
            # Explicitly record candidate-to-target identity; no posthoc removal from main results.
            overlap=[]
            for other in inst:
                truth=E.binary(ev/other['mask']);inter=int(np.count_nonzero(target&truth))
                overlap.append((inter/(int(target.sum())+other['gt_pixels_full']-inter),other['id']))
            associated=max(overlap)[1]
            row=dict(method=method,case_id=cid,task_id=s['task_id'],group_id=s['group_id'],stain=s['stain'],cohort=s['stratum'],
                source_class=s['original_object']['source_class'],instance_id=s['source_gt_id'],
                refinement_available=True,baseline_matched_05=s['original_object']['matched_05']=='True',
                baseline_coverage=float(s['original_object']['pixel_coverage']),
                target_dice_before=a['dice'],target_dice_after=b['dice'],delta_target_dice=b['dice']-a['dice'],
                local_dice_before=old['dice'],local_dice_after=new['dice'],delta_local_dice=new['dice']-old['dice'],
                delta_tp=new['tp']-old['tp'],delta_fp=new['fp']-old['fp'],
                removed_old_tp=int(np.count_nonzero(base&~edited&gt)),added_fp=int(np.count_nonzero(~base&edited&~gt)),
                target_coverage_after=int(np.count_nonzero(proposal&target_gt))/g['gt_pixels_full'],
                bf1_2um_before=ba['bf1_2um'],bf1_2um_after=bb['bf1_2um'],delta_bf1_2um=bb['bf1_2um']-ba['bf1_2um'],
                window_ID_rescued_05=len(after['0.5']-before['0.5']),window_ID_lost_05=len(before['0.5']-after['0.5']),
                own_id_before_05=g['id'] in before['0.5'],own_id_after_05=g['id'] in after['0.5'],
                neighbor_ids_lost_05=len((before['0.5']-after['0.5'])-{g['id']}),
                largest_iou_gt_id=associated,target_is_largest_iou_gt=associated==g['id'],
                changes_touch_window_edge=bool(np.any((edited!=base)[[0,-1],:]) or np.any((edited!=base)[:,[0,-1]])),
                seconds=rec['seconds'],peak_allocated_mb=rec['peak_allocated_mb'],gt_read=False)
            rows.append(row)
        for missing in [o for o in cohort if not o['refinement_available']]:
            rows.append(dict(method=method,case_id=missing['case_id'],task_id=missing['task_id'],group_id=missing['group_id'],
                stain=missing['stain'],cohort=missing['cohort'],source_class=missing['source_class'],instance_id=missing['instance_id'],
                refinement_available=False,baseline_coverage=0.,baseline_matched_05=False,target_dice_before=0.,target_dice_after=0.,
                delta_target_dice=0.,delta_local_dice=0.,delta_bf1_2um=0.,own_id_before_05=False,own_id_after_05=False,
                neighbor_ids_lost_05=0,window_ID_rescued_05=0,window_ID_lost_05=0,seconds=0.))
        for group in ['sclerosis','normal_control']:
            for stain in ['ALL','HE','PAS']:
                cc=[r for r in rows if r['cohort']==group and (stain=='ALL' or r['stain']==stain)]
                available=[r for r in cc if r['refinement_available']]
                ci=cluster_interval(cc,'delta_target_dice')
                summary.append(dict(method=method,cohort=group,stain=stain,n=len(cc),available=len(available),
                    source_wsis=len({r['task_id'] for r in cc}),source_groups=len({r['group_id'] for r in cc}),
                    mean_target_dice_before=float(np.mean([r['target_dice_before'] for r in cc])),
                    mean_target_dice_after=float(np.mean([r['target_dice_after'] for r in cc])),
                    mean_delta_target_dice=float(np.mean([r['delta_target_dice'] for r in cc])),
                    cluster95_low=ci[0],cluster95_high=ci[1],
                    target_improved_001=sum(r['delta_target_dice']>.01 for r in cc),
                    target_worsened_001=sum(r['delta_target_dice']<-.01 for r in cc),
                    mean_delta_bf1_2um=float(np.mean([r['delta_bf1_2um'] for r in cc])),
                    own_ids_before_05=sum(r['own_id_before_05'] for r in cc),own_ids_after_05=sum(r['own_id_after_05'] for r in cc),
                    own_ID_gross_lost_05=sum(r['own_id_before_05'] and not r['own_id_after_05'] for r in cc),
                    neighbor_harm_actions_05=sum(r['neighbor_ids_lost_05']>0 for r in cc),
                    mismatched_target_actions=sum(not r['target_is_largest_iou_gt'] for r in available),
                    seconds=sum(r['seconds'] for r in cc)))
        allrows.extend(rows)
        print('EVALUATED',method,[(r['cohort'],r['mean_delta_target_dice'],r['target_worsened_001']) for r in summary[-6:] if r['stain']=='ALL'],flush=True)
    csvout(REPORT/'专项逐对象结果_EVAL_ONLY.csv',allrows);csvout(REPORT/'专项分组结果_EVAL_ONLY.csv',summary)
    save(HERE/'evaluation_summary.json',summary)
    save(HERE/'evaluation_verification.json',dict(cases=len(cases),methods=protocol['actions'],all_input_shas_equal=True,
      all_target_gt_fully_visible=True,edit_composition_exact=True,inference_GT_read=False,
      unavailable_omissions_retained_in_sclerosis_denominator=True,
      instance_scope='window-clipped components and original GT IDs, not full-slide instance safety',
      whole_slide_policy_dice='not computed: prediction-derived windows may overlap; no double-counted sum',
      cohort='all40 sclerosis +40 deterministic normals; exposed development, patient identity unknown',
      seconds=time.monotonic()-tick))

if __name__=='__main__':main()
