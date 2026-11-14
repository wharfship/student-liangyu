import numpy as np
from PIL import Image
from prepare import HERE,REPORT,read,save,csvout
from evaluate import E

OLD=HERE.parent/'problem_to_repair_20261111'
def main():
    completed=read(HERE/'seam_validation/completed.json');assert completed['cases']==12
    assert read(HERE/'seam_validation/exit.json')['exit_code']==0
    membership=read(HERE/'seam_evaluation_membership.json')
    cases={c['case_id']:c for c in read(OLD/'benchmark_EVAL_ONLY.json')};rows=[]
    for rec in completed['records']:
        cid=rec['case_id'];c=cases[cid];ev=OLD/'evaluation'/cid;on=OLD/'online'/cid
        gt=E.binary(ev/'gt.png');base=E.binary(on/'baseline.png')
        a=E.binary(HERE/'seam_validation'/cid/'phase0.png');b=E.binary(HERE/'seam_validation'/cid/'phase256.png')
        assert np.array_equal(a,base),(cid,'original native crop mismatch')
        instances=read(ev/'instances.json');target=E.binary(ev/next(g for g in instances if g['id']==c['source_gt_id'])['mask'])
        n,labs=__import__('cv2').connectedComponents(b.astype('uint8'),8)
        intersections=np.bincount(labs[target],minlength=n);intersections[0]=0
        # Largest-overlap result is an offline per-target diagnostic, not a selector.
        pick=labs==int(intersections.argmax()) if intersections.max()>0 else np.zeros_like(b)
        oldtarget=E.binary(on/'target.png');before=E.counts(base,gt);after=E.counts(b,gt)
        oldt=E.counts(oldtarget,target);newt=E.counts(pick,target)
        ma=E.matches(a,instances,ev);mb=E.matches(b,instances,ev)
        rows.append(dict(case_id=cid,group=membership[cid],task_id=c['task_id'],source_class=c['original_object']['source_class'],
            before_window_dice=before['dice'],after_window_dice=after['dice'],delta_window_dice=after['dice']-before['dice'],
            before_target_dice=oldt['dice'],after_target_dice=newt['dice'],delta_target_dice=newt['dice']-oldt['dice'],
            delta_tp=after['tp']-before['tp'],delta_fp=after['fp']-before['fp'],
            window_ID_rescued05=len(mb['0.5']-ma['0.5']),window_ID_lost05=len(ma['0.5']-mb['0.5']),
            GT_read_inference=False,seconds=rec['seconds']))
    csvout(REPORT/'接缝反事实逐例_EVAL_ONLY.csv',rows)
    summary=[]
    for group in sorted(set(r['group'] for r in rows)):
        ss=[r for r in rows if r['group']==group]
        summary.append(dict(group=group,n=len(ss),mean_delta_window_dice=float(np.mean([r['delta_window_dice'] for r in ss])),
          mean_delta_target_dice=float(np.mean([r['delta_target_dice'] for r in ss])),
          target_improved001=sum(r['delta_target_dice']>.01 for r in ss),target_worsened001=sum(r['delta_target_dice']<-.01 for r in ss),
          window_ID_rescued05=sum(r['window_ID_rescued05'] for r in ss),window_ID_lost05=sum(r['window_ID_lost05'] for r in ss)))
    save(HERE/'seam_evaluation_summary.json',summary);print(summary)

if __name__=='__main__':main()
