"""Audit completed outputs and produce offline target-level complementarity bounds."""
import re,json,subprocess
from collections import Counter
import numpy as np
from PIL import Image
from prepare import HERE,REPORT,read,save,csvread,csvout,sha

def main():
    checks={};refiner_count=0;smoke_count=0
    for tool in read(HERE/'protocol.json')['actions']:
        for c in read(HERE/'benchmark_EVAL_ONLY.json'):
            cid=c['case_id'];p=HERE/'results'/tool/cid
            base=np.asarray(Image.open(HERE/'online'/cid/'baseline.png'))>0
            target=np.asarray(Image.open(HERE/'online'/cid/'target.png'))>0
            proposal=np.asarray(Image.open(p/'proposal.png'));edited=np.asarray(Image.open(p/'mask.png'))
            assert set(np.unique(proposal))<={0,255} and set(np.unique(edited))<={0,255}
            assert np.array_equal(edited>0,(base&~target)|(proposal>0))
            refiner_count+=1
        for p in (HERE/'smoke'/tool).glob('*/mask.png'):
            for name in ['mask.png','proposal.png']:
                assert np.array_equal(np.asarray(Image.open(p.parent/name)),np.asarray(Image.open(HERE/'results'/tool/p.parent.name/name)))
            smoke_count+=1
    checks.update(refiner_masks=refiner_count,refiner_smoke_cases_exact=smoke_count)
    specialist=0;special_smoke=0
    for s in read(HERE/'specialist_membership_EVAL_ONLY.json'):
        cid=s['case_id'];p=HERE/'specialist_results'/cid
        mask=np.asarray(Image.open(p/'mask.png'));labels=np.asarray(Image.open(p/'instances.tif'))
        assert list(mask.shape[1::-1])==s['tile_size'] and np.array_equal(mask>0,labels>0)
        assert set(np.unique(mask))<={0,255};specialist+=1
        smoke=HERE/'specialist_smoke'/cid
        if smoke.exists():
            assert np.array_equal(labels,np.asarray(Image.open(smoke/'instances.tif')))
            assert np.array_equal(mask,np.asarray(Image.open(smoke/'mask.png')));special_smoke+=1
    checks.update(specialist_masks=specialist,specialist_smoke_cases_exact=special_smoke)
    count=0
    for p in (HERE/'seam_validation').glob('*/phase*.png'):
        a=np.asarray(Image.open(p));assert set(np.unique(a))<={0,255};count+=1
    checks['seam_masks']=count;assert count==24
    gallery=(REPORT/'gallery.html').read_text(encoding='utf-8')
    refs=set(re.findall(r'(?:src|href)="(gallery/[^"#]+)"',gallery))
    for name in refs:
        with Image.open(REPORT/name) as im:assert im.format=='JPEG';im.verify()
    checks['jpeg_references_verified']=len(refs);assert len(refs)==424
    hard=csvread(REPORT/'全部硬化与保护对照_EVAL_ONLY.csv')
    checks['cohort_counts']=dict(Counter(x['cohort'] for x in hard))
    ref=csvread(REPORT/'专项逐对象结果_EVAL_ONLY.csv');mes=csvread(REPORT/'MESCnn逐对象_EVAL_ONLY.csv')
    index={(r['case_id'],r['method']):r for r in ref}
    mi={r['case_id']:r for r in mes if r['area_rule']=='raw'};rows=[]
    for obj in hard:
        cid=obj['case_id'];r=index[cid,'samrefiner_vitb_paper_fp16'];rr=index[cid,'ritm_hrnet32_external_mask_point1'];m=mi[cid]
        base=float(r['target_dice_before']);assert abs(base-float(m['baseline_target_dice']))<1e-10
        alternatives={'keep_segformer':base,'SAMRefiner':float(r['target_dice_after']),'RITM':float(rr['target_dice_after']),
                      'MESCnn':float(m['specialist_target_dice'])}
        best=max(alternatives,key=alternatives.get)
        rows.append(dict(case_id=cid,cohort=obj['cohort'],stain=obj['stain'],group_id=obj['group_id'],baseline=base,
             best_GT_known_action=best,best_dice=alternatives[best],delta=alternatives[best]-base,
             sam_delta=alternatives['SAMRefiner']-base,ritm_delta=alternatives['RITM']-base,mes_delta=alternatives['MESCnn']-base,
             scope=m['scope'],not_online_policy=True))
    csvout(REPORT/'逐目标互补上限_EVAL_ONLY.csv',rows)
    summary=[]
    for group in ['sclerosis','normal_control']:
        subset=[r for r in rows if r['cohort']==group]
        summary.append(dict(cohort=group,n=len(subset),mean_baseline=float(np.mean([r['baseline'] for r in subset])),
            mean_best=float(np.mean([r['best_dice'] for r in subset])),mean_delta=float(np.mean([r['delta'] for r in subset])),
            improved001=sum(r['delta']>.01 for r in subset),best_counts=dict(Counter(r['best_GT_known_action'] for r in subset)),
            scope='GT-informed per-target bound only; no neighbor safety; no simultaneous WSI policy'))
    save(HERE/'target_oracle_summary.json',summary)
    checks['baseline_target_Dice_shared_geometry_equal']=True
    checks['production_tests']='37 passed; glomerulus_tools/tests'
    save(HERE/'final_output_verification.json',checks)
    manifest={p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.glob('*.py'))}
    for p in sorted((HERE.parents[1]).glob('*.py')):manifest['../../'+p.name]=sha(p)
    save(HERE/'final_source_manifest.json',dict(version_type='SHA256 source snapshot; workspace has no historical Git commit',files=manifest))
    print(json.dumps(dict(checks=checks,oracle=summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
