"""Frozen, GT-labelled capability cohort; inference inputs use prediction geometry only."""
from pathlib import Path
from collections import Counter
import csv, hashlib, importlib.util, json, time
import cv2
import numpy as np
from PIL import Image
from inspect_sources import ROOT, read, csvread

HERE=Path(__file__).resolve().parent
AUDIT=ROOT/'已经整理好的内容/医院分割结果分析/重点failure失败原因'
OLD=ROOT/'glomerulus_tools/experiments/problem_to_repair_20261111'
REPORT=ROOT/'已经整理好的内容/医院分割结果分析/硬化专项验证_20261114'

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvout(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def object_key(o):return o['task_id'],str(o['instance_id'])

def main():
    start=time.monotonic();cv2.setNumThreads(1)
    assert not (HERE/'preparation.json').exists(),'Frozen cohort already prepared'
    spec=importlib.util.spec_from_file_location('old_prepare',OLD/'prepare_cases.py')
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    objects=csvread(AUDIT/'objects_EVAL_ONLY.csv')
    hospital=[o for o in objects if o['dataset']=='hospital']
    hard=[o for o in hospital if o['source_class']=='球性硬化']
    assert len(hard)==40
    old=read(OLD/'benchmark_EVAL_ONLY.json');oldkeys={object_key(x['original_object']):x['case_id'] for x in old}
    old_rows=[dict(case_id=c['case_id'],dataset=c['dataset'],stain=c['stain'],stratum=c['stratum'],
                   source_class=c['original_object']['source_class'],task_id=c['task_id'],
                   instance_id=c['source_gt_id']) for c in old]
    csvout(REPORT/'既有56窗病理类别对应_EVAL_ONLY.csv',old_rows)
    # Same-slide random normal controls; label and cohort selection are offline only.
    used=set();controls=[]
    for target in hard:
        pool=[o for o in hospital if o['source_class']=='正常肾小球' and o['matched_05']=='True'
              and object_key(o) not in used and o['task_id']==target['task_id']]
        if not pool:
            pool=[o for o in hospital if o['source_class']=='正常肾小球' and o['matched_05']=='True'
                  and object_key(o) not in used and o['stain']==target['stain']]
        pool.sort(key=lambda o:hashlib.sha256(('20261114:'+o['task_id']+':'+o['instance_id']).encode()).hexdigest())
        assert pool
        pick=pool[0];controls.append(pick);used.add(object_key(pick))
    catalog={c['task']['task_id']:c['task'] for c in read(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
    comps={(x['task_id'],int(x['component_id'])):x for x in csvread(AUDIT/'components_EVAL_ONLY.csv')}
    protocol=dict(date='2026-11-14',purpose='sclerosis-specific capability, not online detection or policy validation',
      sclerosis_population='all 40 hospital original sclerosis IDs; discarded class separate',normal_controls=40,
      controls='one deterministic matched normal per sclerosis ID, same WSI when available, otherwise same stain',
      candidate_selection='highest-IoU existing component for offline cohort association; no GT-created locations',
      box='same fixed prediction-derived recipe as old benchmark: max(512um, 2*longest predicted bbox)',
      unavailable='no overlapping predicted component => no refinement input; keep omission in denominator',
      actions=['samrefiner_vitb_paper_fp16','ritm_hrnet32_external_mask_point1'],
      tuning=False,gt_in_inference_payload=False,primary='paired per-target Dice and whole-window TP/FP/FN; normal harm',
      overlapping_windows='do not sum deltas or claim a simultaneous whole-slide policy score',
      inference_acceptance='all proposals evaluated; no automatic policy or Agent acceptance claimed')
    save(HERE/'protocol.json',protocol)
    manifest_cache={};sources={};selected=[];unavailable=[];cohort=[];online=[]
    for idx,o in enumerate(hard+controls,1):
        ishard=idx<=40;cid=f'S{idx:03d}' if ishard else f'N{idx-40:03d}'
        t=catalog[o['task_id']];tid=t['task_id'];mpp=t['source_mpp'];w,h=t['native_size']
        record=dict(case_id=cid,cohort='sclerosis' if ishard else 'normal_control',**o,
                    previous_case_id=oldkeys.get(object_key(o),''),pathology_review='not_performed')
        links=json.loads(o['links'])
        if not links:
            record.update(refinement_available=False,unavailable_reason='no_predicted_component; no GT prompt generated')
            unavailable.append(record);cohort.append(record);continue
        component_id=int(max(links,key=lambda r:r['iou'])['component_id'])
        component=comps[tid,component_id];pb=json.loads(component['box'])
        side=int(np.ceil(max(512/mpp,2*max(pb[2]-pb[0],pb[3]-pb[1]))))
        cx=(pb[0]+pb[2])/2;cy=(pb[1]+pb[3])/2
        box=[max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(w,int(cx+side/2)),min(h,int(cy+side/2))]
        if (box[2]-box[0])*(box[3]-box[1])>16_000_000:
            record.update(refinement_available=False,unavailable_reason='prediction-derived window exceeds registered size bound')
            unavailable.append(record);cohort.append(record);continue
        base=Path('D:/数据集/医院肾小球数据/2026_11_1_many/glomerulis_for_agents')/t['stain']
        image=base/'images'/Path(t['image']).name
        gtmanifest=base/'instance_masks'/('instance_'+t['source_wsi_id'])/'wsi_gt_manifest.json'
        mask=ROOT/'research/full_networks_20261105/predictions/segformer_b5'/tid/'mask.tif'
        frozen=read(ROOT/'research/segformer_complementarity_20261106/cases'/(tid+'.json'))
        if tid not in sources:
            masksha=sha(mask);gtsha=sha(gtmanifest)
            assert masksha==frozen['audit'][0]['mask_sha256']
            assert gtsha==frozen['instance_manifest_sha256']
            sources[tid]=dict(image=str(image),mask=str(mask),mask_sha256=masksha,manifest=str(gtmanifest),
                              manifest_sha256=gtsha,baseline=frozen['baseline'],task=t)
            manifest_cache[tid]=read(gtmanifest)['slides'][0]['instances']
        x,y,xx,yy=box;rgb=prep.roi(image,box);a=prep.roi(mask,box)>0
        n,labels,stats,_=cv2.connectedComponentsWithStats(a.astype(np.uint8),8)
        matches=[i for i in range(1,n) if stats[i,4]==int(component['pred_pixels'])
                 and [int(stats[i,0])+x,int(stats[i,1])+y,int(stats[i,0]+stats[i,2])+x,int(stats[i,1]+stats[i,3])+y]==pb]
        assert len(matches)==1,(cid,pb,matches)
        target=labels==matches[0];gt=np.zeros_like(a);metadata=[]
        on=HERE/'online'/cid;ev=HERE/'evaluation'/cid;on.mkdir(parents=True);ev.mkdir(parents=True)
        for item in manifest_cache[tid]:
            gb=item['global_box_xyxy']
            if not prep.intersects(box,gb):continue
            gm=np.asarray(Image.open(gtmanifest.parent/item['mask_path']))>0
            l,top,r,b=max(x,gb[0]),max(y,gb[1]),min(xx,gb[2]),min(yy,gb[3])
            q=np.zeros_like(a);q[top-y:b-y,l-x:r-x]=gm[top-gb[1]:b-gb[1],l-gb[0]:r-gb[0]]
            gt|=q;fn=f'instance_{len(metadata):03d}.png';Image.fromarray(q.astype('uint8')*255).save(ev/fn)
            metadata.append(dict(id=str(item['instance_id']),gt_pixels_full=int(gm.sum()),box=gb,
                                 visible_pixels=int(q.sum()),source_class=item.get('source_class_name','unknown'),mask=fn))
        truth=next(g for g in metadata if g['id']==o['instance_id'])
        Image.fromarray(rgb).save(on/'image.png');Image.fromarray(a.astype('uint8')*255).save(on/'baseline.png')
        Image.fromarray(target.astype('uint8')*255).save(on/'target.png');Image.fromarray(gt.astype('uint8')*255).save(ev/'gt.png')
        save(ev/'instances.json',metadata)
        task=dict(schema='local_mask_refinement_v1',case_id=cid,image=str(on/'image.png'),baseline=str(on/'baseline.png'),
                  target=str(on/'target.png'),native_size=[xx-x,yy-y],source_mpp=mpp,scale_basis=t['scale_basis'],
                  species='human',stain=t['stain'],target_box_xyxy=[pb[0]-x,pb[1]-y,pb[2]-x,pb[3]-y],global_box_xyxy=box)
        save(on/'task_NO_GT.json',task);online.append(task)
        record.update(refinement_available=True,component_id=component_id,pred_box=pb,window_box=box,
                      target_gt_visible_fraction=truth['visible_pixels']/truth['gt_pixels_full'])
        s=dict(case_id=cid,task_id=tid,dataset='hospital',stain=t['stain'],component_id=component_id,box=box,pred_box=pb,
               stratum='sclerosis' if ishard else 'normal_control',source_mpp=mpp,scale_basis=t['scale_basis'],
               group_id=t['group_id'],source_gt_id=o['instance_id'],original_object=o,
               target_gt_visible_fraction=record['target_gt_visible_fraction'],
               input_sha256={name:sha(on/name) for name in ['image.png','baseline.png','target.png','task_NO_GT.json']})
        selected.append(s);cohort.append(record);print('PREPARED',cid,record['cohort'],a.shape,flush=True)
    save(HERE/'benchmark_EVAL_ONLY.json',selected);save(HERE/'source_records_EVAL_ONLY.json',sources)
    save(HERE/'batch_NO_GT.json',dict(schema='local_mask_refinement_batch_v1',cases=online))
    csvout(REPORT/'全部硬化与保护对照_EVAL_ONLY.csv',cohort)
    csvout(REPORT/'不可修复候选_EVAL_ONLY.csv',unavailable)
    save(HERE/'cohort_EVAL_ONLY.json',cohort)
    result=dict(all_sclerosis=40,normal_controls=40,prepared_cases=len(selected),unavailable=len(unavailable),
                prepared_by_cohort=dict(Counter(s['stratum'] for s in selected)),
                previous_hospital_sclerosis_targets=sum(r['source_class']=='球性硬化' for r in old_rows if r['dataset']=='hospital'),
                sclerosis_wsis=len({o['task_id'] for o in hard}),sclerosis_groups=len({o['group_id'] for o in hard}),
                target_visible_min=min(s['target_gt_visible_fraction'] for s in selected),
                seconds=time.monotonic()-start,source_masks_and_manifests_verified=True)
    save(HERE/'preparation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
