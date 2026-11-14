"""Prepare metadata-scaled independent-network patches, labels kept offline."""
import importlib.util,re,time
from pathlib import Path
import cv2,numpy as np,tifffile
from PIL import Image
from prepare import ROOT,HERE,read,save,sha

spec=importlib.util.spec_from_file_location('old_prepare',HERE.parent/'problem_to_repair_20261111/prepare_cases.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
OLD=ROOT/'research/segformer_complementarity_20261106'
REMOTE='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
def main():
    assert not (HERE/'specialist_preparation.json').exists()
    start=time.monotonic();cv2.setNumThreads(1)
    cohort=read(HERE/'cohort_EVAL_ONLY.json');catalog={c['task']['task_id']:c['task'] for c in read(OLD/'catalog.json')}
    rows=[];tasks=[];metadata={}
    for o in cohort:
        tid=o['task_id'];t=catalog[tid];cid=o['case_id'];w,h=t['native_size']
        base=Path('D:/数据集/医院肾小球数据/2026_11_1_many/glomerulis_for_agents')/t['stain']
        image=base/'images'/Path(t['image']).name
        if tid not in metadata:
            with tifffile.TiffFile(image) as f:
                desc=f.pages[0].description
                match=re.search(r'AppMag\s*=\s*(\d+(?:\.\d+)?)',desc)
                assert match,tid
                objective=float(match.group(1));assert objective in [20.,40.],objective
                metadata[tid]=dict(objective=objective,description=desc,pyramid_shapes=[list(p.shape[:2]) for p in f.pages if p.ndim>=2][:5])
        objective=metadata[tid]['objective'];factor=objective/10;side=4096
        box=o.get('pred_box')
        scope='prediction_centered'
        if not box:
            box=__import__('json').loads(o['box']);scope='GT_centered_capability_reference_only'
        cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        x=int(cx-side/2);y=int(cy-side/2);xx=x+side;yy=y+side
        bounds=[max(0,x),max(0,y),min(w,xx),min(h,yy)]
        canvas=np.full((side,side,3),255,np.uint8)
        l,top,r,b=bounds;canvas[top-y:b-y,l-x:r-x]=prep.roi(image,bounds)
        tile=cv2.resize(canvas,(round(side/factor),round(side/factor)),interpolation=cv2.INTER_AREA)
        on=HERE/'specialist_online'/cid;on.mkdir(parents=True)
        Image.fromarray(tile).save(on/'image.png')
        # Evaluation uses full native geometry, independent of network output.
        ev=HERE/'specialist_evaluation'/cid;ev.mkdir(parents=True)
        gtmanifest=base/'instance_masks'/('instance_'+t['source_wsi_id'])/'wsi_gt_manifest.json'
        instances=read(gtmanifest)['slides'][0]['instances'];gt=np.zeros((side,side),bool);target=np.zeros_like(gt)
        for g in instances:
            gb=g['global_box_xyxy']
            if not prep.intersects([x,y,xx,yy],gb):continue
            gm=np.asarray(Image.open(gtmanifest.parent/g['mask_path']))>0
            ll,tt,rr,bb=max(x,gb[0]),max(y,gb[1]),min(xx,gb[2]),min(yy,gb[3])
            gt[tt-y:bb-y,ll-x:rr-x]|=gm[tt-gb[1]:bb-gb[1],ll-gb[0]:rr-gb[0]]
            if str(g['instance_id'])==o['instance_id']:
                target[tt-y:bb-y,ll-x:rr-x]=gm[tt-gb[1]:bb-gb[1],ll-gb[0]:rr-gb[0]]
                assert target.sum()==gm.sum(),(cid,'GT does not fit4096 reference patch')
        assert target.any()
        mask=ROOT/'research/full_networks_20261105/predictions/segformer_b5'/tid/'mask.tif'
        baseline=np.zeros_like(gt);baseline[top-y:b-y,l-x:r-x]=prep.roi(mask,bounds)>0
        np.savez_compressed(ev/'labels_EVAL_ONLY.npz',gt=gt,target=target,baseline=baseline)
        task=dict(task_id=cid,image=REMOTE+'/specialist_online/'+cid+'/image.png',species='human',stain=t['stain'],
                  target='whole_glomerular_unit',input_form='patch',native_size=list(tile.shape[1::-1]),
                  source_mpp=t['source_mpp']*factor,scale_basis='mescnn_metadata_10x_prepared_tile',
                  scale_evidence='SVS AppMag '+str(objective)+'; measured native MPP '+str(t['source_mpp'])+'; factor '+str(factor))
        tasks.append(task);rows.append(dict(case_id=cid,task_id=tid,cohort=o['cohort'],stain=t['stain'],source_gt_id=o['instance_id'],
                    scope=scope,box_native=[x,y,xx,yy],factor=factor,tile_size=list(tile.shape[1::-1]),
                    input_sha256=sha(on/'image.png'),metadata=metadata[tid]))
        print('SPECIALIST_PREPARED',cid,scope,tile.shape,flush=True)
    save(HERE/'specialist_batch_NO_GT.json',dict(schema='mescnn_prepared_tile_batch_v1',cases=tasks))
    save(HERE/'specialist_membership_EVAL_ONLY.json',rows)
    save(HERE/'specialist_preparation.json',dict(cases=len(rows),GT_centered_references=sum(r['scope'].startswith('GT_') for r in rows),
      source_scope='37 existing sclerosis candidates +40 normal controls;3 omitted sclerosis GT-centered reference only',
      adaptation='native4096 crop;metadata AppMag/10 downsample with INTER_AREA, not author OpenSlide pyramid JPEG chain; no full WSI/NMS',
      areas='raw and 5000um2 filtering evaluated separately',seconds=time.monotonic()-start))

if __name__=='__main__':main()
