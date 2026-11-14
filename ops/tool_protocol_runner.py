"""No-GT callable segmentation tools with explicit species, stain and scale contracts."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def espnet(task, card, output, candidates):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(card['source_dir'])/'module/espnet/test'))
    from Model import ESPNet
    model=ESPNet(classes=5,p=2,q=8).cuda().eval()
    model.load_state_dict(torch.load(card['checkpoint'],map_location='cpu'),strict=True)
    image=cv2.imread(task['image']);mask=cv2.imread(str(candidates),0)
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError('ESPNet requires an existing image-aligned predicted candidate mask')
    mpp=task['source_mpp_for_crops']
    n,_,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8))
    components=sorted([i for i in range(1,n) if stats[i,4]*mpp*mpp>=5000],key=lambda i:stats[i,4],reverse=True)
    if len(components)>64:
        raise ValueError('More than 64 candidates: split into smaller regions explicitly')
    margin=round(20/mpp);merged=np.zeros(image.shape[:2],np.uint8);class1=merged.copy();records=[]
    with torch.inference_mode():
        for i in components:
            x,y,w,h,area=stats[i];x0=max(0,x-margin);y0=max(0,y-margin)
            x1=min(image.shape[1],x+w+margin);y1=min(image.shape[0],y+h+margin)
            crop=image[y0:y1,x0:x1];data=crop.astype(np.float32)
            for c,(mean,std) in enumerate(zip(card['mean_bgr'],card['std_bgr'])):
                data[...,c]=(data[...,c]-mean)/std
            tensor=torch.from_numpy((cv2.resize(data,(1024,512))/255).transpose(2,0,1).copy()).unsqueeze(0).cuda()
            logits=model(tensor)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite ESPNet logits')
            classes=cv2.resize(logits.argmax(1)[0].cpu().numpy().astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)
            index=len(records);dest=output/f'crop_{index:02d}';dest.mkdir()
            cv2.imwrite(str(dest/'image.png'),crop);cv2.imwrite(str(dest/'classes.png'),classes)
            merged[y0:y1,x0:x1]|=(classes>0).astype(np.uint8)
            class1[y0:y1,x0:x1]|=(classes==1).astype(np.uint8)
            clipped=bool(x-margin<0 or y-margin<0 or x+w+margin>image.shape[1] or y+h+margin>image.shape[0])
            records.append({'box':[int(x0),int(y0),int(x1),int(y1)],
                            'requested_margin_clipped':clipped,
                            'candidate_touches_input_boundary':bool(x==0 or y==0 or x+w==image.shape[1] or y+h==image.shape[0]),
                            'class_pixels':{str(k):int((classes==k).sum()) for k in range(5)}})
    cv2.imwrite(str(output/'mask.png'),merged*255);cv2.imwrite(str(output/'class1_mask.png'),class1*255)
    (output/'crops.json').write_text(json.dumps(records,indent=2))
    return {'valid':True,'mask_path':str(output/'mask.png'),'crops':len(records),
            'context_review_required':any(r['requested_margin_clipped'] for r in records),
            'context_limited_crops':sum(r['requested_margin_clipped'] for r in records),
            'complete_anatomy_guaranteed':False,
            'candidate_mask':str(candidates),'independent_localizer':False,'bn_mode':'author_stored_statistics'}


def run(tool_id, task, output, candidates=None, research=False):
    from glomerulus_tools.catalog import get_card,sha
    from glomerulus_tools.contract import bind_tool_scale, compatibility, prepare_task, validate_task
    from glomerulus_tools.adapters import finalize_record
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU required, no CPU fallback')
    card=get_card(tool_id)
    tool_id=card['id']
    task=validate_task(task)
    if card.get('adapter')!='legacy':task=prepare_task(task)
    task=bind_tool_scale(card,task)
    reasons=compatibility(card,task,research=research)
    if reasons:raise ValueError('Incompatible tool/task: '+','.join(reasons))
    if card.get('checkpoint_sha256') and sha(card['checkpoint'])!=card['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from registered tool identity')
    for path,digest in card.get('detector_checkpoint_sha256',{}).items():
        if sha(path)!=digest:raise ValueError('Detector checkpoint SHA256 differs from registered tool identity')
    for field, key in [('species','species'),('stain','stains')]:
        if task[field] not in card[key]:raise ValueError('Incompatible '+field)
    if task.get('target','whole_glomerular_unit')!='whole_glomerular_unit':raise ValueError('Target mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'inference_record.json').exists():raise FileExistsError('Preserve previous result; use a new output directory')
    (output/'input_protocol.json').write_text(json.dumps({'tool':card,'task':task,'gt_read':False},indent=2))
    start=time.monotonic()
    if card.get('adapter') in ('nepath','qupath','likiseg','glo_matrix'):
        from glomerulus_tools.adapters import infer
        result=infer(card,task,output)
    elif tool_id=='omni_full':
        from ops.validate_native_tools import omni_author
        result=omni_author(task,card,output,full=True)
    elif tool_id=='glo2_whole':
        from ops.glo2_tool import infer
        result=infer(task,card,output)
    elif tool_id.startswith('ki2021_'):
        from ops.run_ki_recovery_tool import infer
        result=infer(task,card,output)
    elif tool_id=='tokyo_espnet_pas':
        if not candidates:raise ValueError('Predicted candidate mask required, never GT')
        result=espnet(task,card,output,candidates)
    elif tool_id=='capybara_kpis':
        import cv2
        import numpy as np
        from mmseg.apis import init_model,inference_model
        from mmengine.registry import init_default_scope
        init_default_scope('mmseg')
        model=init_model(card['config'],card['checkpoint'],device='cuda:0')
        model.cfg.test_pipeline=[dict(type='LoadImageFromNDArray'),dict(type='PackSegInputs')]
        image=cv2.imread(task['image'])
        if task.get('dataset_id')=='kpis_wsi_roi':
            h,w=image.shape[:2];patch=2048;stride=1024
            if min(h,w)<patch:raise ValueError('WSI region smaller than author2048 patch')
            # Author outer aggregation sums patch softmax. Final softmax preserves argmax.
            votes=np.zeros((2,h,w),np.float32);windows=[]
            for x in sorted(set(list(range(0,w-patch+1,stride))+[w-patch])):
                for y in sorted(set(list(range(0,h-patch+1,stride))+[h-patch])):
                    crop=image[y:y+patch,x:x+patch]
                    if len(np.unique(crop))==1:continue
                    logits=inference_model(model,crop).seg_logits.data
                    if not torch.isfinite(logits).all():raise ValueError('Nonfinite Capybara logits')
                    votes[:,y:y+patch,x:x+patch]+=torch.softmax(logits,dim=0).cpu().numpy()
                    windows.append([x,y,x+patch,y+patch])
            pred=votes.argmax(0)
            (output/'outer_windows.json').write_text(json.dumps(windows))
        else:
            prediction=inference_model(model,image)
            if not torch.isfinite(prediction.seg_logits.data).all():raise ValueError('Nonfinite Capybara logits')
            pred=prediction.pred_sem_seg.data.squeeze(0).cpu().numpy()
        if set(np.unique(pred))-{0,1}:raise ValueError('Unexpected classes')
        cv2.imwrite(str(output/'mask.png'),pred.astype(np.uint8)*255)
        result={'valid':True,'mask_path':str(output/'mask.png'),'test_cfg':dict(model.cfg.model.test_cfg)}
    else:raise ValueError('Unsupported tool')
    result.update(tool_id=tool_id,task_id=task['task_id'],status='completed' if result['valid'] else 'invalid',
                  seconds=time.monotonic()-start,gt_read=False,torch=torch.__version__,cuda=torch.version.cuda)
    if result['valid']:result=finalize_record(card,task,output,result)
    (output/'inference_record.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tool',required=True);parser.add_argument('--task-json',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--candidates')
    parser.add_argument('--research',action='store_true')
    args=parser.parse_args()
    task=json.loads(Path(args.task_json).read_text())
    print(json.dumps(run(args.tool,task,args.output,args.candidates,args.research)),flush=True)
