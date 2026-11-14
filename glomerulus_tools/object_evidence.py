"""Research-only object evidence; probabilities never authorize mask edits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .catalog import get_card, sha
from .classification import validate_task, dump


def crop_square(image, box, side):
    from PIL import Image
    cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
    x=round(cx-side/2);y=round(cy-side/2)
    result=Image.new('RGB',(side,side),'white')
    bounds=[max(0,x),max(0,y),min(image.width,x+side),min(image.height,y+side)]
    result.paste(image.crop(bounds),(bounds[0]-x,bounds[1]-y))
    return result,[x,y,x+side,y+side]


def prepare_gncnn_input(crop, card):
    """Match the author's Java4096/downsample1 + TorchScript Resize800 path."""
    from PIL import Image
    if card.get('input_recipe')!='author_qupath4096_resize800' or crop.size!=(4096,4096):
        raise ValueError('GNCnn requires the audited4096 crop and explicit input recipe')
    # torchvision Resize(800) uses PIL bilinear for this square RGB input.
    return crop.resize((800,800),Image.Resampling.BILINEAR)


def run(task,card,output):
    import numpy as np
    import torch
    import torchvision
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    images=validate_task(task,card)
    if not card['enabled'] or card['agent_enabled']:raise ValueError('This protocol is research only')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():raise ValueError('Select an available CUDA GPU')
    if card.get('checkpoint_sha256') and sha(card['checkpoint'])!=card['checkpoint_sha256']:
        raise ValueError('Registered checkpoint hash mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    dump(output/'online_task_NO_GT.json',task)
    dump(output/'runtime_card.json',card)
    start=time.perf_counter();kind=card['model_kind'];audit={}
    if kind=='gncnn_detection':
        # No map_location: traced CPU shape constants must remain on CPU.
        model=torch.jit.load(card['checkpoint']).eval()
        audit['schema']=str(model.forward.schema)
    elif kind=='osaka_detection':
        from ultralytics import YOLO
        model=YOLO(card['checkpoint'])
        expected={0:'Glo',1:'GS',2:'Cres',3:'SegSc'}
        if model.names!=expected:raise ValueError('Unexpected author YOLO classes')
        audit.update(class_names=model.names,confidence_floor=.001,nms_iou=.7)
    elif kind=='hats_cap_probe':
        sys.path.insert(0,str(Path(card['source_dir'])/'EfficientSAM_token_dynamichead_logits'))
        from efficient_sam.efficient_sam import build_efficient_sam
        model=build_efficient_sam(384,6,15,4,checkpoint=None)
        weights=torch.load(card['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(weights,strict=True);model=model.cuda().eval()
        audit.update(strict_load=True,task_id=7,scale_token=0,scale_mapping='unknown; token0 experimental, not an author protocol reproduction')
    else:raise ValueError('Unsupported evidence model')
    loaded=time.perf_counter();records=[];cache={};seen=set()
    for c in task['candidates']:
        meta=images[c['image_id']]
        if c['image_id'] not in cache:
            if sha(meta['image'])!=meta['image_sha256']:raise ValueError('Source hash mismatch')
            cache={c['image_id']:Image.open(meta['image']).convert('RGB')};seen.add(c['image_id'])
        source=cache[c['image_id']]
        if list(source.size)!=meta['native_size']:raise ValueError('Image size mismatch')
        detector=kind in ('gncnn_detection','osaka_detection')
        side=card['native_crop_side'] if detector else round(1024/meta['source_mpp'])
        crop,cb=crop_square(source,c['box'],side)
        row=dict(schema_version='glomerulus_object_evidence_v1',tool_id=card['id'],candidate_id=c['candidate_id'],
            image_id=c['image_id'],candidate_box=c['box'],crop_box=cb,source_mpp=meta['source_mpp'],
            scale_basis=meta['scale_basis'],scale_evidence=meta['scale_evidence'],
            source_sha256=meta['image_sha256'],crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
            model_mpp='unknown' if detector else '2um/px nominal paper; token mapping unknown',
            boundary_assessed=False,edit_authorized=False,accuracy_verified=False,decision='review_only',gt_read=False)
        visible_width=max(0,min(source.width,cb[2])-max(0,cb[0]))
        visible_height=max(0,min(source.height,cb[3])-max(0,cb[1]))
        row['source_image_fraction']=visible_width*visible_height/side**2
        b=c['box'];inter=max(0,min(b[2],cb[2])-max(b[0],cb[0]))*max(0,min(b[3],cb[3])-max(b[1],cb[1]))
        row['candidate_box_coverage']=inter/((b[2]-b[0])*(b[3]-b[1]))
        torch.cuda.synchronize();t=time.perf_counter()
        if detector:
            if kind=='gncnn_detection':
                pixels=800
                crop=prepare_gncnn_input(crop,card)
                x=torch.from_numpy(np.asarray(crop)[:,:,::-1].copy().astype('float32').transpose(2,0,1))
                row.update(input_recipe=card['input_recipe'],model_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                    model_bgr_float32_chw_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
                    tensor_channels='BGR',tensor_dtype='float32',tensor_range=[float(x.min()),float(x.max())],
                    source_pixels_per_model_pixel=side/pixels,
                    effective_input_mpp=meta['source_mpp']*side/pixels,
                    effective_input_mpp_basis=meta['scale_basis'],required_training_mpp='unknown')
                if card.get('save_input_previews'):
                    preview=output/'input_previews';preview.mkdir(exist_ok=True)
                    crop.save(preview/(c['candidate_id']+'.jpg'),quality=95)
                    row['input_preview_jpeg']='input_previews/'+c['candidate_id']+'.jpg'
                    row['preview_is_inference_input']=False
                with torch.inference_mode():out=model(x)
                boxes,classes,patches,scores,shape=out
                if shape.tolist()!=[800,800] or not bool((classes==0).all()):raise ValueError('Detector mapping mismatch')
                boxes=boxes.cpu().numpy();scores=scores.cpu().numpy();classes=classes.cpu().numpy()
            else:
                pixels=640
                crop=crop.resize((680,680),Image.Resampling.LANCZOS).resize((640,640),Image.Resampling.LANCZOS)
                with torch.inference_mode():out=model.predict(crop,imgsz=640,conf=.001,iou=.7,device=0,verbose=False)[0]
                boxes=out.boxes.xyxy.cpu().numpy();scores=out.boxes.conf.cpu().numpy();classes=out.boxes.cls.cpu().numpy().astype(int)
            boxes=boxes*side/pixels+np.array([cb[0],cb[1],cb[0],cb[1]])
            b=np.array(c['box']);left=np.maximum(boxes[:,:2],b[:2]);right=np.minimum(boxes[:,2:],b[2:])
            inter=np.prod(np.maximum(0,right-left),axis=1);area=np.prod(boxes[:,2:]-boxes[:,:2],axis=1);ba=np.prod(b[2:]-b[:2])
            iou=inter/np.maximum(area+ba-inter,1)
            match=iou>=.1
            names={0:'glomerulus'} if kind=='gncnn_detection' else model.names
            row.update(detections=[dict(box=z.tolist(),class_id=int(cl),class_name=names[int(cl)],score=float(s),candidate_box_iou=float(v)) for z,s,v,cl in zip(boxes,scores,iou,classes)],
                context_max_score=float(scores.max()) if len(scores) else 0.,
                candidate_matched_score=float(scores[match].max()) if match.any() else 0.,
                score_semantics='max detector score among boxes with candidate box IoU>=0.1; zero means no matched detection, not certain non-glomerulus',
                model_input_pixels=[pixels,pixels],native_crop_side=side,undersampling=1 if kind=='gncnn_detection' else side/680,
                preprocessing='native4096 square white padded ->800 PIL bilinear;BGR float32 0..255;internal author normalization;author JPEG encoding not replicated' if kind=='gncnn_detection' else 'native5440 ->680 ->640 Lanczos RGB; author YOLO normalization; no stain translation')
            arrays=dict(boxes=boxes,scores=scores,classes=classes)
            if kind=='gncnn_detection':arrays['roi_mask_probabilities']=patches.cpu().numpy()
            np.savez_compressed(output/(c['candidate_id']+'_instances.npz'),**arrays)
        else:
            crop=crop.resize((512,512),Image.Resampling.BILINEAR)
            x=torch.from_numpy(np.asarray(crop).copy().transpose(2,0,1)).float().cuda()[None]/255
            probs=[]
            with torch.inference_mode():
                for dims in [(),(-1,),(-2,),(-2,-1)]:
                    xx=x.flip(dims) if dims else x
                    logits=model(xx,torch.tensor([7.],device='cuda'),torch.tensor([0.],device='cuda'))
                    p=logits.softmax(1)[:,1]
                    probs.append((p.flip(dims) if dims else p)[0].cpu().numpy())
            probs=np.stack(probs);p=probs.mean(0);mask=p>=.5
            b=np.array(c['box']);b=(b-np.array([cb[0],cb[1],cb[0],cb[1]]))*512/side
            x1,y1,x2,y2=np.clip(np.rint(b).astype(int),0,512);x2=max(x2,x1+1);y2=max(y2,y1+1)
            q=p[y1:y2,x1:x2];variance=probs.var(0);entropy=-(p*np.log(np.maximum(p,1e-7))+(1-p)*np.log(np.maximum(1-p,1e-7)))
            Image.fromarray(mask.astype('uint8')*255).save(output/(c['candidate_id']+'_mask.png'))
            np.savez_compressed(output/(c['candidate_id']+'_probabilities.npz'),mean=p.astype('float16'),variance=variance.astype('float16'))
            row.update(candidate_mean_probability=float(q.mean()),candidate_foreground_fraction=float((q>=.5).mean()),
                candidate_entropy=float(entropy[y1:y2,x1:x2].mean()),candidate_tta_variance=float(variance[y1:y2,x1:x2].mean()),
                tta_binary_disagreement=float(((probs>=.5).any(0)!=(probs>=.5).all(0))[y1:y2,x1:x2].mean()),
                mask_path=c['candidate_id']+'_mask.png',mask_to_source=dict(box=cb,pixels=[512,512]),
                task_id=7,scale_token=0,scale_token_mapping='unknown',model_input_pixels=[512,512],
                score_semantics='mean foreground probability inside candidate bounding box; not calibrated identity probability',
                preprocessing='1024um nominal square using supplied source MPP ->512 bilinear; RGB/255; internal ImageNet normalization; four flip TTA')
        torch.cuda.synchronize();row['forward_seconds']=time.perf_counter()-t;records.append(row)
        with (output/'evidence.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(c['candidate_id'],row['forward_seconds'],flush=True)
    dump(output/'inference_record.json',dict(valid=True,gt_read=False,count=len(records),images=len(seen),tool_id=card['id'],
        output_sha256=sha(output/'evidence.jsonl'),checkpoint_sha256=sha(card['checkpoint']),task_sha256=hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),
        model_load_seconds=loaded-start,total_seconds=time.perf_counter()-start,forward_seconds=sum(r['forward_seconds'] for r in records),
        device=torch.cuda.get_device_name(),physical_device=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cuda=torch.version.cuda,
        runner_sha256=sha(__file__),catalog_sha256=sha(Path(__file__).parent/'catalog.csv'),
        api_cost_usd=0,hardware_cost='unknown; institutional GPU not billed through API',audit=audit,
        edit_authorized=False,accuracy_verified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tool',required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(json.loads(a.task.read_text(encoding='utf-8-sig')),get_card(a.tool),a.output)
