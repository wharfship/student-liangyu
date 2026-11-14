"""Research-only, image-and-prediction local mask refinement. No GT inputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from .catalog import get_card, sha

KEYS={'schema','case_id','image','baseline','target','native_size','source_mpp','scale_basis',
      'species','stain','target_box_xyxy','global_box_xyxy'}

def validate_task(task,card):
    if task.get('schema')=='local_mask_refinement_batch_v1':
        if set(task)!={'schema','cases'} or not 1<=len(task['cases'])<=128:
            raise ValueError('Invalid bounded refinement batch')
        ids=[]
        for case in task['cases']:
            validate_task(case,card);ids.append(case['case_id'])
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate case ID')
        return
    if set(task)!=KEYS or task['schema']!='local_mask_refinement_v1':
        raise ValueError('Unexpected fields; use image/prediction-only refinement schema')
    if task['species'] not in card['species'] or task['stain'] not in card['stains']:
        raise ValueError('Incompatible species/stain')
    if not task['case_id'].isalnum():raise ValueError('Invalid case ID')
    w,h=task['native_size'];x,y,xx,yy=task['target_box_xyxy']
    if not (0<x+1<=xx<=w and 0<y+1<=yy<=h and w*h<=16_000_000 and task['source_mpp']>0):
        raise ValueError('Invalid prediction geometry/scale')
    if not all(Path(task[k]).is_file() for k in ['image','baseline','target']):
        raise ValueError('Missing online image/mask')

def main():
    p=argparse.ArgumentParser();p.add_argument('--tool',required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    card=get_card(args.tool);task=json.loads(args.task.read_text(encoding='utf-8-sig'));validate_task(task,card)
    if not card['enabled'] or card['adapter']!='mask_refinement':raise ValueError('Disabled or incorrect adapter')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Set CUDA_VISIBLE_DEVICES after live resource check')
    import torch
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20261111)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback is not enabled')
    torch.cuda.set_per_process_memory_fraction(card.get('cuda_memory_fraction',.08),0)
    torch.backends.cudnn.benchmark=False
    for path in reversed(card.get('extra_paths',[])):sys.path.insert(0,path)
    if sha(card['checkpoint'])!=card['checkpoint_sha256']:raise ValueError('Checkpoint differs from registered identity')
    start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    records=[];engine=card['engine']
    if engine=='cascade_psp':
        from segmentation_refinement import Refiner
        model=Refiner(device='cuda:0',model_folder=str(Path(card['checkpoint']).parent),download_and_check_model=False)
    elif engine=='ritm':
        from .ritm_refinement import load_ritm, predict_ritm, coarse_interior_point
        model,ritm_config=load_ritm(card['checkpoint'],max_size=card['max_size'])
    elif engine in ('sam','sam_hq','sam_refiner'):
        if engine in ('sam','sam_refiner'):from segment_anything import sam_model_registry,SamPredictor
        else:from segment_anything_hq import sam_model_registry,SamPredictor
        network=sam_model_registry['vit_b'](checkpoint=card['checkpoint']).to('cuda:0').eval()
        from .chunked_sam_attention import install,verify
        attention_check=verify(type(network.image_encoder.blocks[0].attn))
        patched_blocks=install(network.image_encoder)
        model=network if engine=='sam_refiner' else SamPredictor(network)
        if engine=='sam_refiner':
            import FastGeodis
            from sam_refiner import sam_refiner
    else:raise ValueError('Unknown refinement engine')
    loaded=time.monotonic()-start
    cases=task['cases'] if task['schema'].endswith('batch_v1') else [task]
    for t in cases:
        case_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        if engine=='sam_refiner':
            # Variable-size embedding upsampling can fragment cached CUDA blocks.
            torch.cuda.empty_cache()
        image=np.asarray(Image.open(t['image']).convert('RGB'))
        base=np.asarray(Image.open(t['baseline']))>0;target=np.asarray(Image.open(t['target']))>0
        if image.shape[:2]!=base.shape or base.shape!=target.shape or list(image.shape[1::-1])!=t['native_size']:
            raise ValueError('Input sizes do not match native geometry')
        if np.any(target&~base) or not target.any():raise ValueError('Target must be a nonempty part of baseline')
        with torch.inference_mode():
            if engine=='cascade_psp':
                # Official package quick start uses cv2.imread (BGR).
                proposal=model.refine(image[:,:,::-1].copy(),target.astype(np.uint8)*255,
                                      fast=card['fast'],L=card['L'])>127
                score=None
            elif engine=='ritm':
                points=coarse_interior_point(target)
                probability=predict_ritm(model,image,target,points)
                proposal=probability>=.5;score=None
            elif engine=='sam_refiner':
                with torch.autocast('cuda',dtype=torch.float16):
                    masks,scores,_=sam_refiner(t['image'],[target.astype(np.uint8)],model,
                        iters=card['iterations'],margin=card['box_margin'],gamma=card['gamma'],
                        strength=card['mask_strength'],use_samhq=card['use_samhq'])
                    proposal=masks[0].astype(bool);score=float(scores.max().item())
            else:
                # Autocast is a registered bounded-memory recipe, not the FP32 paper protocol.
                with torch.autocast('cuda',dtype=torch.float16):
                    model.set_image(image)
                    kwargs=dict(box=np.array(t['target_box_xyxy'],dtype=np.float32),multimask_output=False)
                    if card['prompt']=='box_mask':
                        hh,ww=model.input_size
                        padded=np.zeros((1024,1024),np.float32)
                        padded[:hh,:ww]=cv2.resize(target.astype(np.float32),(ww,hh),interpolation=cv2.INTER_LINEAR)
                        coarse=cv2.resize(padded,(256,256),interpolation=cv2.INTER_LINEAR)
                        kwargs['mask_input']=(coarse*8-4)[None]
                    masks,scores,_=model.predict(**kwargs)
                    proposal=masks[0].astype(bool);score=float(scores[0])
                model.reset_image()
        if proposal.shape!=base.shape:raise ValueError('Output changed coordinate frame')
        edited=(base&~target)|proposal
        folder=args.output/t['case_id'];folder.mkdir()
        Image.fromarray(proposal.astype(np.uint8)*255).save(folder/'proposal.png')
        Image.fromarray(edited.astype(np.uint8)*255).save(folder/'mask.png')
        rec=dict(case_id=t['case_id'],tool_id=args.tool,seconds=time.monotonic()-case_start,
            peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,proposal_pixels=int(proposal.sum()),
            mask_pixels=int(edited.sum()),model_score=score,gt_read=False,input_sha256={k:sha(t[k]) for k in ['image','baseline','target']},
            mask_sha256=sha(folder/'mask.png'),native_size=t['native_size'])
        if engine=='ritm':
            rec.update(points=points,prompt_source='deepest interior pixel of predicted coarse mask; no error-localizing evidence')
        (folder/'record.json').write_text(json.dumps(rec,indent=2));records.append(rec)
        (args.output/'progress.json').write_text(json.dumps(dict(completed=len(records),total=len(cases),last=t['case_id'])))
        print(json.dumps(rec),flush=True)
    result=dict(tool_id=args.tool,cases=len(records),records=records,seconds=time.monotonic()-start,loading_seconds=loaded,
                checkpoint_sha256=card['checkpoint_sha256'],torch=torch.__version__,cuda=torch.version.cuda,
                cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],gt_read=False,status='complete')
    if engine in ('sam','sam_hq','sam_refiner'):
        result.update(attention_equivalence_check=attention_check,query_chunk=256,patched_global_blocks=patched_blocks)
    if engine=='sam_refiner':
        result.update(author_commit=card['author_commit'],distance_impl='FastGeodis.generalised_geodesic2d',
            parameters={k:card[k] for k in ['iterations','box_margin','gamma','mask_strength','use_samhq']},
            precision='float16 autocast',author_source_dir=card['source_dir'])
    if engine=='ritm':
        result.update(model_config=ritm_config,precision='float32',max_size=card['max_size'],
            with_flip=True,zoom_in=False,threshold=.5,strict_checkpoint_load=True,
            prompt_policy='single positive deepest coarse-mask point; no GT',source_receipt=card['source_receipt'])
    (args.output/'completed.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
