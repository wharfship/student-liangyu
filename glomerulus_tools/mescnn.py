"""MESCnn task checkpoint on an already prepared author-scale RGB tile."""
from pathlib import Path
import argparse,json,os,sys,time

def load_model(card):
    import torch
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    cfg=get_cfg();cfg.merge_from_file(model_zoo.get_config_file('Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES=1;cfg.TEST.DETECTIONS_PER_IMAGE=1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=.5;cfg.MODEL.WEIGHTS=card['checkpoint']
    cfg.MODEL.DEVICE='cuda'
    predictor=DefaultPredictor(cfg)
    # Detectron2 checkpointer permits missing heads; require strict task matching here.
    state=torch.load(card['checkpoint'],map_location='cpu',weights_only=False)
    predictor.model.load_state_dict(state['model'],strict=True)
    return predictor,cfg

def infer_tile(card,task,output,predictor=None):
    import cv2,numpy as np,torch
    from PIL import Image
    from .catalog import sha
    if sha(card['checkpoint'])!=card['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    # Scale is a required preparation contract; no silent resize from arbitrary ROI/WSI.
    if task['input_form']!='patch':raise ValueError('MESCnn pilot requires a prepared author-scale patch')
    if task.get('scale_basis')!='mescnn_metadata_10x_prepared_tile':
        raise ValueError('Require metadata-derived 10x tile; arbitrary native crop is not accepted')
    torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.cuda.set_per_process_memory_fraction(.35,0)
    if predictor is None:predictor,cfg=load_model(card)
    else:cfg=predictor.cfg
    rgb=np.asarray(Image.open(task['image']).convert('RGB'))
    tick=time.monotonic()
    with torch.inference_mode():instances=predictor(rgb[:,:,::-1].copy())['instances'].to('cpu')
    masks=instances.pred_masks.numpy();scores=instances.scores.numpy()
    labels=np.zeros(rgb.shape[:2],np.uint16)
    # Higher scores retain overlap ownership. Area filtering is recorded separately;
    # raw instance output is preserved so small sclerosis is not silently discarded.
    for idx in np.argsort(scores):labels[masks[idx]]=int(idx)+1
    mask=labels>0;Image.fromarray(mask.astype('uint8')*255).save(output/'mask.png')
    Image.fromarray(labels).save(output/'instances.tif')
    return dict(mask_path=str(output/'mask.png'),instance_path=str(output/'instances.tif'),
      instance_count=len(masks),scores=scores.tolist(),instance_areas_pixels=masks.sum(axis=(1,2)).tolist(),
      score_threshold=.5,area_filter_applied=False,task_checkpoint_strict_load=True,
      config_dump=cfg.dump(),network_inference_seconds=time.monotonic()-tick,
      scale_scope='prepared10x patch; full-slide tiling/NMS/QuPath area filter not reproduced')

def validate_batch(task):
    from .contract import prepare_task
    if set(task)!={'schema','cases'} or task['schema']!='mescnn_prepared_tile_batch_v1':raise ValueError('Invalid prepared-tile batch')
    if not 1<=len(task['cases'])<=128:raise ValueError('Invalid batch size')
    ids=[]
    for case in task['cases']:
        t=prepare_task(case);ids.append(t['task_id'])
        if not t['task_id'].isalnum() or t['scale_basis']!='mescnn_metadata_10x_prepared_tile':raise ValueError('Invalid prepared tile')
        if max(t['native_size'])>2048:raise ValueError('Pilot supports prepared tile up to2048')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate tile ID')

def main():
    from .catalog import get_card,sha
    import torch
    parser=argparse.ArgumentParser();parser.add_argument('--tool',required=True);parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    card=get_card(a.tool);task=json.loads(a.task.read_text(encoding='utf-8-sig'));validate_batch(task)
    if not card['enabled'] or card['adapter']!='mescnn':raise ValueError('Disabled/incorrect tool')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():raise ValueError('Need live-checked CUDA device')
    if sha(card['checkpoint'])!=card['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261114)
    torch.cuda.set_per_process_memory_fraction(.35,0)
    start=time.monotonic();predictor,cfg=load_model(card);loaded=time.monotonic()-start;rows=[]
    for t in task['cases']:
        from .contract import compatibility
        reasons=compatibility(card,t,research=True)
        if reasons:raise ValueError(reasons)
        dest=a.output/t['task_id'];dest.mkdir()
        result=infer_tile(card,t,dest,predictor);result.update(task_id=t['task_id'],gt_read=False,
            input_sha256=sha(t['image']),checkpoint_sha256=card['checkpoint_sha256'])
        (dest/'record.json').write_text(json.dumps(result,indent=2));rows.append(result)
        (a.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(task['cases']))))
        print(t['task_id'],result['instance_count'],flush=True)
    (a.output/'completed.json').write_text(json.dumps(dict(status='complete',cases=len(rows),records=rows,
        GT_read=False,loading_seconds=loaded,seconds=time.monotonic()-start,strict_checkpoint_load=True,
        torch=torch.__version__,torch_cuda=torch.version.cuda,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']),indent=2))

if __name__=='__main__':main()
