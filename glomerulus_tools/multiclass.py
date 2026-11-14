"""Research export of registered NePath semantic classes; image-only inference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zlib

from .catalog import get_card, sha, asset_status
from .contract import prepare_task, compatibility


def dump(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def validate_batch(batch,card):
    if set(batch)!={'schema','cases'} or batch['schema']!='nepath_multiclass_batch_v1':
        raise ValueError('Expected nepath_multiclass_batch_v1, image-only batch')
    if not card.get('multiclass_export') or card['adapter']!='nepath':
        raise ValueError('Registered card does not expose semantic classes')
    if not 1<=len(batch['cases'])<=64:raise ValueError('Invalid bounded batch size')
    ids=[]
    for case in batch['cases']:
        if set(case)!={'task','rgb_cache'}:raise ValueError('Unknown case fields')
        t=prepare_task(case['task'])
        errors=compatibility(card,t,research=True)
        if errors:raise ValueError(str(errors))
        if t['input_form']!='wsi':raise ValueError('This export requires WSI tasks')
        if Path(t['task_id']).name!=t['task_id'] or '/' in t['task_id'] or '\\' in t['task_id']:
            raise ValueError('Unsafe task ID')
        cache=Path(case['rgb_cache'])
        info=json.loads(cache.with_name('completed.json').read_text(encoding='utf-8'))
        if not (info['task_id']==t['task_id'] and info['native_size']==t['native_size'] and
                abs(info['source_mpp']-t['source_mpp'])<1e-6 and
                info['target_working_mpp']==card['target_mpp'] and info['GT_read'] is False):
            raise ValueError('RGB cache identity/scale does not match input WSI')
        ids.append(t['task_id'])
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate task IDs')


def export_native(labels,native_size,path,class_id=None):
    """Stream nearest-neighbor native export with exact native class pixel counts."""
    import numpy as np
    import tifffile
    width,height=native_size
    wh,ww=labels.shape
    xs=np.minimum(np.arange(width,dtype=np.int64)*ww//width,ww-1)
    counts=np.zeros(4,np.int64)
    def strips():
        for y in range(0,height,128):
            ys=np.minimum(np.arange(y,min(y+128,height),dtype=np.int64)*wh//height,wh-1)
            block=np.ascontiguousarray(labels[np.ix_(ys,xs)])
            counts[:]+=np.bincount(block.ravel(),minlength=4)
            if class_id is not None:block=(block==class_id).astype(np.uint8)*255
            yield zlib.compress(block.tobytes(),1)
    tifffile.imwrite(path,data=strips(),shape=(height,width),dtype=np.uint8,photometric='minisblack',
                      bigtiff=True,metadata=None,rowsperstrip=128,compression='deflate',maxworkers=1)
    return counts.tolist()


def infer_case(network,model,model_card,card,case,dest,batch_sha):
    import numpy as np
    import torch
    tick=time.monotonic()
    task=prepare_task(case['task']);cache=Path(case['rgb_cache'])
    meta=json.loads(cache.with_name('completed.json').read_text(encoding='utf-8'))
    if sha(cache)!=meta['rgb_sha256']:raise ValueError('RGB cache SHA256 mismatch')
    rgb=np.load(cache,mmap_mode='r');h,w=rgb.shape[:2]
    expected=[max(1,round(n*task['source_mpp']/card['target_mpp'])) for n in task['native_size']]
    if [w,h]!=expected or rgb.dtype!=np.uint8:raise ValueError('Cache grid or dtype mismatch')
    size,stride=card['patch'],card['stride']
    ys,xs=network.starts(h,size,stride),network.starts(w,size,stride)
    if w*h>1_000_000_000 or len(ys)*len(xs)>20000:raise ValueError('WSI budget exceeded')
    sums=np.lib.format.open_memmap(dest/'probability_sums.npy',mode='w+',dtype=np.float32,shape=(4,h,w));sums[:]=0
    counts=np.zeros((h,w),np.uint8);windows=0
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop=rgb[y:y+size,x:x+size];ch,cw=crop.shape[:2]
                tile=np.full((size,size,3),255,np.uint8);tile[:ch,:cw]=crop
                pred=network.probabilities(model,model_card,tile)
                sums[:,y:y+ch,x:x+cw]+=pred[:,:ch,:cw]
                counts[y:y+ch,x:x+cw]+=1;windows+=1
            dump(dest/'progress.json',dict(windows=windows,total_windows=len(ys)*len(xs),seconds=time.monotonic()-tick))
    if int(counts.min())<1:raise ValueError('Incomplete image coverage')
    labels=np.lib.format.open_memmap(dest/'working_labels.npy',mode='w+',dtype=np.uint8,shape=(h,w))
    for y in range(0,h,256):
        labels[y:y+256]=np.argmax(sums[:,y:y+256]/counts[None,y:y+256],axis=0).astype(np.uint8)
    labels.flush()
    native_counts=export_native(labels,task['native_size'],dest/'labels.tif')
    for class_id,name in [(2,'artery'),(3,'arteriole')]:
        other=export_native(labels,task['native_size'],dest/(name+'_mask.tif'),class_id)
        if other!=native_counts:raise ValueError('Class export inconsistency')
    result=dict(schema='nepath_semantic_classes_v1',status='completed',valid=True,task_id=task['task_id'],
                tool_id=card['id'],class_names=model_card['class_names'],native_size=task['native_size'],
                working_size=[w,h],source_mpp=task['source_mpp'],target_working_mpp=card['target_mpp'],
                class_native_pixels=native_counts,windows=windows,coverage_min=int(counts.min()),coverage_max=int(counts.max()),
                class_area_um2=[n*task['source_mpp']**2 for n in native_counts],
                cache_sha256=meta['rgb_sha256'],checkpoint_sha256=card['checkpoint_sha256'],
                config_sha256=card['config_sha256'],network_code_sha256=sha(Path(card['source_dir'])/'run.py'),
                inference_code_sha256=sha(__file__),batch_sha256=batch_sha,
                physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],seconds=time.monotonic()-tick,
                GT_read=False,old_predictions_read=False,accuracy_verified=False,needs_review=True,
                outputs={n:dict(path=str(dest/n),sha256=sha(dest/n),bytes=(dest/n).stat().st_size)
                         for n in ['labels.tif','artery_mask.tif','arteriole_mask.tif','working_labels.npy']})
    for a in [rgb,sums,labels]:a._mmap.close()
    (dest/'probability_sums.npy').unlink()
    dump(dest/'inference_record.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--tool',required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    card=get_card(a.tool);batch=json.loads(a.task.read_text(encoding='utf-8-sig'))
    validate_batch(batch,card)
    if not asset_status(card)['available']:raise ValueError('Missing registered assets')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Select a GPU after live resource check')
    import torch
    import numpy as np
    torch.set_num_threads(4);torch.manual_seed(20261105);np.random.seed(20261105)
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    for folder in reversed(card.get('extra_paths',[])):sys.path.insert(0,folder)
    from .adapters import module_file
    network=module_file(Path(card['source_dir'])/'run.py','_nepath_multiclass_network')
    if sha(card['checkpoint'])!=card['checkpoint_sha256'] or sha(card['config'])!=card['config_sha256']:
        raise ValueError('Model identity mismatch')
    model,model_card=network.load(card['model_id'])
    if model_card['class_names']!=card['semantic_class_names']:raise ValueError('Class mapping mismatch')
    if (card['patch'],card['stride'],card['target_mpp'])!=(model_card['tile_pixels'],model_card['stride_pixels'],1.):
        raise ValueError('Registered model grid mismatch')
    a.output.mkdir(parents=True,exist_ok=True)
    plan_sha=sha(a.task)
    dump(a.output/'runtime.json',dict(torch=torch.__version__,torch_cuda=torch.version.cuda,GT_read=False,
                                   class_names=model_card['class_names'],tool_id=card['id'],batch_sha256=plan_sha))
    completed=[]
    for i,case in enumerate(batch['cases'],1):
        dest=a.output/case['task']['task_id'];dest.mkdir(exist_ok=True)
        record=dest/'inference_record.json'
        if record.exists():
            r=json.loads(record.read_text())
            if r['batch_sha256']!=plan_sha or any(sha(dest/n)!=v['sha256'] for n,v in r['outputs'].items()):
                raise ValueError('Prior run identity mismatch')
        else:r=infer_case(network,model,model_card,card,case,dest,plan_sha)
        completed.append(r['task_id'])
        dump(a.output/'progress.json',dict(completed=len(completed),total=len(batch['cases']),last=r['task_id']))
        print('COMPLETE',i,len(batch['cases']),r['task_id'],round(r['seconds'],2),flush=True)
    dump(a.output/'completed.json',dict(status='PASS',completed=completed,GT_read=False,accuracy_verified=False))


if __name__=='__main__':main()
