"""Six NePathTK renal networks: RGB ROI -> whole-glomerulus semantic mask."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time

ROOT=Path(__file__).resolve().parent
for folder in reversed(os.environ.get('NEPATH_EXTRA_PATHS','').split(os.pathsep)):
    if folder:sys.path.insert(0,folder)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def load(model_id,device='cuda:0'):
    import torch
    cards=json.loads((ROOT/'models.json').read_text(encoding='utf8'))['models']
    card=next(x for x in cards if x['id']==model_id)
    ckpt=ROOT/card['checkpoint']
    if sha(ckpt)!=card['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    if card['backend']=='smp':
        import segmentation_models_pytorch as smp
        from torch import nn
        class NePathModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model=smp.create_model(card['architecture'],encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=4)
                p=smp.encoders.get_preprocessing_params('resnet34')
                self.register_buffer('std',torch.tensor(p['std']).view(1,3,1,1))
                self.register_buffer('mean',torch.tensor(p['mean']).view(1,3,1,1))
            def forward(self,x):return self.model((x-self.mean)/self.std)
        model=NePathModel()
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
    else:
        from mmengine.config import Config
        from mmseg.registry import MODELS
        from mmseg.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        cfg=Config.fromfile(str(ROOT/card['config']))
        cfg.model.backbone.init_cfg=None
        model=MODELS.build(cfg.model)
        state=torch.load(ckpt,map_location='cpu')['state_dict']
    result=model.load_state_dict(state,strict=True)
    if result.missing_keys or result.unexpected_keys:raise ValueError(str(result))
    return model.to(device).eval(),card

def probabilities(model,card,rgb):
    import numpy as np
    import torch
    device=next(model.parameters()).device
    x=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()[None].to(device)
    if card['backend']=='smp':
        # Exact author checkpoint graph: raw RGB0..255; mean/std applied inside model.
        logits=model(x)
    else:
        # PIL provides RGB already. mmseg's original bgr_to_rgb converts file-reader BGR.
        pre=card['preprocessing']
        mean=x.new_tensor(pre['mean'])[None,:,None,None]
        std=x.new_tensor(pre['std'])[None,:,None,None]
        x=(x-mean)/std
        h,w=rgb.shape[:2]
        meta=dict(ori_shape=(h,w),img_shape=(h,w),pad_shape=(h,w),
                  padding_size=[0,0,0,0],flip=False)
        logits=model.inference(x,[meta])
    if logits.shape[1]!=4 or not torch.isfinite(logits).all():raise ValueError('Invalid 4-class logits')
    return logits.softmax(1)[0].cpu().numpy()

def starts(length,size,stride):
    if length<=size:return [0]
    return sorted(set(range(0,length-size+1,stride))|{length-size})

def segment(model,card,rgb,source_mpp,target_mpp=1.0):
    import cv2
    import numpy as np
    import torch
    h,w=rgb.shape[:2];factor=source_mpp/target_mpp
    ww,wh=max(1,round(w*factor)),max(1,round(h*factor))
    if ww*wh>100_000_000:raise ValueError('ROI working grid exceeds 100M pixels; use WSI tiling adapter')
    working=cv2.resize(rgb,(ww,wh),interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    size,stride=card['tile_pixels'],card['stride_pixels']
    ys,xs=starts(wh,size,stride),starts(ww,size,stride)
    sums=np.zeros((4,wh,ww),np.float32);counts=np.zeros((wh,ww),np.uint16)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop=working[y:y+size,x:x+size];ch,cw=crop.shape[:2]
                tile=np.full((size,size,3),255,np.uint8);tile[:ch,:cw]=crop
                pred=probabilities(model,card,tile)
                sums[:,y:y+ch,x:x+cw]+=pred[:,:ch,:cw];counts[y:y+ch,x:x+cw]+=1
    if counts.min()<1:raise ValueError('Incomplete tile coverage')
    labels=(sums/counts[None]).argmax(0).astype(np.uint8)
    labels=cv2.resize(labels,(w,h),interpolation=cv2.INTER_NEAREST)
    mask=(labels==1).astype(np.uint8)*255
    return mask,labels,dict(native_size=[w,h],working_size=[ww,wh],windows=len(ys)*len(xs),
                           coverage_min=int(counts.min()),coverage_max=int(counts.max()))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['unet','fpn','deeplabv3plus','segformer_b5','mask2former_swint','segnext_mscant'])
    p.add_argument('--image',type=Path)
    p.add_argument('--source-mpp',type=float,help='Explicit source working/measured micrometres per pixel')
    p.add_argument('--scale-basis',choices=['working_assumption','measured_metadata'])
    p.add_argument('--target-mpp',type=float,default=1.0,help='Fixed d4 adapter working grid; default 1.0, not measured training MPP')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true',help='CPU architecture/checkpoint verification only')
    p.add_argument('--preflight',action='store_true',help='CUDA synthetic tile inference only')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory must be new or empty')
    if not a.check_only and not a.preflight:
        if not a.image or not a.source_mpp or a.source_mpp<=0 or not a.scale_basis:p.error('Need --image, positive --source-mpp, and --scale-basis')
        if a.image.suffix.lower() in ['.svs','.ndpi','.mrxs','.scn']:p.error('This interface accepts RGB ROI; WSI requires external MPP-aware tiling')
        if a.target_mpp!=1.0:p.error('This tool protocol fixes target working MPP at 1.0; use a new protocol for other grids')
    import torch
    import numpy as np
    from PIL import Image
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261105);np.random.seed(20261105)
    if not a.check_only and not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU image-inference fallback')
    model,card=load(a.model,'cpu' if a.check_only else 'cuda:0')
    a.output.mkdir(parents=True,exist_ok=True)
    record=dict(model=a.model,checkpoint_sha256=card['sha256'],strict_load=True,GT_read=False,
                code_sha256=sha(__file__),torch=torch.__version__,torch_cuda=torch.version.cuda,
                physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),training_sources=card['training_sources'],
                preprocessing=card['preprocessing'],input_kind='RGB_ROI')
    if a.check_only:record.update(status='STRICT_LOAD_PASSED',image_inference=False)
    elif a.preflight:
        with torch.inference_mode():pred=probabilities(model,card,np.zeros((1024,1024,3),np.uint8))
        record.update(status='CUDA_SYNTHETIC_PREFLIGHT_PASSED',image_inference=False,synthetic_inference=True,
                      output_shape=list(pred.shape),probability_sum_max_error=float(np.max(np.abs(pred.sum(0)-1))))
    else:
        with Image.open(a.image) as im:
            if im.width*im.height>100_000_000:raise ValueError('Native ROI exceeds 100M pixels; use external WSI adapter')
            rgb=np.asarray(im.convert('RGB'))
        mask,labels,coverage=segment(model,card,rgb,a.source_mpp,a.target_mpp)
        for name,arr in [('mask.png',mask),('labels.png',labels)]:Image.fromarray(arr).save(a.output/name)
        record.update(status='FROZEN_COMPLETE_NOT_SCORED',image_inference=True,image=str(a.image),
                      source_mpp=a.source_mpp,scale_basis=a.scale_basis,target_working_mpp=a.target_mpp,
                      foreground_pixels=int((mask>0).sum()),mask_sha256=sha(a.output/'mask.png'),
                      labels_sha256=sha(a.output/'labels.png'),**coverage)
    record['seconds']=time.time()-start
    (a.output/'receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(record,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
