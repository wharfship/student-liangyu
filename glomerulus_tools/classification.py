"""Candidate classification protocol. No masks, labels, or evaluation fields enter inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from .catalog import get_card, sha, load_registry

SCHEMA = 'glomerulus_candidate_classification_v1'
POSITIVE_TEXTS = [
    'a histopathology image of a renal glomerulus.',
    'a histopathology image of a sclerotic renal glomerulus.',
    'a histopathology image of a diseased renal glomerulus.',
]
NEGATIVE_TEXTS = [
    'a histopathology image of renal tubules.',
    'a histopathology image of a renal artery or blood vessel.',
    'a histopathology image of renal interstitial tissue.',
    'a histopathology image of inflammatory cells in kidney tissue.',
    'a histopathology image of background or tissue artifact.',
]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def exact_keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value)-set(required)-set(optional) or set(required)-set(value):
        raise ValueError('Invalid classification fields; expected '+str(required)+' optional '+str(optional))


def validate_task(task, card):
    exact_keys(task, ['schema_version','species','target','images','candidates'])
    if task['schema_version'] != 1 or task['target'] != 'candidate_identity':
        raise ValueError('Expected candidate_identity classification schema_version=1')
    if task['species'] not in card['species']:
        raise ValueError('Unsupported species')
    images={}
    if not isinstance(task['images'],list) or not task['images']:
        raise ValueError('Need source images')
    for item in task['images']:
        exact_keys(item,['image_id','image','image_sha256','native_size','stain','source_mpp','scale_basis','scale_evidence'])
        if item['image_id'] in images: raise ValueError('Duplicate image ID')
        if item['stain'] not in card['stains']: raise ValueError('Unsupported stain: '+item['stain'])
        mpp=item['source_mpp']
        if isinstance(mpp,bool) or not isinstance(mpp,(float,int)) or not math.isfinite(mpp) or mpp<=0:
            raise ValueError('Source MPP must be supplied with its evidence; never infer from magnification')
        if not item['scale_basis'] or not item['scale_evidence']: raise ValueError('Need scale evidence')
        size=item['native_size']
        if len(size)!=2 or any(type(v)!=int or v<=0 for v in size): raise ValueError('Invalid native_size')
        if len(item['image_sha256'])!=64: raise ValueError('Need source SHA256')
        images[item['image_id']]=item
    ids=set()
    if not isinstance(task['candidates'],list) or not task['candidates']: raise ValueError('Need predicted candidates')
    for c in task['candidates']:
        exact_keys(c,['candidate_id','image_id','box'])
        if c['candidate_id'] in ids or c['image_id'] not in images: raise ValueError('Duplicate or unlinked candidate')
        ids.add(c['candidate_id'])
        b=c['box'];w,h=images[c['image_id']]['native_size']
        if len(b)!=4 or any(type(v)!=int for v in b) or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):
            raise ValueError('Invalid half-open level0 pixel box')
    return images


def make_crop(source, box, model_kind):
    from PIL import Image
    w,h=source.size
    x1,y1,x2,y2=box
    crop_box=[max(0,x1-50),max(0,y1-50),min(w,x2+50),min(h,y2+50)]
    crop=source.crop(crop_box).convert('RGB')
    native=crop.size
    padding=[0,0,0,0]
    if model_kind=='glo5':
        # Match author generate_patches.py's initial 256-square resampling.
        crop=crop.resize((256,256),Image.Resampling.BICUBIC)
    else:
        # Preserve the complete candidate when the model's center-crop runs.
        side=max(crop.size)
        dx=(side-crop.width)//2;dy=(side-crop.height)//2
        padding=[dx,dy,side-crop.width-dx,side-crop.height-dy]
        square=Image.new('RGB',(side,side),(255,255,255))
        square.paste(crop,(dx,dy));crop=square
    record=dict(crop_box=crop_box,native_crop_size=list(native),white_padding=padding,
                model_input_pixels=[224,224],crop_pixel_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                context_clipped=any([x1<50,y1<50,x2+50>w,y2+50>h]),
                source_pixels_per_model_pixel=[native[0]/224,native[1]/224] if model_kind=='glo5' else [crop.width/224,crop.height/224])
    return crop,record


class NativeSVSSource:
    """Decode only level-0 JPEG tiles needed by the candidate crop."""
    def __init__(self,path):
        import tifffile
        self.file=tifffile.TiffFile(path)
        self.page=self.file.pages[0]
        p=self.page
        if not (p.is_tiled and int(p.compression)==7 and p.samplesperpixel==3):
            self.file.close()
            raise ValueError('SVS reader requires RGB JPEG tiled level0')
        self.size=(p.imagewidth,p.imagelength)

    def crop(self,box):
        import io
        import numpy as np
        from PIL import Image
        x,y,xx,yy=box;p=self.page;handle=self.file.filehandle
        th,tw=p.tilelength,p.tilewidth;nx=(p.imagewidth+tw-1)//tw
        out=np.full((yy-y,xx-x,3),255,np.uint8)
        for ty in range(y//th,(yy-1)//th+1):
            for tx in range(x//tw,(xx-1)//tw+1):
                i=ty*nx+tx;handle.seek(p.dataoffsets[i]);data=handle.read(p.databytecounts[i])
                tables=p.jpegtables
                tile=np.asarray(Image.open(io.BytesIO(tables[:-2]+data[2:] if tables else data)).convert('RGB'))
                l,t,r,b=max(x,tx*tw),max(y,ty*th),min(xx,(tx+1)*tw),min(yy,(ty+1)*th)
                out[t-y:b-y,l-x:r-x]=tile[t-ty*th:b-ty*th,l-tx*tw:r-tx*tw,:3]
        return Image.fromarray(out)

    def close(self):self.file.close()


def load_model(card):
    import torch
    from torchvision import transforms
    kind=card['model_kind'];folder=Path(card['model_dir'])
    if kind=='glo5':
        import importlib.util
        path=Path(card['source_dir'])/'models/resnetv2.py'
        spec=importlib.util.spec_from_file_location('author_glo_resnetv2',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        model=module.KNOWN_MODELS['BiT-M-R101x1'](head_size=5,zero_head=True)
        weights=torch.load(card['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(weights['state_dict'],strict=True)
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        return model.cuda().eval(),transform,None,dict(checkpoint_keys=list(weights),strict_load=True)
    if kind=='plip':
        from transformers import CLIPModel,CLIPProcessor
        model=CLIPModel.from_pretrained(folder,local_files_only=True).cuda().eval()
        processor=CLIPProcessor.from_pretrained(folder,local_files_only=True)
        tokens=processor(text=POSITIVE_TEXTS+NEGATIVE_TEXTS,return_tensors='pt',padding=True)
        with torch.inference_mode():
            features=model.get_text_features(**{k:v.cuda() for k,v in tokens.items()})
            features=torch.nn.functional.normalize(features,dim=-1)
        transform=lambda im:processor(images=im,return_tensors='pt')['pixel_values'][0]
        return model,transform,features,dict(processor=str(processor),strict_load=True)
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS
    cfg=json.loads((folder/'open_clip_config.json').read_text())
    name='glomerulus_'+kind
    if kind=='biomedclip':
        cfg['model_cfg']['text_cfg']['hf_model_name']=str(folder)
        cfg['model_cfg']['text_cfg']['hf_tokenizer_name']=str(folder)
    _MODEL_CONFIGS[name]=cfg['model_cfg']
    model,_,transform=open_clip.create_model_and_transforms(name,pretrained=card['checkpoint'],
            pretrained_hf=False,image_mean=cfg['preprocess_cfg']['mean'],image_std=cfg['preprocess_cfg']['std'])
    model=model.cuda().eval()
    if kind=='biomedclip':
        from open_clip.tokenizer import HFTokenizer
        tokenizer=HFTokenizer(str(folder),context_length=256)
    else:
        tokenizer=open_clip.get_tokenizer('ViT-B-32')
    with torch.inference_mode():
        features=model.encode_text(tokenizer(POSITIVE_TEXTS+NEGATIVE_TEXTS).cuda(),normalize=True)
    return model,transform,features,dict(transform=str(transform),strict_load=True)


def run(task,card,output):
    import torch
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS=None
    images=validate_task(task,card)
    if not card['enabled']:raise ValueError('Tool disabled')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():
        raise ValueError('Need explicitly selected available CUDA GPU; no automatic CPU fallback')
    dest=Path(output);dest.mkdir(parents=True,exist_ok=False)
    dump(dest/'online_task.json',task)
    started=time.perf_counter()
    if sha(card['checkpoint'])!=card['checkpoint_sha256']:raise ValueError('Checkpoint SHA mismatch')
    torch.manual_seed(20261108);torch.set_num_threads(4)
    model,transform,text_features,model_info=load_model(card)
    load_seconds=time.perf_counter()-started
    dump(dest/'input_protocol.json',dict(schema=SCHEMA,species=task['species'],stains=card['stains'],
        source_scale='per-source task MPP and evidence; never inferred',model_mpp='required physical sampling unknown; actual sampling is candidate-dependent',
        author_training_scale=card.get('author_training_scale','unknown'),
        fixed_model_physical_fov='unknown; candidate-dependent crop',
        crop='predicted half-open box plus50 native pixels clipped to image',
        preprocessing='author256-square then224 and ImageNet normalization' if card['model_kind']=='glo5' else 'white square pad then model-provided224 transform',
        model_info=model_info,positive_texts=POSITIVE_TEXTS if text_features is not None else None,
        negative_texts=NEGATIVE_TEXTS if text_features is not None else None,
        research_stain_override=card.get('research_stain_override'),
        score='1-softmax(class4)' if text_features is None else 'softmax of scaled maximum positive vs maximum negative cosine; uncalibrated',
        thresholds=dict(accept=.8,reject=.2),gt_read=False))
    results=[];forwards=0.;cached_image=None;source=None
    with (dest/'classifications.jsonl').open('w',encoding='utf-8') as stream:
        for offset in range(0,len(task['candidates']),16):
            group=task['candidates'][offset:offset+16]
            tensors=[];receipts=[]
            for c in group:
                info=images[c['image_id']]
                if cached_image!=c['image_id']:
                    if sha(info['image'])!=info['image_sha256']:raise ValueError('Source SHA mismatch')
                    if source is not None:source.close()
                    source=NativeSVSSource(info['image']) if Path(info['image']).suffix.lower()=='.svs' else Image.open(info['image']).convert('RGB')
                    if list(source.size)!=info['native_size']:raise ValueError('Source dimensions mismatch')
                    cached_image=c['image_id']
                crop,receipt=make_crop(source,c['box'],card['model_kind'])
                receipt['physical_crop_um']=[v*info['source_mpp'] for v in receipt['native_crop_size']]
                receipt['source_mpp']=info['source_mpp'];receipt['scale_basis']=info['scale_basis']
                tensors.append(transform(crop));receipts.append(receipt)
            batch=torch.stack(tensors).cuda()
            torch.cuda.synchronize();t0=time.perf_counter()
            with torch.inference_mode():
                if card['model_kind']=='glo5':
                    logits=model(batch);prob=logits.softmax(-1);scores=1-prob[:,4]
                    raw=prob.cpu().numpy();similarities=None
                else:
                    feat=model.get_image_features(pixel_values=batch) if card['model_kind']=='plip' else model.encode_image(batch)
                    feat=torch.nn.functional.normalize(feat,dim=-1)
                    similarities=feat@text_features.T
                    identity_logits=torch.stack([similarities[:,:3].max(-1).values,similarities[:,3:].max(-1).values],-1)*model.logit_scale.exp()
                    prob=identity_logits.softmax(-1);scores=prob[:,0]
                    raw=prob.cpu().numpy();similarities=similarities.cpu().numpy()
            torch.cuda.synchronize();seconds=time.perf_counter()-t0;forwards+=seconds
            scores=scores.cpu().tolist()
            for j,(c,s,r) in enumerate(zip(group,scores,receipts)):
                if not math.isfinite(s) or not np.isfinite(raw[j]).all():raise ValueError('Nonfinite score')
                record=dict(schema=SCHEMA,tool_id=card['id'],candidate_id=c['candidate_id'],image_id=c['image_id'],
                    box=c['box'],identity_score=s,decision='glomerulus' if s>=.8 else 'non_glomerulus' if s<=.2 else 'uncertain',
                    class_scores=raw[j].tolist(),classes=['normal','obsolescent','solidified','disappearing','non_glomerulus'] if card['model_kind']=='glo5' else ['glomerulus','non_glomerulus'],
                    argmax_class=int(np.argmax(raw[j])),similarities=None if similarities is None else similarities[j].tolist(),
                    score_calibrated=False,boundary_assessed=False,edit_authorized=False,accuracy_verified=False,
                    crop=r,forward_seconds_amortized=seconds/len(group))
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();results.append(record)
            print(json.dumps(dict(completed=len(results),total=len(task['candidates']),forward_seconds=forwards)),flush=True)
    if source is not None:source.close()
    record=dict(schema=SCHEMA,tool_id=card['id'],valid=len(results)==len(task['candidates']),
        accuracy_verified=False,needs_review=True,gt_read=False,boundary_assessed=False,edit_authorized=False,
        count=len(results),elapsed_seconds=time.perf_counter()-started,load_seconds=load_seconds,forward_seconds=forwards,
        device=torch.cuda.get_device_name(0),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        torch_version=torch.__version__,torch_cuda=torch.version.cuda,python=sys.version,host=platform.node(),
        checkpoint_sha256=sha(card['checkpoint']),catalog_sha256=load_registry()['catalog_sha256'],
        code_sha256=sha(__file__),task_sha256=sha(dest/'online_task.json'),output_sha256=sha(dest/'classifications.jsonl'),
        api_calls=0,api_fee=0,currency='CNY',hardware_cost='not metered')
    dump(dest/'inference_record.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--tool',required=True);parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--research-stain-override',choices=['HE'])
    args=parser.parse_args()
    card=get_card(args.tool)
    if args.research_stain_override:
        if card['id']!='glo5_pas_identity':parser.error('Override only supports Glo5 research classification')
        card=dict(card,stains=card['stains']+[args.research_stain_override],research_stain_override=args.research_stain_override)
    print(json.dumps(run(json.loads(args.task.read_text(encoding='utf-8-sig')),card,args.output),ensure_ascii=False))
