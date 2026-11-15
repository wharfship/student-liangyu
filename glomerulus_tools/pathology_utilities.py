"""Image-only research utilities: nuclei instances, pathology QC and feature vectors.

GrandQC attribution: Weng et al., Nature Communications (2024),
doi:10.1038/s41467-024-54769-y; original code/weights are noncommercial,
https://github.com/cpath-ukk/grandqc/blob/main/LICENSE .
These outputs are not glomerulus diagnoses or automatic mask-edit permissions.
"""
import argparse,json,math,re,sys,time,os
from pathlib import Path
from .catalog import get_card,sha


def validate_batch(task,card):
 if not isinstance(task,dict) or set(task)!={'schema_version','samples'} or task['schema_version']!=1:
  raise ValueError('Require schema_version=1 and samples only; GT/offline fields forbidden')
 if not isinstance(task['samples'],list) or not task['samples']:raise ValueError('Empty sample batch')
 seen=set()
 for s in task['samples']:
  if not isinstance(s,dict) or set(s)!={'sample_id','image','image_sha256','source_mpp','scale_evidence','stain'}:
   raise ValueError('Only explicit image-only sample fields accepted')
  sid=s['sample_id']
  if not isinstance(sid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',sid) or sid in seen:raise ValueError('Invalid/duplicate ID')
  seen.add(sid)
  if s['stain'] not in card['stains']:raise ValueError('Unsupported stain')
  mpp=s['source_mpp']
  if not isinstance(mpp,(float,int)) or isinstance(mpp,bool) or not math.isfinite(mpp) or mpp<=0:raise ValueError('Require measured positive MPP')
  if not isinstance(s['scale_evidence'],str) or not s['scale_evidence'].strip():raise ValueError('Require scale evidence')
  if not isinstance(s['image'],str) or not Path(s['image']).is_absolute():raise ValueError('Require absolute image path')
  if not isinstance(s['image_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',s['image_sha256']):raise ValueError('Require image SHA256')
 return task


def dump(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def grandqc_models(card,torch):
 import segmentation_models_pytorch as smp
 tissue=smp.UnetPlusPlus(encoder_name='timm-efficientnet-b0',encoder_weights=None,classes=2,activation=None)
 tissue.load_state_dict(torch.load(card['tissue_checkpoint'],map_location='cpu',weights_only=True),strict=True)
 tissue.eval().cuda()
 # Author distributes a complete model pickle, loaded only after pinned hash check.
 artifact=None
 if card['utility_kind']=='grandqc_artifact':
  artifact=torch.load(card['checkpoint'],map_location='cpu',weights_only=False);artifact.eval().cuda()
 return tissue,artifact,smp.encoders.get_preprocessing_fn('timm-efficientnet-b0','imagenet')


def tiled_predict(rgb,model,preprocess,torch,fill):
 import numpy as np
 h,w=rgb.shape[:2];out=np.full((h,w),fill,np.uint8)
 for y in range(0,h,512):
  for x in range(0,w,512):
   tile=np.full((512,512,3),255,np.uint8);part=rgb[y:y+512,x:x+512];hh,ww=part.shape[:2];tile[:hh,:ww]=part
   tensor=torch.from_numpy(preprocess(tile).transpose(2,0,1).astype('float32')).cuda()[None]
   pred=model(tensor).argmax(1)[0].cpu().numpy().astype('uint8');out[y:y+hh,x:x+ww]=pred[:hh,:ww]
 return out


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tool',required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 card=get_card(a.tool);task=json.loads(a.task.read_text(encoding='utf-8-sig'));validate_batch(task,card)
 if not card['enabled'] or card.get('adapter')!='pathology_utility':raise ValueError('Utility not enabled')
 if a.output.exists():raise ValueError('Output directory must be new; preserve previous run')
 if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Set CUDA_VISIBLE_DEVICES after live GPU check')
 for key in ['checkpoint','tissue_checkpoint']:
  if card.get(key):
   assert sha(card[key])==card[key+'_sha256'],'Checkpoint hash mismatch: '+key
 import numpy as np,torch,cv2
 from PIL import Image
 if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(20261115)
 a.output.mkdir(parents=True,exist_ok=False);dump(a.output/'task_NO_GT.json',task);start=time.monotonic()
 kind=card['utility_kind'];model=None
 if kind=='instanseg':
  sys.path.insert(0,card['source_dir']);from instanseg import InstanSeg
  compiled=torch.jit.load(card['checkpoint'],map_location='cuda').eval()
  model=InstanSeg(compiled,device='cuda',image_reader='skimage.io',verbosity=0,channels_last=False)
 elif kind=='dinov2':
  sys.path.insert(0,card['source_dir']);from dinov2.models.vision_transformer import vit_small
  model=vit_small(patch_size=14,img_size=518,init_values=1.0,block_chunks=0)
  model.load_state_dict(torch.load(card['checkpoint'],map_location='cpu',weights_only=True),strict=True);model=model.eval().cuda()
  from torchvision import transforms as T
  transform=T.Compose([T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
 elif kind.startswith('grandqc'):tissue,artifact,preprocess=grandqc_models(card,torch)
 else:raise ValueError('Unknown utility kind')
 records=[]
 with torch.inference_mode():
  for s in task['samples']:
   begin=time.monotonic();assert sha(s['image'])==s['image_sha256'],'Input SHA mismatch'
   with Image.open(s['image']) as im:rgb=np.array(im.convert('RGB'))
   h,w=rgb.shape[:2]
   if max(h,w)>8192:raise ValueError('Patch protocol max side8192; use bounded WSI crops')
   dest=a.output/s['sample_id'];dest.mkdir();torch.cuda.reset_peak_memory_stats();rec={}
   if kind=='instanseg':
    output=model.eval_small_image(rgb,pixel_size=s['source_mpp'],return_image_tensor=False,rescale_output=True)
    labels=output.squeeze().cpu().numpy().astype('uint32');assert labels.shape==(h,w)
    Image.fromarray(labels).save(dest/'instances.tif')
    ids,areas=np.unique(labels[labels>0],return_counts=True)
    rec.update(instances=len(ids),foreground_fraction=float((labels>0).mean()),median_nuclear_area_um2=float(np.median(areas)*s['source_mpp']**2) if len(ids) else None,output_mpp=s['source_mpp'],model_mpp=float(compiled.pixel_size))
    edge=(labels>0)&(cv2.erode((labels>0).astype('uint8'),np.ones((3,3),'uint8'))==0);overlay=rgb.copy();overlay[edge]=[0,240,100]
    Image.fromarray(overlay).save(dest/'overlay.jpg',quality=93)
   elif kind=='dinov2':
    feat=model(transform(Image.fromarray(rgb))[None].cuda()).float().cpu().numpy()[0]
    assert feat.shape==(384,) and np.isfinite(feat).all();norm=float(np.linalg.norm(feat));assert norm>0
    np.save(dest/'embedding.npy',feat);np.save(dest/'embedding_l2.npy',feat/norm)
    rec.update(feature_dimension=384,feature_norm=norm,classifier_head=False,output_semantics='DINOv2 CLS embedding, not glomerulus identity or sclerosis probability')
   else:
    # Tissue author chain: RGB JPEG80 encode then OpenCV BGR decode (preserved author order).
    tw,th=max(1,round(w*s['source_mpp']/10)),max(1,round(h*s['source_mpp']/10))
    low=np.asarray(Image.fromarray(rgb).resize((tw,th),Image.Resampling.LANCZOS));ok,buf=cv2.imencode('.jpg',low,[int(cv2.IMWRITE_JPEG_QUALITY),80]);assert ok
    low=cv2.imdecode(buf,1);td=tiled_predict(low,tissue,preprocess,torch,1)
    Image.fromarray(td).save(dest/'tissue_classes.png');rec.update(tissue_mpp=10.,tissue_fraction=float((td==0).mean()),tissue_labels={'0':'tissue','1':'background'})
    if artifact is not None:
     aw,ah=max(1,round(w*s['source_mpp']/1.5)),max(1,round(h*s['source_mpp']/1.5))
     ar=np.asarray(Image.fromarray(rgb).resize((aw,ah),Image.Resampling.LANCZOS));classes=tiled_predict(ar,artifact,preprocess,torch,7)
     gate=cv2.resize(td,(aw,ah),interpolation=cv2.INTER_NEAREST);classes[gate==1]=7
     Image.fromarray(classes).save(dest/'artifact_classes.png')
     fractions={str(i):float((classes==i).mean()) for i in range(8)}
     rec.update(artifact_mpp=1.5,class_fractions=fractions,artifact_fraction=float(np.isin(classes,[2,3,4,5,6]).mean()),unmapped_class0_fraction=fractions['0'],
       class_map={'1':'normal_tissue','2':'fold','3':'darkspot_foreign','4':'pen','5':'bubble_edge','6':'out_of_focus','7':'background'},
       wrapper_difference='bounded ROI; round physical-size grid; nearest tissue gate; no author WSI offsets or remainder clipping')
     colors=np.array([[255,0,255],[255,255,255],[255,140,0],[128,0,128],[0,0,255],[255,255,0],[255,0,0],[0,0,0]],np.uint8)
     overlay=np.clip(ar*.65+colors[classes]*.35,0,255).astype('uint8');Image.fromarray(overlay).save(dest/'overlay.jpg',quality=93)
    else:
     overlay=rgb.copy();mask=cv2.resize(td,(w,h),interpolation=cv2.INTER_NEAREST)==0;overlay[mask]=(overlay[mask]*.65+np.array([0,220,100])*.35).astype('uint8');Image.fromarray(overlay).save(dest/'overlay.jpg',quality=93)
   torch.cuda.synchronize();rec.update(sample_id=s['sample_id'],tool_id=card['id'],input_sha256=s['image_sha256'],native_size=[w,h],source_mpp=s['source_mpp'],stain=s['stain'],
    seconds=time.monotonic()-begin,peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha256=card['checkpoint_sha256'],GT_read=False,device='cuda',adapter_sha256=sha(__file__))
   dump(dest/'record.json',rec);records.append(rec);dump(a.output/'progress.json',dict(done=len(records),total=len(task['samples'])))
   print(s['sample_id'],rec['seconds'],flush=True)
 dump(a.output/'completed.json',dict(status='complete',samples=len(records),tool_id=card['id'],seconds=time.monotonic()-start,GT_read=False,torch=torch.__version__,cuda=torch.version.cuda,records=records))


if __name__=='__main__':main()
