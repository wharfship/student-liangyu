"""Build a GT-stratified capability benchmark; all prompts/ROIs use predictions.

This is a development capability study, not a GT-free failure detector evaluation.
Online payloads contain only RGB and the frozen prediction. Original files read-only.
"""
from pathlib import Path
from collections import Counter
import csv, json, hashlib, io, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tifffile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
AUDIT=ROOT/'已经整理好的内容/医院分割结果分析/重点failure失败原因'
DATA=Path('D:/数据集')
Image.MAX_IMAGE_PIXELS=None

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def csvout(p,a):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in a for k in r)));w.writeheader();w.writerows(a)

def roi(path,box):
 x,y,xx,yy=map(int,box)
 with tifffile.TiffFile(path) as f:
  p=f.pages[0];h,w=p.imagelength,p.imagewidth;channels=p.samplesperpixel
  out=np.zeros((yy-y,xx-x,channels),p.dtype)
  if p.is_tiled:th,tw=p.tilelength,p.tilewidth
  else:th,tw=p.rowsperstrip,w
  nx=(w+tw-1)//tw
  ny=(h+th-1)//th
  planes=channels if p.planarconfig==2 else 1
  for plane in range(planes):
   for ty in range(y//th,(yy-1)//th+1):
    for tx in range(x//tw,(xx-1)//tw+1):
     idx=plane*nx*ny+ty*nx+tx;f.filehandle.seek(p.dataoffsets[idx]);data=f.filehandle.read(p.databytecounts[idx])
     if int(p.compression)==7:
      tables=p.jpegtables;tile=np.asarray(Image.open(io.BytesIO(tables[:-2]+data[2:] if tables else data)).convert('RGB'))
     else:
      tile=p.decode(data,idx,jpegtables=p.jpegtables)[0][0]
      if tile.ndim==2:tile=tile[...,None]
     l,t,r,b=max(x,tx*tw),max(y,ty*th),min(xx,(tx+1)*tw),min(yy,(ty+1)*th)
     if planes>1:out[t-y:b-y,l-x:r-x,plane]=tile[t-ty*th:b-ty*th,l-tx*tw:r-tx*tw,0]
     else:out[t-y:b-y,l-x:r-x]=tile[t-ty*th:b-ty*th,l-tx*tw:r-tx*tw,:channels]
 return out[:,:,0] if channels==1 else out

def intersects(a,b):return max(a[0],b[0])<min(a[2],b[2]) and max(a[1],b[1])<min(a[3],b[3])

def category(o):
 c=float(o['pixel_coverage']);i=float(o['best_iou']);ll=json.loads(o['links'])
 if not ll:return None
 link=max(ll,key=lambda r:r['intersection'])
 if 0<c<.5:return 'severe_partial'
 if i>=.5 and c<.9:return 'undercoverage'
 if c>=.9 and link['pred_fraction']<.8:return 'overcoverage'
 if i>=.85 and c>=.9:return 'good_control'
 return None

def main():
 cv2.setNumThreads(1);start=time.time()
 objects=rows(AUDIT/'objects_EVAL_ONLY.csv');components=rows(AUDIT/'components_EVAL_ONLY.csv')
 comp={(r['task_id'],int(r['component_id'])):r for r in components}
 catalog={r['task']['task_id']:r for r in load(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
 # Membership uses offline GT strata. No GT defines a prompt, edit or acceptance.
 quotas={('hospital','HE','severe_partial'):4,('hospital','PAS','severe_partial'):2,
 ('hospital','HE','undercoverage'):4,('hospital','PAS','undercoverage'):4,
 ('hospital','HE','overcoverage'):5,('hospital','PAS','overcoverage'):1,
 ('hospital','HE','good_control'):4,('hospital','PAS','good_control'):4,
 **{('hubmap2021','PAS',s):n for s,n in [('severe_partial',6),('undercoverage',8),('overcoverage',6),('good_control',8)]}}
 used=set();boxes={};selected=[];counts=Counter()
 for stratum,quota in quotas.items():
  pool=[o for o in objects if (o['dataset'],o['stain'],category(o))==stratum]
  pool.sort(key=lambda o:hashlib.sha256((o['task_id']+':'+o['instance_id']).encode()).hexdigest())
  for o in pool:
   if counts[stratum]>=quota:break
   tid=o['task_id'];ll=json.loads(o['links']);cid=max(ll,key=lambda r:r['intersection'])['component_id']
   if (tid,cid) in used:continue
   p=comp[tid,cid];pb=json.loads(p['box']);t=catalog[tid]['task'];mpp=t['source_mpp'];w,h=t['native_size']
   side=int(np.ceil(max(512/mpp,2*max(pb[2]-pb[0],pb[3]-pb[1]))))
   if side>3072:continue
   cx=(pb[0]+pb[2])/2;cy=(pb[1]+pb[3])/2
   box=[max(0,int(cx-side/2)),max(0,int(cy-side/2)),min(w,int(cx+side/2)),min(h,int(cy+side/2))]
   if any(intersects(box,b) for b in boxes.get(tid,[])):continue
   boxes.setdefault(tid,[]).append(box);used.add((tid,cid));counts[stratum]+=1
   selected.append(dict(case_id=f'C{len(selected)+1:03d}',task_id=tid,component_id=cid,box=box,
    stratum=stratum[2],dataset=stratum[0],stain=stratum[1],source_mpp=mpp,scale_basis=t['scale_basis'],
    group_id=t['group_id'],source_gt_id=o['instance_id'],original_object=o,pred_box=pb))
 (HERE/'online').mkdir(exist_ok=True);(HERE/'evaluation').mkdir(exist_ok=True)
 source_records={};summary=[];manifest_cache={}
 for s in selected:
  tid=s['task_id'];c=catalog[tid];t=c['task'];ds=t['dataset'];st=t['stain'];sid=t['source_wsi_id']
  base=DATA/('hubmap-kidney-segmentation/train' if ds=='hubmap2021' else f'医院肾小球数据/2026_11_1_many/glomerulis_for_agents/{st}')
  image=base/'images'/Path(t['image']).name
  manifest=base/'instance_masks'/('instance_'+sid)/'wsi_gt_manifest.json'
  mask=ROOT/'research/full_networks_20261105/predictions/segformer_b5'/tid/'mask.tif'
  frozen=load(ROOT/'research/segformer_complementarity_20261106/cases'/(tid+'.json'))
  if tid not in source_records:
   got=sha(mask);assert got==frozen['audit'][0]['mask_sha256']
   manifest_sha=sha(manifest);assert manifest_sha==frozen['instance_manifest_sha256']
   source_records[tid]=dict(image=str(image),mask=str(mask),mask_sha256=got,manifest=str(manifest),manifest_sha256=manifest_sha,
     baseline=frozen['baseline'],task=t)
   manifest_cache[tid]=load(manifest)['slides'][0]['instances']
  box=s['box'];x,y,xx,yy=box
  rgb=roi(image,box);a=roi(mask,box)>0
  assert a.shape==rgb.shape[:2]
  # Target component is selected only by overlap with its prediction box seed.
  n,labels,stats,_=cv2.connectedComponentsWithStats(a.astype(np.uint8),8)
  pb=s['pred_box'];seed=(slice(max(0,pb[1]-y),min(yy-y,pb[3]-y)),slice(max(0,pb[0]-x),min(xx-x,pb[2]-x)))
  ids,ns=np.unique(labels[seed][a[seed]],return_counts=True)
  target=int(ids[np.argmax(ns)]);target_mask=labels==target
  assert int(target_mask.sum())==int(comp[tid,s['component_id']]['pred_pixels'])
  gt=np.zeros_like(a);instances=[]
  for item in manifest_cache[tid]:
   gb=item['global_box_xyxy']
   if not intersects(box,gb):continue
   gm=np.asarray(Image.open(manifest.parent/item['mask_path']))>0
   l,top,r,b=max(x,gb[0]),max(y,gb[1]),min(xx,gb[2]),min(yy,gb[3])
   q=np.zeros_like(a);q[top-y:b-y,l-x:r-x]=gm[top-gb[1]:b-gb[1],l-gb[0]:r-gb[0]]
   gt|=q;instances.append(dict(id=str(item['instance_id']),gt_pixels_full=int(gm.sum()),box=gb,visible_pixels=int(q.sum()),
     source_class=item.get('source_class_name','unknown'),mask=q))
  on=HERE/'online'/s['case_id'];on.mkdir(exist_ok=True)
  ev=HERE/'evaluation'/s['case_id'];ev.mkdir(exist_ok=True)
  Image.fromarray(rgb).save(on/'image.png');Image.fromarray(a.astype(np.uint8)*255).save(on/'baseline.png')
  Image.fromarray(target_mask.astype(np.uint8)*255).save(on/'target.png')
  payload=dict(schema='local_mask_refinement_v1',case_id=s['case_id'],image=str(on/'image.png'),baseline=str(on/'baseline.png'),
   target=str(on/'target.png'),native_size=[xx-x,yy-y],source_mpp=s['source_mpp'],scale_basis=s['scale_basis'],species='human',stain=st,
   target_box_xyxy=[pb[0]-x,pb[1]-y,pb[2]-x,pb[3]-y],global_box_xyxy=box)
  dump(on/'task_NO_GT.json',payload)
  Image.fromarray(gt.astype(np.uint8)*255).save(ev/'gt.png')
  inst_meta=[]
  for ii,g in enumerate(instances):
   fn=f'instance_{ii:03d}.png';Image.fromarray(g.pop('mask').astype(np.uint8)*255).save(ev/fn);g['mask']=fn;inst_meta.append(g)
  dump(ev/'instances.json',inst_meta)
  s['input_sha256']={f:sha(on/f) for f in ['image.png','baseline.png','target.png','task_NO_GT.json']}
  target_g=next(g for g in inst_meta if g['id']==s['source_gt_id'])
  s['target_gt_visible_fraction']=target_g['visible_pixels']/target_g['gt_pixels_full']
  summary.append({k:v for k,v in s.items() if k not in ['original_object','input_sha256']})
  print('PREPARED',s['case_id'],ds,st,s['stratum'],a.shape,flush=True)
 dump(HERE/'benchmark_EVAL_ONLY.json',selected);dump(HERE/'source_records_EVAL_ONLY.json',source_records)
 csvout(HERE/'cases_EVAL_ONLY.csv',summary)
 dump(HERE/'preparation.json',dict(cases=len(selected),requested=56,counts={str(k):v for k,v in counts.items()},
    sampling='GT-stratified deterministic development capability benchmark, disjoint prediction-derived windows',
    candidate_box='center frozen predicted component; square side max(512um, 2x longest prediction box side), <=3072px',
    gt_enters_online_payload=False,mask_source_sha_verified=True,instance_manifest_sha_verified=True,seconds=time.time()-start))

if __name__=='__main__':main()
