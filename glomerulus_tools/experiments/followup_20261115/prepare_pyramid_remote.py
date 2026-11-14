"""Exact author OpenSlide level and JPEG encoding on frozen reference boxes."""
from pathlib import Path
import json,hashlib,time,cv2,numpy as np,openslide
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2))

def main():
 start=time.monotonic();plan=json.loads((P/'pyramid_plan_EVAL_ONLY.json').read_text());records=[];tasks={'PNG':[],'JPG':[]};slides={}
 for s in plan:
  cid=s['case_id'];path=s['source_image'];slide=slides.setdefault(path,openslide.OpenSlide(path))
  power=int(slide.properties['openslide.objective-power'])
  ops=np.array([power*d[0]/slide.dimensions[0] for d in slide.level_dimensions]);level=int(np.argmin(np.abs(ops-10)))
  closest=float(ops[level]);allowed=abs((closest-10)/10)<.1
  record=dict(case_id=cid,level=level,level_dimensions=[list(x) for x in slide.level_dimensions],objective=power,level_powers=ops.tolist(),
      closest_op=closest,author_check_passed=allowed,scope=s['scope'],GT_read=False)
  if not allowed:records.append(record);print('REJECT',cid,ops.tolist(),flush=True);continue
  factor=power/closest;x,y,xx,yy=s['box_native'];side=int(4096/round(factor,1))
  # These sources have no stitching offset; do not silently ignore it if present.
  offsets={k:v for k,v in slide.properties.items() if 'COMPRESSED_STITCHING_ORIG_SLIDE_SCANNED_AREA_IN_PIXELS' in k}
  assert not offsets,offsets
  rgb=np.array(slide.read_region((x,y),level,(side,side)).convert('RGB'))
  out=P/'pyramid_inputs'/cid;out.mkdir(parents=True,exist_ok=False)
  cv2.imwrite(str(out/'image.png'),rgb[:,:,::-1]);cv2.imwrite(str(out/'image.jpg'),rgb[:,:,::-1])
  old=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(s['previous_image']).convert('RGB'))
  record.update(factor=factor,size=[side,side],old_same_shape=rgb.shape==old.shape,
    rgb_mean_abs_difference=float(np.abs(rgb.astype('float32')-old).mean()) if rgb.shape==old.shape else None,
    coordinate_mapping='level coordinates scaled back to fixed original4096 window',box_native=s['box_native'])
  for mode,ext in [('PNG','png'),('JPG','jpg')]:
   image=out/('image.'+ext);task=dict(task_id=cid+mode,image=str(image),species='human',stain=s['stain'],
    target='whole_glomerular_unit',input_form='patch',native_size=[side,side],source_mpp=s['native_mpp']*factor,
    scale_basis='mescnn_metadata_10x_prepared_tile',scale_evidence='author nearest actual level10x within10%; nativeMPP recorded')
   tasks[mode].append(task);record[mode+'_sha256']=sha(image)
  records.append(record);print('PREP',cid,level,closest,side,flush=True)
 for key,t in tasks.items():save(P/('pyramid_'+key+'_NO_GT.json'),dict(schema='mescnn_prepared_tile_batch_v1',cases=t))
 save(P/'pyramid_preparation.json',dict(records=records,prepared=len(tasks['PNG']),rejected=sum(not r['author_check_passed'] for r in records),seconds=time.monotonic()-start))
 for slide in slides.values():slide.close()

if __name__=='__main__':main()
