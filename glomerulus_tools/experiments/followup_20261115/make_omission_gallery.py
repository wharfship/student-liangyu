"""Display existing independent-model predictions on the three old omissions."""
import importlib.util
import cv2,numpy as np
from PIL import Image
from common import *
spec=importlib.util.spec_from_file_location('roi_reader',OLD.parent/'problem_to_repair_20261111/prepare_cases.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def main():
 out=REPORT/'三例遗漏';out.mkdir(exist_ok=True);cohort={s['case_id']:s for s in read(OLD/'cohort_EVAL_ONLY.json')}
 members={s['case_id']:s for s in read(OLD/'specialist_membership_EVAL_ONLY.json')};cards=[]
 for cid in ['S019','S033','S040']:
  o=cohort[cid];s=members[cid];mask=ROOT/'research/full_networks_20261105/predictions/segnext_mscant'/s['task_id']/'mask.tif'
  assert mask.exists(),mask
  # Verify the exact frozen artifact before reusing its crop.
  case=read(ROOT/'research/segformer_complementarity_20261106/cases'/(s['task_id']+'.json'))
  expected=next(a['mask_sha256'] for a in case['audit'] if a['model']=='segnext_mscant');assert sha(mask)==expected
  x,y,xx,yy=s['box_native'];w,hgt=case['meta']['native_size'] if 'native_size' in case['meta'] else [0,0]
  # Reader accepts in-slide crops; all three frozen boxes are within boundaries.
  pred=h.roi(mask,[x,y,xx,yy])>0;assert pred.shape==(4096,4096)
  with np.load(OLD/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:gt=f['gt'];target=f['target'];base=f['baseline']
  raw=np.asarray(Image.open(OLD/'specialist_online'/cid/'image.png').convert('RGB').resize((1024,1024)))
  panels=[]
  for label,m in [('SegFormer',base),('SegNeXt冻结全图推理',pred)]:
   canvas=raw.copy()
   for a,c in [(gt,(20,220,20)),(target,(255,150,0)),(m,(230,20,230))]:
    b=cv2.resize(a.astype('uint8'),(1024,1024),interpolation=cv2.INTER_NEAREST);canvas[(b>0)&(cv2.erode(b,np.ones((3,3),'uint8'))==0)]=c
   fn=cid+'_'+('baseline' if label=='SegFormer' else 'segnext')+'.jpg';Image.fromarray(canvas).save(out/fn,quality=94)
   panels.append(f'<figure><figcaption>{label}</figcaption><img src="三例遗漏/{fn}"></figure>')
  cards.append('<section><h2>'+cid+'</h2><div class="grid">'+''.join(panels)+'</div></section>')
 (REPORT/'三例遗漏与旧工具.html').write_text('<!doctype html><meta charset="utf8"><title>三例遗漏复用旧工具</title><style>body{font:16px system-ui;margin:20px}section{background:#eee;margin:20px;padding:15px}.grid{display:grid;grid-template-columns:1fr 1fr}img{width:100%}</style><h1>3个硬化遗漏：复用旧SegNeXt完整预测</h1><p>橙=目标GT，绿=其他GT，紫=预测。显示窗口由GT定位，仅供离线核对；SegNeXt本身为已完成、不读GT的全图推理，没有在本轮重新运行。S040旧原始ID一对一匹配IoU0.7969，其他两个没有救回。医院40图SegNeXt独有组件65个中仅1个匹配、64个未匹配，不能用这一成功例证明自动补漏安全。</p>'+''.join(cards),encoding='utf8')
 save(HERE/'omission_gallery_verification.json',dict(cases=3,frozen_sha_verified=True,new_inference=False,GT_centered_display_only=True))

if __name__=='__main__':main()
