"""Freeze image-only observation assets; no LLM calls or GT decision features."""
import sys,hashlib,html,time
import numpy as np,cv2
from PIL import Image
from common import *
sys.path.insert(0,str(ROOT))
from glomerulus_tools.candidate_views import CandidateViews
from glomerulus_tools.vlm_jpeg import jpeg_message

def main():
 start=time.monotonic();cv2.setNumThreads(1)
 cases=read(OLD/'benchmark_EVAL_ONLY.json');cases.sort(key=lambda s:hashlib.sha256(('obs20261115:'+s['case_id']).encode()).hexdigest())
 online=[];linkage=[];cards=[]
 for i,s in enumerate(cases,1):
  vid=f'V{i:04d}';cid=s['case_id'];source=OLD/'online'/cid;out=HERE/'observations_NO_GT'/vid;out.mkdir(parents=True,exist_ok=True)
  target=np.asarray(Image.open(source/'target.png'))>0
  proposal=np.asarray(Image.open(OLD/'results/ritm_hrnet32_external_mask_point1'/cid/'proposal.png'))>0
  union=target|proposal;ys,xs=np.where(union);box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1];x,y,xx,yy=box
  np.savez_compressed(out/'support.npz',a=target[y:yy,x:xx],b=proposal[y:yy,x:xx])
  task=dict(candidate_id=vid,image=str(source/'image.png'),image_sha256=sha(source/'image.png'),native_size=list(target.shape[1::-1]),
       stain=s['stain'],source_mpp=s['source_mpp'],scale_basis=s['scale_basis'],box=box,
       support_npz=str(out/'support.npz'),support_sha256=sha(out/'support.npz'))
  observer=CandidateViews(task,out/'views');receipt=[];display=[]
  for name in ['initial','context','native_center']:
   paths,rec=observer.render(name);jpeg_message('Read-only observation. No annotation supplied.',paths,out/'JPEG')
   receipt.append(rec)
   # Explicit named JPEG copies for human audit; VLM asset manifest only references JPEG.
   for j,path in enumerate(paths):
    dest=out/'JPEG'/(name+f'_{j}.jpg')
    with Image.open(path) as im:im.convert('RGB').save(dest,quality=95,subsampling=0)
    display.append((name+(' raw' if j==0 else ' candidate support'),dest))
  with Image.open(source/'image.png') as im:rgb=np.asarray(im.convert('RGB')).copy()
  baseedge=target&~cv2.erode(target.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
  newedge=proposal&~cv2.erode(proposal.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
  rgb[baseedge]=[255,220,0];rgb[newedge]=[0,180,255]
  diff=Image.fromarray(rgb);diff.thumbnail((1024,1024));diff.save(out/'JPEG/difference.jpg',quality=95,subsampling=0)
  display.append(('current yellow / proposal cyan',out/'JPEG/difference.jpg'))
  assets={name:{'path':str(path),'sha256':sha(path)} for name,path in display}
  # IDs are shuffled and carry no sclerosis/outcome label. Paths remain internal
  # provenance; never send paths or offline membership to a model.
  package=dict(candidate_id=vid,stain=s['stain'],source_mpp=s['source_mpp'],views=assets,
    current_tool='SegFormer',proposed_action='RITM external-mask interior-positive-point',
    questions=['Is the depicted candidate a glomerulus or uncertain?','Does the proposed contour improve boundaries without adding surrounding tissue?',
               'What extra view is needed, if any?','Accept, retain, or request manual review?'],
    GT_read=False,allowed_view_names=list(assets),action_budget=1,observation_budget=3,
    no_GT_fields_or_labels=True)
  save(out/'package_NO_GT.json',package);online.append(package)
  linkage.append(dict(view_id=vid,case_id=cid,cohort=s['stratum'],task_id=s['task_id'],group_id=s['group_id'],
     original_selection='GT-informed capability cohort; not online discovery; 3 entirely omitted cases absent'))
  panels=''.join(f'<figure><figcaption>{html.escape(name)}</figcaption><img loading="lazy" src="../../../glomerulus_tools/experiments/followup_20261115/observations_NO_GT/{vid}/JPEG/{p.name}"></figure>' for name,p in display)
  cards.append(f'<section id="{vid}"><h2>{vid} · {s["stain"]}</h2><div class="grid">{panels}</div></section>')
  print('OBS',vid,flush=True)
 save(HERE/'observation_packages_NO_GT.json',online);save(HERE/'observation_linkage_EVAL_ONLY.json',linkage)
 (REPORT/'无GT观察样例.html').write_text('<!doctype html><meta charset="utf8"><title>无GT观察资产</title><style>body{font:16px system-ui;margin:20px}section{padding:20px;background:#eee;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}img{width:100%}figure{margin:0}</style><h1>77个已定位对象的观察资产</h1><p>原图、预测支持、上下文、原生局部及修改差异；全部JPEG，无GT图层。本轮未让VLM判定，未证明观察能改善决策；样本来自已暴露GT分组队列，不是无GT发现失败。此处可比较不同观察内容是否足以回答问题。</p>'+''.join(cards),encoding='utf8')
 save(HERE/'observation_preparation.json',dict(packages=77,views_per_case=7,total_named_JPEG=539,
      no_model_calls=True,GT_not_read_by_view_renderer=True,seconds=time.monotonic()-start,
      missing_cases='3 complete omissions cannot be discovered by this candidate-based observation mechanism'))

if __name__=='__main__':main()
