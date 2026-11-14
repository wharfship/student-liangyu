"""All80 original and author-input variants; no selected-success-only gallery."""
import numpy as np,cv2,html
from PIL import Image
from common import *

def main():
 cv2.setNumThreads(1);out=REPORT/'金字塔图';out.mkdir(exist_ok=True)
 cohort=read(OLD/'cohort_EVAL_ONLY.json');ev=rows(REPORT/'金字塔配对逐目标_EVAL_ONLY.csv');index={(r['case_id'],r['mode']):r for r in ev};cards=[]
 for o in cohort:
  cid=o['case_id'];raw=np.array(Image.open(OLD/'specialist_online'/cid/'image.png').convert('RGB').resize((800,800)))
  with np.load(OLD/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:gt=f['gt'];target=f['target']
  masks=[('原图',None),('原配方MESCnn',np.asarray(Image.open(OLD/'specialist_results'/cid/'mask.png'))>0)]
  for mode in ['PNG','JPG']:
   masks.append(('实际金字塔'+mode+f"：Dice {float(index[cid,mode]['new_target_dice']):.4f}",np.asarray(Image.open(HERE/('pyramid_results_'+mode)/(cid+mode)/'mask.png'))>0))
  panels=[]
  for i,(title,mask) in enumerate(masks):
   rgb=raw.copy()
   if mask is not None:
    for a,c in [(gt,[20,220,20]),(target,[255,150,0]),(mask,[230,20,230])]:
     b=cv2.resize(a.astype('uint8'),(800,800),interpolation=cv2.INTER_NEAREST);edge=(b>0)&(cv2.erode(b,np.ones((3,3),'uint8'))==0);rgb[edge]=c
   fn=cid+'_'+str(i)+'.jpg';Image.fromarray(rgb).save(out/fn,quality=93)
   panels.append(f'<figure><figcaption>{title}</figcaption><a href="金字塔图/{fn}"><img loading="lazy" src="金字塔图/{fn}"></a></figure>')
  cards.append(f'<section class="case" data-cohort="{o["cohort"]}" id="{cid}"><h2>{cid} · {o["stain"]} · '+('硬化' if o['cohort']=='sclerosis' else '普通保护')+'</h2><div class="grid">'+''.join(panels)+'</div></section>')
 page='''<!doctype html><meta charset="utf8"><title>MESCnn输入链配对</title><style>body{font:16px system-ui;margin:20px;background:#eee}section{background:white;margin:20px;padding:15px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}img{width:100%}figure{margin:0}nav{position:sticky;top:0;background:white;padding:12px}</style><h1>MESCnn真实金字塔与JPEG输入链：全80例</h1><p>橙=目标GT，绿=其他GT，紫=预测；保持权重、原生4096窗与所有评价目标。PNG隔离金字塔差异，JPEG再匹配作者默认压缩。3个完全遗漏仍仅GT定位能力参考，没有完整WSI搜索。原配方结果不覆盖。</p><p><a href="结果汇报.md">本轮汇报</a></p><nav><select id="g"><option value="ALL">全部80</option><option value="sclerosis">硬化40</option><option value="normal_control">普通40</option></select> <input id="q" placeholder="S019"></nav>'''+''.join(cards)+'''<script>function filter(){let g=document.querySelector('#g').value,q=document.querySelector('#q').value.toUpperCase();document.querySelectorAll('.case').forEach(e=>e.hidden=(g!='ALL'&&e.dataset.cohort!=g)||(q&&!e.id.includes(q)))}document.querySelector('#g').onchange=filter;document.querySelector('#q').oninput=filter;</script>'''
 (REPORT/'金字塔配对.html').write_text(page,encoding='utf8');save(HERE/'paired_gallery_receipt.json',dict(cases=80,JPEG=320,all_cases=True))

if __name__=='__main__':main()
