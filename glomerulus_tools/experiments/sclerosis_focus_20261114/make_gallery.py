"""All80 objects plus12 grid interventions, shared image coordinates, JPEG views."""
import html,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from prepare import HERE,REPORT,read,save,csvread

def edge(mask,size):
    a=cv2.resize(mask.astype('uint8'),size,interpolation=cv2.INTER_NEAREST)
    return (a>0)&(cv2.erode(a,np.ones((3,3),'uint8'))==0)

def panel(raw,gt,mask,path,target=None):
    rgb=np.asarray(raw.resize((800,800))).copy()
    if gt is not None:rgb[edge(gt,(800,800))]=(35,220,35)
    if target is not None:rgb[edge(target,(800,800))]=(255,175,0)
    if mask is not None:rgb[edge(mask,(800,800))]=(230,30,220)
    Image.fromarray(rgb).save(path,quality=90)

def place(arr,box,dest,output_box):
    x,y,xx,yy=box;l,t,r,b=output_box
    ll,tt,rr,bb=max(x,l),max(y,t),min(xx,r),min(yy,b)
    if ll<rr and tt<bb:dest[tt-t:bb-t,ll-l:rr-l]=arr[tt-y:bb-y,ll-x:rr-x]

def main():
    assets=REPORT/'gallery';assets.mkdir(exist_ok=True)
    members=read(HERE/'specialist_membership_EVAL_ONLY.json')
    refs={s['case_id']:s for s in read(HERE/'benchmark_EVAL_ONLY.json')}
    scores=csvread(REPORT/'专项逐对象结果_EVAL_ONLY.csv')
    special=csvread(REPORT/'MESCnn逐对象_EVAL_ONLY.csv')
    index={(r['case_id'],r['method']):r for r in scores}
    mes={r['case_id']:r for r in special if r['area_rule']=='raw'}
    cards=[]
    for s in members:
        cid=s['case_id'];raw=Image.open(HERE/'specialist_online'/cid/'image.png').convert('RGB')
        with np.load(HERE/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:gt=f['gt'];baseline=f['baseline'];target=f['target']
        outputs=[('原图',None,None),('GT绿线／SegFormer紫线',gt,baseline)]
        for method,label in [('samrefiner_vitb_paper_fp16','SAMRefiner'),('ritm_hrnet32_external_mask_point1','RITM')]:
            edited=baseline.copy()
            if cid in refs:
                ref=refs[cid];arr=np.asarray(Image.open(HERE/'results'/method/cid/'mask.png'))>0
                place(arr,ref['box'],edited,s['box_native'])
                delta=float(index[cid,method]['delta_target_dice'])
                title=f'{label}：目标ΔDice {delta:+.4f}'
            else:title=label+'：无粗mask，不可调用'
            outputs.append((title,gt,edited))
        instance=np.asarray(Image.open(HERE/'specialist_results'/cid/'instances.tif'))
        merged=cv2.resize((instance>0).astype('uint8'),(4096,4096),interpolation=cv2.INTER_NEAREST)>0
        outputs.append(('MESCnn独立分割：目标ΔDice '+f"{float(mes[cid]['delta_target_dice']):+.4f}",gt,merged))
        panels=[]
        for j,(title,truth,mask) in enumerate(outputs):
            name=f'{cid}_{j}.jpg';panel(raw,truth,mask,assets/name,target if truth is not None else None)
            panels.append(f'<figure><figcaption>{html.escape(title)}</figcaption><a href="gallery/{name}" target="_blank"><img loading="lazy" src="gallery/{name}"></a></figure>')
        label='硬化' if s['cohort']=='sclerosis' else '普通对照'
        scope='；此例为GT定位的能力参考，不是部署补漏' if s['scope'].startswith('GT_') else ''
        cards.append(f'<section class="case" data-group="{s["cohort"]}" id="{cid}"><h2>{cid} · {label} · {s["stain"]}</h2><p>{html.escape(s["task_id"])}{scope}</p><div class="grid">'+''.join(panels)+'</div></section>')
        print('GALLERY',cid,flush=True)
    old=HERE.parent/'problem_to_repair_20261111';seam=csvread(REPORT/'接缝反事实逐例_EVAL_ONLY.csv');seamcards=[]
    for row in seam:
        cid=row['case_id'];raw=Image.open(old/'online'/cid/'image.png').convert('RGB')
        gt=np.asarray(Image.open(old/'evaluation'/cid/'gt.png'))>0
        items=[]
        for phase in [0,256]:
            arr=np.asarray(Image.open(HERE/'seam_validation'/cid/f'phase{phase}.png'))>0
            name=f'{cid}_phase{phase}.jpg';panel(raw,gt,arr,assets/name)
            items.append(f'<figure><figcaption>网格偏移{phase}</figcaption><a href="gallery/{name}"><img loading="lazy" src="gallery/{name}"></a></figure>')
        seamcards.append(f'<section><h2>{cid} · 接缝取证 · 整窗ΔDice {float(row["delta_window_dice"]):+.4f}</h2><div class="grid">'+''.join(items)+'</div></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>硬化专项与接缝验证</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f5f5;color:#222}section{padding:20px;background:white;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}figure{margin:0}img{width:100%}figcaption{padding:8px;background:#eee}p{overflow-wrap:anywhere}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2}button,select,input{font:inherit;padding:6px}</style>
<h1>医院40硬化＋40普通对照，及12例切块取证</h1><p>2026-11-14。所有病例均展示。目标GT橙线、其他GT绿线、预测紫线；点图片看大图。每个修复面板为单次局部动作，独立裁窗可能重叠，不能合并为一套系统成绩。MESCnn为预制10×裁片适配，没有完整WSI搜索；3个完全漏检例的裁窗由GT定位，仅作能力参考。既有标注类别不是本次病理复诊。</p>
<p><a href="结果报告.md">结果报告</a> · <a href="专项逐对象结果_EVAL_ONLY.csv">修复逐对象表</a> · <a href="MESCnn逐对象_EVAL_ONLY.csv">MESCnn逐对象表</a></p>
<nav><label>分组 <select id="group"><option value="ALL">全部80</option><option value="sclerosis">硬化40</option><option value="normal_control">普通40</option></select></label> <label>病例 <input id="query" placeholder="S019 / N001"></label> <a href="#seams">接缝12例</a></nav>
'''+''.join(cards)+'<h1 id="seams">切块网格反事实</h1><p>相同权重、相同图像，原网格逐像素复现；只改变网格位置。个别严重恶化也完整保留。</p>'+''.join(seamcards)+'''
<script>function filter(){let g=document.querySelector('#group').value,q=document.querySelector('#query').value.trim().toUpperCase();document.querySelectorAll('.case').forEach(e=>e.hidden=(g!='ALL'&&e.dataset.group!=g)||(q&&!e.id.includes(q)))}document.querySelector('#group').onchange=filter;document.querySelector('#query').oninput=filter;</script></html>'''
    (REPORT/'gallery.html').write_text(page,encoding='utf-8')
    save(HERE/'gallery_receipt.json',dict(objects=len(cards),seam_cases=len(seamcards),jpeg_assets=len(list(assets.glob('*.jpg'))),
      all_cases_included=True,GT_display_only=True,source_files_unchanged=True))

if __name__=='__main__':main()
