"""Show every acceptance sample, with task-appropriate outputs and limitations."""
import csv,html,shutil
import numpy as np,cv2
from PIL import Image
from common import *

def jpg(src,name):
 dest=REPORT/'images'/name;dest.parent.mkdir(exist_ok=True)
 if isinstance(src,np.ndarray):im=Image.fromarray(src)
 else:im=Image.open(src).convert('RGB')
 im.thumbnail((1000,1000));im.save(dest,quality=94);im.close();return 'images/'+name

def panel(title,path):return f'<figure><figcaption>{html.escape(title)}</figcaption><a href="{path}"><img loading="lazy" src="{path}"></a></figure>'
def section(group,sid,title,content):return f'<section class="case" data-group="{group}" id="{group}_{sid}"><h2>{html.escape(title)}</h2><div class="grid">{content}</div></section>'

def main():
 REPORT.mkdir(parents=True,exist_ok=True);cards=[]
 for s in read(HERE/'nuclei_tasks_NO_GT.json')['samples']:
  sid=s['sample_id'];r=read(HERE/'results/nuclei_hospital16'/sid/'record.json')
  content=panel('原图 · '+s['stain'],jpg(s['image'],sid+'_nuc_raw.jpg'))
  labels=np.asarray(Image.open(HERE/'results/nuclei_hospital16'/sid/'instances.tif'));rgb=np.array(Image.open(s['image']).convert('RGB'))
  edge=np.zeros(labels.shape,bool);edge[1:]|=labels[1:]!=labels[:-1];edge[:,1:]|=labels[:,1:]!=labels[:,:-1];rgb[edge&(labels>0)]=[0,240,100]
  content+=panel('细胞核实例边界；无本地核GT',jpg(rgb,sid+'_nuc_pred.jpg'))
  cards.append(section('nuclei',sid,f"{sid} {s['stain']} · {r['instances']}核 · 输入MPP{s['source_mpp']:.4f}",content))
 metrics={r['sample_id']:r for r in csv.DictReader((REPORT/'TNBC8_细胞核验收_EVAL_ONLY.csv').open(encoding='utf-8-sig'))}
 for s in read(HERE/'tnbc_tasks_NO_GT.json')['samples']:
  sid=s['sample_id'];link=next(r for r in read(HERE/'tnbc_labels_EVAL_ONLY.json') if r['sample_id']==sid)
  raw=np.array(Image.open(s['image']).convert('RGB'));gt=np.asarray(Image.open(link['gt']))>0
  truth=raw.copy();edge=gt&(cv2.erode(gt.astype('uint8'),np.ones((3,3),'uint8'))==0);truth[edge]=[255,150,0]
  content=panel('原图',jpg(raw,sid+'_raw.jpg'))+panel('标注橙（训练来源验收）',jpg(truth,sid+'_gt.jpg'))+panel('InstanSeg预测绿',jpg(HERE/'results/nuclei_TNBC8'/sid/'overlay.jpg',sid+'_pred.jpg'))
  cards.append(section('tnbc',sid,sid+' · Dice '+f"{float(metrics[sid]['binary_dice']):.3f}"+' · 不属于独立盲测',content))
 qtasks={r['sample_id']:r for r in read(HERE/'qc_paired_tasks_NO_GT.json')['samples']}
 for pair in read(HERE/'qc_blur_linkage_EVAL_ONLY.json'):
  sid=pair['original'];bid=pair['blur'];ra=read(HERE/'results/artifact24'/sid/'record.json');rb=read(HERE/'results/artifact24'/bid/'record.json')
  content=panel('原始局部',jpg(qtasks[sid]['image'],sid+'_qc_raw.jpg'))+panel('原始QC',jpg(HERE/'results/artifact24'/sid/'overlay.jpg',sid+'_qc_pred.jpg'))
  content+=panel('人工严重模糊σ=8μm',jpg(qtasks[bid]['image'],sid+'_blur.jpg'))+panel('模糊后QC',jpg(HERE/'results/artifact24'/bid/'overlay.jpg',sid+'_blur_pred.jpg'))
  cards.append(section('quality',sid,f"{sid} · OOF比例 {ra['class_fractions']['6']:.3f} → {rb['class_fractions']['6']:.3f} · 人工扰动",content))
 features={r['sample_id']:r for r in read(HERE/'feature_tasks_NO_GT.json')['samples']};labels={r['sample_id']:r for r in read(HERE/'feature_labels_EVAL_ONLY.json')}
 for r in csv.DictReader((REPORT/'77候选_跨来源组同染色检索_EVAL_ONLY.csv').open(encoding='utf-8-sig')):
  if r['model']!='DINOv2':continue
  sid=r['sample_id'];content=panel('查询 · '+labels[sid]['cohort'],jpg(features[sid]['image'],sid+'_retrieval.jpg'))
  for j,nid in enumerate(__import__('json').loads(r['neighbors'])[:3],1):content+=panel(f'近邻{j} {nid} · '+labels[nid]['cohort'],jpg(features[nid]['image'],nid+'_retrieval.jpg'))
  cards.append(section('features',sid,sid+' · 同染色、排除同来源号组的近邻；标签仅离线展示',content))
 page='''<!doctype html><meta charset="utf-8"><title>开源工具实际验收</title><style>body{font:16px system-ui;margin:24px;background:#eee;color:#222}section{background:white;margin:20px 0;padding:18px}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}figure{margin:0}img{width:100%}nav{position:sticky;top:0;background:white;padding:14px;z-index:5}p{max-width:1200px}</style><h1>新工具：实际输出与小范围验收</h1><p>InstanSeg核实例：医院16局部无核GT，TNBC8有标注但属于已知训练来源；DINOv2为通用表征，不是肾小球分类器；GrandQC为组织/伪影研究质控、非商业许可。人工模糊响应不代表真实伪影敏感性，所有组保留全部样本。</p><p>QC颜色：组织白、折叠橙、暗点紫、笔迹蓝、气泡/边缘黄、失焦红、背景黑；未知类0品红保留。核边界绿、TNBC标注橙。检索标签只离线显示，不进入特征网络。</p><p><a href="结果报告.md">完整结果与用法</a></p><nav><select id="g"><option value="ALL">全部</option><option value="nuclei">医院细胞核16</option><option value="tnbc">公开细胞核验收8</option><option value="quality">质量扰动12对</option><option value="features">跨组检索77</option></select> <input id="q" placeholder="P004 / TNBC01"></nav>'''+''.join(cards)+'''<script>function filter(){let g=document.querySelector('#g').value,q=document.querySelector('#q').value.toUpperCase();document.querySelectorAll('.case').forEach(e=>e.hidden=(g!='ALL'&&e.dataset.group!=g)||(q&&!e.id.toUpperCase().includes(q)))}document.querySelector('#g').onchange=filter;document.querySelector('#q').oninput=filter;</script>'''
 (REPORT/'gallery.html').write_text(page,encoding='utf8');save(HERE/'gallery_receipt.json',dict(cases=len(cards),groups={'nuclei':16,'tnbc':8,'quality':12,'features':77},JPG=len(list((REPORT/'images').glob('*.jpg')))))
 print('All113 cards written',flush=True)

if __name__=='__main__':main()
