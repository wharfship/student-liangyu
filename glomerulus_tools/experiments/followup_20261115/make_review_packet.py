"""Human-only native-resolution evidence; no inferred pathology diagnoses."""
import importlib.util,html,time
import numpy as np,cv2
from PIL import Image,ImageDraw
from common import *

spec=importlib.util.spec_from_file_location('roi_helpers',OLD.parent/'problem_to_repair_20261111/prepare_cases.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def contour(rgb,mask,color):
 out=rgb.copy();e=mask&~cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8')).astype(bool);out[e]=color;return out

def main():
 cv2.setNumThreads(1);tick=time.monotonic();out=REPORT/'人工复核';out.mkdir(parents=True,exist_ok=True)
 cohort=read(OLD/'cohort_EVAL_ONLY.json');members={s['case_id']:s for s in read(OLD/'specialist_membership_EVAL_ONLY.json')}
 catalog={c['task']['task_id']:c['task'] for c in read(ROOT/'research/segformer_complementarity_20261106/catalog.json')}
 scores={r['case_id']:r for r in rows(ROOT/'已经整理好的内容/医院分割结果分析/硬化专项验证_20261114/专项逐对象结果_EVAL_ONLY.csv') if r['method']=='samrefiner_vitb_paper_fp16'}
 allrows=[];cards=[]
 for o in cohort[:40]:
  cid=o['case_id'];s=members[cid];t=catalog[o['task_id']];gbox=json.loads(o['box']);cx=(gbox[0]+gbox[2])//2;cy=(gbox[1]+gbox[3])//2
  side=max(1024,2*max(gbox[2]-gbox[0],gbox[3]-gbox[1]));x=max(s['box_native'][0],cx-side//2);y=max(s['box_native'][1],cy-side//2)
  xx=min(s['box_native'][2],x+side);yy=min(s['box_native'][3],y+side)
  w,h=t['native_size'];assert 0<=x<xx<=w and 0<=y<yy<=h
  image=Path('D:/数据集/医院肾小球数据/2026_11_1_many/glomerulis_for_agents')/o['stain']/'images'/Path(t['image']).name
  raw=helper.roi(image,[x,y,xx,yy]);a,b=s['box_native'][:2]
  with np.load(OLD/'specialist_evaluation'/cid/'labels_EVAL_ONLY.npz') as f:
   target=f['target'][y-b:yy-b,x-a:xx-a];gt=f['gt'][y-b:yy-b,x-a:xx-a];base=f['baseline'][y-b:yy-b,x-a:xx-a]
  assert raw.shape[:2]==target.shape
  views=[('原图',raw),('原预测紫',contour(raw,base,(230,20,230))),('目标GT橙其他GT绿',contour(contour(raw,gt,(20,220,20)),target,(255,150,0)))]
  err=raw.copy();err[base&~gt]=[240,40,40];err[gt&~base]=[0,190,255];views.append(('整窗差异：多分红、欠分蓝',err))
  panels=[]
  for i,(name,rgb) in enumerate(views):
   im=Image.fromarray(rgb);im.thumbnail((1000,1000));fn=f'{cid}_{i}.jpg';im.save(out/fn,quality=94)
   panels.append(f'<figure><figcaption>{name}</figcaption><a href="人工复核/{fn}"><img loading="lazy" src="人工复核/{fn}"></a></figure>')
  status='完全无预测' if float(o['pixel_coverage'])==0 else '覆盖不足90%' if float(o['pixel_coverage'])<.9 else '已覆盖≥90%'
  row=dict(case_id=cid,task_id=o['task_id'],instance_id=o['instance_id'],stain=o['stain'],group_id=o['group_id'],
    recorded_class=o['source_class'],baseline_dice=float(scores[cid]['target_dice_before']),coverage=float(o['pixel_coverage']),
    geometry_status=status,native_box=[x,y,xx,yy],source_mpp=t['source_mpp'],
    source_low_gray_fraction=float(np.mean(raw.max(axis=2)-raw.min(axis=2)<5)),
    source_white_fraction=float(np.mean(np.min(raw,axis=2)>245)),
    pathology_confirmed='待人工复核',GT_boundary_definition='待人工复核',confounding_tissue='未作病理判断',
    image_quality_issue='待人工复核',recommended_action='待研究者判定',human_notes='')
  allrows.append(row)
  cards.append((row['baseline_dice'],f'<section id="{cid}"><h2>{cid} · {o["stain"]} · {status} · Dice {row["baseline_dice"]:.4f}</h2><p>{html.escape(o["task_id"])} / 原ID{o["instance_id"]}。倍率/染色/颜色不能替代病理复诊。</p><div class="grid">'+''.join(panels)+'</div></section>'))
  print('REVIEW',cid,status,flush=True)
 csvout(REPORT/'硬化40_人工复核工作表_EVAL_ONLY.csv',sorted(allrows,key=lambda r:r['baseline_dice']))
 page='''<!doctype html><meta charset="utf-8"><title>硬化40人工复核证据</title><style>body{font:16px system-ui;background:#eee;margin:24px}section{background:white;padding:18px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}img{width:100%}figure{margin:0}</style><h1>全部40硬化：原生高分辨率复核材料</h1><p>仅供离线人工/病理复核，含GT，禁止作为在线Agent观察。按原SegFormer目标Dice升序展示；图像现象不自动归因为组织学因素。原标注类别未由本轮医生重新确认。原图、原预测及原GT只读。</p><p><a href="硬化40_人工复核工作表_EVAL_ONLY.csv">填写复核工作表</a> · <a href="结果汇报.md">本轮汇报</a></p>'''+''.join(card for _,card in sorted(cards))
 (REPORT/'人工复核.html').write_text(page,encoding='utf8')
 save(HERE/'review_packet_receipt.json',dict(cases=40,JPEG=160,GT_display_only=True,pathology_diagnosis_not_performed=True,
   source_format_preserved=True,seconds=time.monotonic()-tick))

if __name__=='__main__':main()
