"""Only update the two experimental recipes in the sole catalog after scoring."""
import csv
from common import *


def main():
 path=ROOT/'glomerulus_tools/catalog.csv';before=sha(path);rr=rows(path)
 reference={r['tool_id']:dict(r) for r in rr};summary=read(HERE/'paired_summary.json')
 for recipe,mode in [('png','PNG'),('jpeg','JPG')]:
  row=next(r for r in rr if r['tool_id']=='mescnn_cascade_r50_pyramid_'+recipe)
  s=next(r for r in summary if r['mode']==mode and r['cohort']=='sclerosis' and r['stain']=='ALL')
  n=next(r for r in summary if r['mode']==mode and r['cohort']=='normal_control' and r['stain']=='ALL')
  row.update(status='research_paired80_scored_no_sclerosis_net_gain',
   verification='20261115真实80/80输出，严格权重加载；输入SHA/尺寸/实例并集检查；执行快照与远程21文件SHA一致；exit0',
   results=f"同40硬化目标Dice{s['new']:.6f} vs SegFormer{s['baseline']:.6f}，改善/恶化>0.01={s['improved001']}/{s['worsened001']}；普通40 Dice{n['new']:.6f}，匹配31/40；三例GT定位遗漏参考均未救回；非完整WSI/NMS",
   researcher_note='同一MESCnn权重的输入配方，不是新网络；无硬化净收益，普通误伤。已暴露开发样本，普通为已检出保护对照，训练成员/精确MPP未知，Agent禁用。')
 changed=[r['tool_id'] for r in rr if r!=reference[r['tool_id']]]
 assert set(changed)<= {'mescnn_cascade_r50_pyramid_png','mescnn_cascade_r50_pyramid_jpeg'}
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
 save(HERE/'catalog_finalization.json',dict(before=before,after=sha(path),changed=changed,
  agent_enabled_changed=False,default_changed=False,accuracy_scope='paired80 reference windows only'))
 print('Scored recipes updated:',changed)


if __name__=='__main__':main()
