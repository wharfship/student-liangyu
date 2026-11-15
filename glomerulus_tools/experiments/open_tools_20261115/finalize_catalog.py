"""Replace pending statuses with observed outcomes in the sole source catalog."""
import csv
from common import *

def main():
 p=ROOT/'glomerulus_tools/catalog.csv';rr=list(csv.DictReader(p.open(encoding='utf-8-sig')));before=sha(p)
 ss=read(HERE/'acceptance_summary.json');control=read(HERE/'fixed_fov_control_summary.json')
 updates={
 'instanseg_brightfield_nuclei_v011':('research_CUDA24_verified_nuclei_acceptance_promising',
  'CUDA医院16局部＋TNBC8完成，实例尺寸/输入SHA通过，smoke重复字节一致；模型SHA与作者rdf一致',
  f"TNBC8/4slides核前景Dice{ss['nuclei_TNBC']['mean_binary_dice']:.6f}，组件代理F1{ss['nuclei_TNBC']['mean_instance_f1']:.6f}；该数据为已知训练来源，不是独立测试；医院无核GT，PAS4仅探索"),
 'dinov2_vits14_features':('research_CUDA154_verified_retrieval_promising_FOV_sensitive',
  'CUDA77原候选＋77固定224um视野完成，384维有限向量及L2检查；无重复RGB图；源组排除；smoke重复字节一致',
  '同染色跨来源号组同类首近邻：原裁片73/77 vs RGB51/77；固定224um为63/77 vs RGB42/77。检索任务开发验收，不是肾病分类准确率；无专用头/无训练'),
 'grandqc_tissue_mpp10':('research_CUDA24_verified_tissue_gate_no_real_GT',
  '24HE局部（12原图＋12人工模糊）CUDA完成；任务state_dict严格加载，官方Zenodo MD5核对；输出/SHA通过',
  '组织门控可执行，真实组织边界GT尚未评价；10um/px粗格与白补512的ROI协议，非完整作者WSI复现；非商业许可'),
 'grandqc_artifact_mpp15':('research_CUDA24_verified_synthetic_defocus_response',
  '完整作者任务模型＋组织门控，24HE局部CUDA完成；Zenodo MD5核对；重复输出字节一致',
  '人工σ8um严重模糊12/12失焦比例增加>0.01，均值0→0.649134；无真实伪影GT，不作为真实敏感性/特异性；1.5um/px、非商业许可')}
 for r in rr:
  if r['tool_id'] in updates:
   r['status'],r['verification'],r['results']=updates[r['tool_id']]
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
 save(HERE/'catalog_finalization.json',dict(before=before,after=sha(p),updated=list(updates),default_changed=False,Agent_enabled_changed=False))
 print('Four observed utility statuses recorded; Agent pool unchanged')

if __name__=='__main__':main()
