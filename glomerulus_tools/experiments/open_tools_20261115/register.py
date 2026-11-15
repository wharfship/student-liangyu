"""Add runnable research utilities to the sole source catalog, preserving defaults."""
import csv
from common import *

def main():
 path=ROOT/'glomerulus_tools/catalog.csv';rr=list(csv.DictReader(path.open(encoding='utf-8-sig')));keys=list(rr[0]);before=sha(path)
 source=HERE/'assets/source';inst=next(source.glob('instanseg-*'));dino=next(source.glob('dinov2-*'));qc=next(source.glob('grandqc-*'))
 rootremote=REMOTE+'/glomerulus_tools/experiments/open_tools_20261115';py=rootremote+'/tool_venv/bin/python'
 configs=[
 ('instanseg_brightfield_nuclei_v011','InstanSeg 明场细胞核实例','instanseg',inst,'instanseg_model/instanseg.pt','HE|PAS','nucleus_instances','Apache-2.0',
  '公开TNBC/LyNSeC/NuInsSeg/IHC_TMA/CoNSeP；训练成员未完整取得；TNBC本轮仅作已知训练来源验收',
  '模型0.5um/px；输入必须提供实测MPP，作者wrapper缩放后回原尺寸；PAS仅探索',
  '同官方TorchScript+固定作者wrapper；0.1/99.9百分位归一化；CUDA；实例输出；channels_last=False',
  'https://github.com/instanseg/instanseg ; https://doi.org/10.48550/arXiv.2408.15954'),
 ('dinov2_vits14_features','DINOv2 ViT-S/14 特征与检索','dinov2',dino,'dinov2_vits14_pretrain.pth','HE|PAS','image_feature_extraction','Apache-2.0',
  'LVD-142M非监督自然图像；不是病理专网；具体成员未知',
  'Resize短边256+CenterCrop224；原MPP与视野必须留档，不声称训练物理尺度',
  '作者ViT-S/14严格加载；RGB ImageNet mean/std；384维CLS及L2向量；CUDA；无分类头',
  'https://github.com/facebookresearch/dinov2 ; https://arxiv.org/abs/2304.07193'),
 ('grandqc_tissue_mpp10','GrandQC 组织/背景检测','grandqc_tissue',qc,'Tissue_Detection_MPP10.pth','HE','tissue_quality_control','CC-BY-NC-SA-4.0',
  '作者病理多来源训练；本地训练成员/重叠未知，按作者发布记录使用',
  '模型10um/px；图像实测MPP映射到10um格；ROI外白色补齐512',
  '作者Unet++ efficientnet-b0严格任务权重；作者JPEG80/OpenCV通道链；ImageNet预处理；bounded ROI研究适配',
  'https://github.com/cpath-ukk/grandqc ; https://doi.org/10.1038/s41467-024-54769-y'),
 ('grandqc_artifact_mpp15','GrandQC 七类组织/伪影质控','grandqc_artifact',qc,'GrandQC_MPP15.pth','HE','tissue_quality_control','CC-BY-NC-SA-4.0',
  '作者多机构HE病理训练；训练成员/本地重叠未知',
  '组织10um/px门控＋伪影1.5um/px；原MPP必填；ROI外白补512',
  '完整官方任务模型＋组织检测权重；RGB/ImageNet；保留原始类编号；组织门控；bounded ROI适配，非全WSI复现',
  'https://github.com/cpath-ukk/grandqc ; https://doi.org/10.1038/s41467-024-54769-y')]
 for tid,name,kind,src,checkpoint,stains,target,license,training,scale,recipe,link in configs:
  assert not any(r['tool_id']==tid for r in rr),'Already registered; edit status only'
  weight=HERE/'assets'/checkpoint;assert weight.is_file()
  execution=dict(adapter='pathology_utility',utility_kind=kind,python=py,source_dir=rootremote+'/assets/source/'+src.name,
   checkpoint=rootremote+'/assets/'+checkpoint,checkpoint_sha256=sha(weight),requires_mpp=True,output_semantics=target)
  if kind.startswith('grandqc'):
   execution.update(tissue_checkpoint=rootremote+'/assets/Tissue_Detection_MPP10.pth',tissue_checkpoint_sha256=sha(HERE/'assets/Tissue_Detection_MPP10.pth'),required_assets=[rootremote+'/assets/Tissue_Detection_MPP10.pth'])
  row={k:'' for k in keys};row.update(tool_id=tid,name=name,family=kind.split('_')[0],architecture={'instanseg':'InstanSeg TorchScript','dinov2':'ViT-S/14','grandqc_tissue':'Unet++ efficientnet-b0','grandqc_artifact':'author serialized SMP model'}[kind],
   species='human',stains=stains,agent_stains='',target=target,input_forms='patch',enabled='yes',agent_enabled='no',status='research_assets_ready_smoke_pending',
   training_sources=training,training_access='公开来源见作者；完整成员未知',test_overlap='unknown；本轮不作独立泛化证明',training_manifest='unknown',
   scale=scale,scale_evidence='作者源码/模型包rdf与输入实测MPP',preprocessing=recipe,dependencies='无GT；输入局部图；图像定位由调用者提供',
   recipe=recipe,verification='已取得真实权重与固定源码，待CUDA实图和小范围验收',results='not scored',evidence=link+' ; glomerulus_tools/experiments/open_tools_20261115/README.md',
   code_license=license,weight_license=license,limitations='不是肾小球真假或硬化专用裁判；不自动修改SegFormer。PAS仅InstanSeg/DINO探索；ROI协议不冒称WSI全链复现。',
   execution_json=json.dumps(execution,ensure_ascii=False),researcher_note='用户本轮授权扩展为任意可用开源工具。先验证实际功能与任务边界，再由研究者决定是否作为Agent工具。')
  rr.append(row)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rr)
 save(HERE/'registration.json',dict(before=before,after=sha(path),tool_ids=[c[0] for c in configs],Agent_enabled_changed=False,default_changed=False))
 print('Registered4 research-only utility recipes')
if __name__=='__main__':main()
