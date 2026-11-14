"""Register distinct input recipes without changing the previous baseline row."""
import csv,json,copy
from common import *

def main():
 p=ROOT/'glomerulus_tools/catalog.csv';before=sha(p);rr=rows(p);keys=list(rr[0]);base=next(r for r in rr if r['tool_id']=='mescnn_cascade_r50')
 for mode in ['png','jpeg']:
  tid='mescnn_cascade_r50_pyramid_'+mode;assert not any(r['tool_id']==tid for r in rr)
  r=copy.deepcopy(base);r.update(tool_id=tid,name='MESCnn R50 作者金字塔 '+mode,
   status='research_input_chain_paired80_pending_accuracy',recipe='internal-validation; author nearest10x level within10%; '+mode+'; score0.5; raw mask',
   preprocessing='冻结原生4096参考窗；OpenSlide实际最近10x金字塔层，作者倍率容差<10%；RGB转换；'+('lossless PNG' if mode=='png' else 'OpenCV默认JPEG95编码再cv2.imread，匹配作者tile.py')+'；Detectron2作者配置',
   verification='20261115原80窗同坐标、无GT推理；倍率检查80/80；真实图像推理待评分',results='配对输入链验证待完成；不宣称完整WSI/NMS复现',
   evidence='glomerulus_tools/experiments/followup_20261115/README.md',researcher_note='从旧recipe派生的不同输入配方，不能当独立网络；Agent禁用')
  rr.append(r)
 skin=next(r for r in rr if r['tool_id']=='skinet_mest_main_candidate')
 skin.update(status='task_assets_complete_tf_import_blocked_no_avx',verification='20261115完整主网络+皮质SavedModel取得，ZIP字节与release一致、SHA留档、label_map证实3nsg/4pac；TF2.15.1隔离安装成功但导入SIGILL(-4)，VM未暴露AVX，未运行模型',
  results='无本轮分割精度结果；不是模型失败。普通/硬化与完整皮质链待兼容GPU环境验收',
  researcher_note='不要重复下载权重/安装同官方TF；平台CPU指令集阻塞先解决。训练MPP/染色/成员未知，部署禁用。')
 if 'followup_20261115' not in skin['evidence']:skin['evidence']+='；glomerulus_tools/experiments/followup_20261115/README.md'
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rr)
 save(HERE/'registration.json',dict(before=before,after=sha(p),new_recipes=2,network_count_added=0,Agent_enabled_changed=False))

if __name__=='__main__':main()
