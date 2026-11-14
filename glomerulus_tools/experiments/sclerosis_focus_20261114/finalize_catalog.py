"""Record completed evidence in the sole editable tool catalog."""
import csv,json
from prepare import ROOT,HERE,sha,save

def main():
    p=ROOT/'glomerulus_tools/catalog.csv';before=sha(p)
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')));keys=list(rows[0])
    byid={r['tool_id']:r for r in rows}
    r=byid['mescnn_cascade_r50']
    r.update(status='research_prepared10x80_tested_no_sclerosis_rescue_negative_gain',
      scale='作者tile默认10x；本轮SVS AppMag20/40原生4096窗缩至2048/1024；原生MPP留档；精确作者物理尺度未知',
      verification='20261114真实internal-validation任务权重573457459B与HF LFS SHA一致；隔离Detectron2构建；strict加载；80图/实例二值尺寸及2smoke一致性通过；非完整作者WSI链',
      results='医院全40硬化+40已检出普通保护对照；目标Dice硬化0.842106→0.676153(6改善/20恶化>0.01)，普通0.914157→0.740630(7/16)；硬化匹配37→30，普通40→32；3完全遗漏GT定位参考仍全未检出；非Agent收益',
      researcher_note='本轮配方无净收益，不进入Agent池。3漏球GT定位参考不算独立搜索；page0 INTER_AREA不同于作者金字塔/JPEG；raw与5000um2像素过滤分列，不代表精确QuPath。权重许可与训练成员未知保持原登记。')
    additions={
      'samrefiner_vitb_paper_fp16':'20261114全40硬化(37可调用+3保留遗漏)目标Dice0.842106→0.824927，Δ-0.017179，5改善/21恶化>0.01；40普通Δ-0.010081，11/16；不支持硬化专项自动替换',
      'ritm_hrnet32_external_mask_point1':'20261114全40硬化(37可调用+3保留遗漏)目标Dice0.842106→0.841164，Δ-0.000942，2改善/4恶化>0.01；40普通Δ+0.000286，5/0；非可靠净收益',
    }
    evidence='glomerulus_tools/experiments/sclerosis_focus_20261114/README.md'
    for tool,addition in additions.items():
        old=byid[tool]
        if '20261114全40硬化' not in old['results']:old['results']+='；'+addition
    for tool in [*additions,'mescnn_cascade_r50']:
        old=byid[tool]
        if evidence not in old['evidence']:old['evidence']+='；'+evidence
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    save(HERE/'catalog_final_update.json',dict(before=before,after=sha(p),updated=[*additions,'mescnn_cascade_r50'],
       enabled=sum(r['enabled']=='yes' for r in rows),agent_enabled=sum(r['agent_enabled']=='yes' for r in rows)))

if __name__=='__main__':main()
