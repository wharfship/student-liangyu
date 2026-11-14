import csv,json
from prepare import ROOT,HERE,read,save,sha

path=ROOT/'glomerulus_tools/catalog.csv'
before=sha(path);rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')));keys=list(rows[0])
row=next(r for r in rows if r['tool_id']=='mescnn_cascade_r50')
remote='/home/chenliangyu/project/new_agent/glomerulus_tools/experiments/sclerosis_focus_20261114'
row.update(enabled='yes',agent_enabled='no',status='research_asset_ready_prepared10x_pilot_pending_accuracy',
    stains='HE|PAS',input_forms='patch',
    verification='20261114真实任务权重573457459B与HF LFS SHA一致；隔离Detectron2已构建；strict加载及metadata10x裁片实测待验收；HE/PAS为本项目研究范围非作者染色已核清',
    preprocessing='SVS AppMag及原生MPP记录；4096原生窗按AppMag/10用INTER_AREA下采样；BGR/Detectron2默认resize；研究适配非完整作者WSI/金字塔JPEG流程；score0.5，无面积过滤输出保留',
    code_license='GPL-3.0（作者仓库）；Detectron2 Apache-2.0',
    results='20261114准备40硬化+40普通对照；3硬化遗漏仅GT定位能力参考，未证明部署定位；待评分',
    execution_json=json.dumps(dict(adapter='mescnn',python=remote+'/specialist_venv/bin/python',
      checkpoint=remote+'/specialist_assets/mescnn_cascade_r50_internal_model_final.pth',
      checkpoint_sha256='007181ac33b5a4a0e6f13d5d6dfef9ee06a45fa0e8d33494ff7c885f98d812d4',
      source_dir=remote+'/specialist_assets/MESCnn-284e83781ae8e83a1362d896c2ea11fa95f82a9e',requires_mpp=True,timeout_seconds=7200),ensure_ascii=False),
    researcher_note='20261114仅开放metadata10x预制裁片研究；严格配对与成绩待验收。现68行不另增别名配方；all-training未用。')
with path.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
save(HERE/'catalog_initial_update.json',dict(before=before,after=sha(path),tool_id=row['tool_id'],enabled=True,agent_enabled=False))
