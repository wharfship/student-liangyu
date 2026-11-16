# 版本对应与复现边界

原工作区没有可用的历史Git记录。历史运行通过执行快照、源码/配置SHA、工具ID、checkpoint SHA、任务和输出收据追溯；本地整理提交不能冒充实际运行提交。

## 当前审阅版本

上一整理提交为 `0a6370fb0b49be0db03a9d7552cc5b3e9fa3dd23`。2026-11-16先在其工作树上同步9项代码/测试/主表文件并修订文档及协议，后按用户要求准备发布提交。发布前的本地核验收据保留历史原文；发布commit与远端状态由实际Git记录及发布收据确认。当前文件SHA见 `provenance/export_manifest.json`；其中不包括清单本身。

`provenance/source_sync_20261116.json`记原工作区同步文件SHA及当时主表SHA。原工作区catalog仍为唯一人工来源；不要在两个目录各自维护。修改机器路径会使原导出SHA核查失败，部署副本须另留配置身份。

`python scripts/verify_export.py`只用标准库验证清单文件、catalog/registry一致及关键停用状态；元数据和合同测试另用pytest。通过不表示完成模型环境迁移或分割精度认证。当前检查环境是本机控制环境，不能代替历史CUDA/TF环境。

## 每组历史结果怎样追溯

|结果组|代码/配置定位|实际执行身份|当前修订边界|
|---|---|---|---|
|硬化40＋普通40、SAMRefiner/RITM/MESCnn原配方、接缝干预|glomerulus_tools/experiments/sclerosis_focus_20261114；results同名目录protocol.json和preparation.json|executed_source_verification.json、上游/权重收据、final_source_manifest.json；原逐例record留内部|代码整理后SHA与执行SHA分开；目标oracle不等于WSI策略|
|MESCnn真实金字塔PNG/JPEG与旧工具配对|glomerulus_tools/experiments/followup_20261115；results同名目录paired_summary.json及pyramid_runs.json|executed_manifest.json＋executed_source_verification.json|旧配方现已停用；不得按旧脚本重开|
|InstanSeg/DINOv2/GrandQC首轮与固定视野|glomerulus_tools/experiments/open_tools_20261115；results同名目录|executed_manifest.json、weight_verification.json、各验收收据|核/特征/QC输出，不是整球分割收益|
|后续同视野联合证据、GT更正、跨层输入恢复、Skinet单图|results/revision_20261116/evidence_summary.json及source_evidence_index.json|本轮索引记录原收据SHA；实际运行record留内部|仅新增脱敏摘录，不虚称病例级数据已导出或可免数据复算|

完整结果与版本索引见 `results/revision_20261116/source_evidence_index.json`。其中“原始来源SHA”只标识本次核对的文件；只有原运行收据/执行快照能证明当时使用的代码，不能把此次摘要SHA写成过去运行版本。

C037/C038授权GT修正仅影响HuBMAP旧56窗口相关评价，不直接改变医院40硬化队列。当前旧56分数已注明为修正后GT重评分，预测未重跑，GT提示动作仍是修正前产物。原导出的历史分组结果保留原版本，不静默换成新GT分数。

## 可复跑与尚不可复跑

元数据生成、接口合同、文件SHA和停用约束可在具备核心依赖的环境复跑。本轮另在独立路径的文件副本验证相同接口；这不是独立机器或新依赖环境验证。

完整模型复算需要合法图像/GT、内部冻结任务/预测、任务权重、固定作者源码及隔离模型环境。准备脚本仍有历史绝对路径，全部旧Agent依赖尚未移植；不声称一键复算医院结果。

未来运行记录：Git HEAD及dirty diff、配置SHA、输入manifest SHA、checkpoint SHA、实际环境、命令、坐标/MPP、GPU/墙时、重试与输出SHA。GT仅用于离线评价，不进入决策输入。

医院图像、病例号、逐例结果、GT/预测掩膜、权重、凭据与AGENTS未新增导出。白名单和模式检查是本次内容核查的边界，不代表通用匿名化认证。早期本地核验没有上传或飞书发送；本次代码发布状态以Git远端和发布收据为准，未代发飞书。
