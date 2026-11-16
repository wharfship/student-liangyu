# student-liangyu

围绕 SegFormer 残余错误的、由证据驱动的肾小球分割 Agent 研究。

**2026-11-16 研究代码与结果审阅版。** SegFormer-B5 仍是人肾默认基线。目前没有找到在医院硬化队列上稳定优于它的专用补救方案，尚未证明动态 Agent 带来净收益。先验证动作能改善，再研究何时调用、查看什么证据以及是否接受结果。

仓库：https://github.com/wharfship/student-liangyu 。导师或接手Agent请先读 [START_HERE.md](START_HERE.md)；[研究进展网页](site/index.html)可下载后直接打开，内容为脱敏分组结果。完整病例图库另行受限提供，未纳入公开代码。

## 环境与运行

元数据生成和合同测试不需要 GPU。建议将工具控制环境与各模型环境分开；历史模型环境使用 Python 3.10，不能将所有模型依赖装入同一个环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python -m glomerulus_tools.build --skip-browser
python -m glomerulus_tools list
python -m pytest glomerulus_tools/tests -q
python scripts/verify_export.py
```

Linux 激活环境用 `source .venv/bin/activate`。以上只验证登记、边界和接口，不执行模型分割。此次检查的实际解释器、依赖版本和通过数量见交付收据；模型的历史 freeze 在 `environments/`，它们不代表通用一键安装环境。

准备合法数据和任务权重后，先按[资产说明](docs/DATA_AND_ASSETS.md)修改机器路径：

```powershell
Copy-Item configs/execution_paths.example.json configs/execution_paths.local.json
# 编辑本地路径；不要变更工具身份、权重SHA或启用状态。
python scripts/configure_assets.py --overrides configs/execution_paths.local.json
Copy-Item configs/segformer_patch_NO_GT.example.json configs/my_patch_NO_GT.local.json
# 编辑真实图像、SHA、MPP及标定依据。
python -m glomerulus_tools check --tool nepath_segformer_b5 --task configs/my_patch_NO_GT.local.json --research
# 仅在实时确认空闲GPU和模型环境后执行；下列0只是示例。
$env:CUDA_VISIBLE_DEVICES="0"
python -m glomerulus_tools run --tool nepath_segformer_b5 --task configs/my_patch_NO_GT.local.json --output outputs/new_run --research
```

`configure_assets.py` 会改变本机 catalog 路径，导出字节清单随之失效是预期现象；保留原审阅副本，部署副本另记录配置和源码 SHA。缺权重或旧适配依赖时拒绝执行，`list` 能显示工具不等于该工具已在新机器部署。默认禁止 CPU 回退。

## 当前结果

医院专项包含全部 40 个已标注硬化目标（HE35/PAS5）和 40 个已检出普通保护对照，属于已经用于分析的开发数据。

|方法／历史配方|硬化40平均目标Dice|普通40平均目标Dice|
|---|---:|---:|
|SegFormer 保留原结果|0.842106|0.914157|
|SAMRefiner 自动提示|0.824927|0.904076|
|RITM 原mask＋自动内部正点|0.841164|0.914443|
|MESCnn 真实10×金字塔JPEG|0.676851|0.717119|

SegFormer 硬化目标达到 IoU≥0.5 的为37/40；普通保护对照40/40，因为它们按已检出选取，不能据此估计全部普通球召回。平均 Dice、实例匹配和整图 Dice 是不同指标。

最新决定：**MESCnn 弃用，RITM 停用，SegNeXt 停止用于补漏。** Skinet 后来在本机完成皮质和主网各一张 CPU 推理，但尚无正式精度或硬化配对收益，仍禁用。历史结果不删除，也不能据旧复现说明重新启用。

C004 改变切块位置有单例恢复证据，后续6候选试验也出现目标消失；C003 从正常金字塔层恢复了输入，尚未生成新分割。InstanSeg、DINOv2、GrandQC 目前提供核测量、检索和质量证据，尚未证明能改善肾小球分割。HistoQC、TIAToolbox 已选入待部署，其接口描述可查看，不能执行。

[完整结果与限制](docs/RESULTS.md) · [给导师的简短答复](docs/SUPERVISOR_REPLY_20261116.md) · [下一步与停止条件](docs/NEXT_EXPERIMENTS.md) · [版本对应](docs/REPRODUCIBILITY.md)

## 目录与复现范围

- `glomerulus_tools/`：统一入口、适配器、测试、原工作区唯一 catalog 的冻结副本，以及生成的注册表和 Excel。当前77条登记、33条研究启用、10条Agent启用；包含配方和组件，不能按网络数量理解。
- `research/nepathtk_tools_20261105/`：冻结的基线包装器和配置。旧 research 为实验档案。
- `glomerulus_tools/experiments/`：已纳入导出的实验脚本；历史准备脚本可能依赖内部路径，不应直接重跑。
- `configs/`、`environments/`：任务示例、机器路径示例、环境收据和**尚未执行的**下一阶段实验协议。
- `results/`：脱敏分组结果和验证收据；不含医院逐例结果、原图、GT、预测掩膜或图册。
- `provenance/`：实际执行快照、上游身份和本轮文件 SHA 清单。历史运行版本与当前整理版本分别记录。
- `third_party/`：第三方许可说明；权重不随仓库分发。

原工作区仍是主维护位置，此处为生成的本地审阅副本，禁止双处独立维护主表。发布版本在既有整理提交 `0a6370fb0b49be0db03a9d7552cc5b3e9fa3dd23` 上继续整理；当前文件身份以 `provenance/export_manifest.json` 为准。历史推理身份与发布提交分别记录，不用发布提交冒充过去运行版本。

尚未完成：可靠硬化补救动作、标注疑难病例的专业复核、无 GT 验收器、主动观察消融、同预算四组完整比较、独立患者留出评价及完整模型环境移植。VISTA 仅是主动观察设计参考，原文身份尚待核清，未声称复现。原创代码的公开许可尚待确定；第三方源码和权重各自适用原许可，GrandQC 有非商业限制。
