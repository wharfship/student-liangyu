# 11月15日硬化专项后续实验

本轮只推进已有研究，不上传仓库、不发送飞书、不调用付费VLM。内部结论与全队列图册见[公开汇总](../../../docs/FOLLOWUP_20261115.md)（医院内部图册不在本仓库）。

## 已完成与未完成

- 完成：原80参考窗上的MESCnn实际10×金字塔PNG/JPEG各80次推理；同权重、窗、阈值，两个进程均exit0。未获得硬化净收益，三例完全遗漏均未救回。
- 完成：复用旧全图推理，将SegNeXt/Glo/Omni/KI/Tokyo与同40硬化＋40已检出普通保护目标按原ID连接。SegNeXt救回1例、丢失2例；不等于部署策略有收益。
- 完成：全部40例人工复核160张JPEG、80例输入配对320张JPEG、三例遗漏6张JPEG。病理确认和组织混淆工作表待人工填写。
- 完成：77个已定位对象539张具名无GT观察JPEG；固定保留/全接受的离线评分器复现原结果，尚无VLM决策。
- 受阻：Skinet皮质＋主网络完整资产取得，TensorFlow2.15.1导入SIGILL；VM无AVX/AVX2/FMA，模型尚未加载。平台阻塞不算精度失败。
- 未完成：完整MESCnn WSI切块/NMS复现、Skinet真实推理、独立图像搜索、无GT验收、动态Agent有效性和隔离患者测试。

## 配对实验协议

冻结原`../sclerosis_focus_20261114/cohort_EVAL_ONLY.json`全部80目标：40原标注硬化、40已检出普通保护目标；21WSI/19来源号组，硬化HE35/PAS5。全部为已暴露开发数据。保留原始标注，未替代医生复诊。

MESCnn source commit `284e83781ae8e83a1362d896c2ea11fa95f82a9e`，internal-validation Cascade Mask R50任务权重SHA `007181ac33b5a4a0e6f13d5d6dfef9ee06a45fa0e8d33494ff7c885f98d812d4`。Detectron2作者配置、score0.5、无附加面积过滤。沿用隔离specialist环境和实时检查后选定的GPU4。

原生4096窗不变，按作者OpenSlideReader用objective和金字塔宽度比选择最近10×层，要求误差<10%；边长按作者`4096/round(factor,1)`。80例全通过倍率检查。PNG隔离实际金字塔与page0 INTER_AREA的区别，JPEG进一步按作者OpenCV默认JPEG95编码再读取。同一网络的输入配方，不是两个独立模型。

原生窗由77个现有预测中心和3个GT中心组成。推理任务无GT，**但GT中心裁片有定位特权**；三例只能作补救能力参考，不能作部署发现遗漏成绩。每窗最佳预测实例由离线GT IoU选择后评价Dice，整窗另记Dice/TP/FP；不是全图一对一评价。实际金字塔倍率略偏离整数，固定4096映射仍有取整/亚像素误差。没有完整WSI tiling/NMS，不扩张结论。

`paired_summary.json`按来源号组自助法给配对差区间；`金字塔配对逐目标_EVAL_ONLY.csv`位于内部报告目录。普通目标是已检出保护对照，不能估计普通总体漏检率。仅5例PAS不足以论证跨染色泛化。

## 命令和结果位置

以下是**历史研究脚本**，包含本机/服务器路径，复跑须先修改为已授权资产路径，不能无数据一键复算。每个脚本先看文件头和现有收据，不应对已完成任务原样重跑。

```powershell
# 只读复算评分与图册（需内部原GT和旧产物）
python -X utf8 glomerulus_tools/experiments/followup_20261115/evaluate_pyramid.py
python -X utf8 glomerulus_tools/experiments/followup_20261115/make_paired_gallery.py
python -X utf8 glomerulus_tools/experiments/followup_20261115/write_report.py
python -X utf8 glomerulus_tools/experiments/followup_20261115/score_decisions.py --verify-baselines
python -X utf8 glomerulus_tools/experiments/followup_20261115/audit_delivery.py
```

推理历史步骤为`register_recipes.py`→build→`stage_pyramid.py`→`launch_pyramid.py`→`retrieve_results.py`。**登记脚本只运行一次，已完成；不再执行。** 远程任何操作先完整阅读项目指定服务器手册。统一入口两次`python -m glomerulus_tools run --tool mescnn_cascade_r50_pyramid_{png,jpeg} --task ..._NO_GT.json --output ... --research`，完整argv在`pyramid_runs.json`。

内部输入`pyramid_inputs/`，输出`pyramid_results_PNG/`、`pyramid_results_JPG/`含binary mask、实例标签、record。`pyramid_preparation.json`含每窗坐标、倍率、像素差和SHA，`evaluation_verification.json`核对160输入SHA、输出尺寸和标签并集。`pyramid_complete.json`与每批completed证明预期产物齐全，单独启动日志不能当完成。

`execution_snapshot/`保留实际执行代码和当时catalog/registry，21个文件已与远程逐字节SHA一致核对，见`executed_source_verification.json`。正式进程已退出，GPU4显存归零量级；未结束他人进程。最终主表补充评分状态后SHA会变化，**不得替换执行快照**。本地整理Git commit另记，不伪造过去的运行commit。

## Skinet资产与平台阻塞

作者入口：https://github.com/SkinetTeam/Skinet-MEST-C/releases/tag/v1.0 。下载`skinet_cortex.zip`、`skinet_mest_main.zip`，完整SavedModel含变量；主网络label3=nsg、4=pac。地址/bytes/SHA在`asset_receipts.json`。源码为main压缩包、本地存档SHA，不冒称固定发布commit。无作者独立哈希证据。

已在项目新建隔离`skinet_venv`安装TF2.15.1，不动已有环境。`tf_probe.json`保存CPU flags和导入−4栈；`setup.log`/`skinet_freeze.txt`保存安装。没有模型推理或CPU回退。后续先解决兼容GPU主机/VM指令集，再smoke和全链验收，不重复下载或反复安装官方同版本。训练MPP/染色/成员仍未知。

## 无GT观察对照：已备资产，待执行的冻结草案

`observations_NO_GT/Vxxxx/package_NO_GT.json`含7个允许视图和JPEG SHA。显示ID不含S/N组别；内部路径仍可能泄露来源编号，**只向VLM发图像字节和允许描述，不发路径**。`observation_linkage_EVAL_ONLY.json`仅离线评分使用。所有原图/GT/mask原格式不覆盖，JPEG发送副本单独保存。

待验证比较：固定单视图、固定多视图、按疑问选择视图。三组在同一77候选、同一冻结RITM提案上决定keep/accept/manual_review。每例最多3张请求视图，均从7个具名JPEG中选择，每个名字对应一张实际发送图片。当前没有额外免费初始图像；若未来给共享初始图，必须先修改协议和评分成本、重新冻结。固定多视图预先规定三张，动态组从问题出发逐次请求最多三张；每次请求/停止和实际传输须完整记录，不能只保存最终理由后倒推调度。

固定规则需另建同候选同提案的强基线，阈值只在开发集定；当前keep/accept仅为评分复现参照。拒绝/接受阈值不在已暴露80例反复调参后叫盲测。该队列完全遗漏三例不在候选池，不能证明无GT发现能力。

决策JSON为列表，每项严格为`candidate_id, decision, views_requested, reason`，枚举`keep/accept/manual_review`，一例一条，最多3个合法唯一视图。`score_decisions.py --decisions ... --output ...`仅离线读取GT评价。人工复核保留原结果、另计复核量。当前只统计请求次数，真实VLM运行还必须记录实际JPEG/Token、时延/重试/工具费用，且校验请求清单与传输日志相符。没有实发记录不能当真实观察成本。

硬化RITM提案仅2个明显改善、4个明显恶化，样本很少。先确认动作池与能否观察到有用差异，再决定是否投入付费Agent实验。VISTA原文身份尚未核清，不宣称复现。

## 交付与成本

160次正式推理墙时合计390.59秒；裁图78.43秒、观察资产生成125.17秒，环境安装和下载单列。付费API调用0次，无训练，无上传/飞书。助手平台实际Token账单不可读取，不能据此保证平台总费用。主表70行含配方/组件/暂停候选，34研究enabled、11 Agent enabled；没有增加Agent授权。

完整结果、负结果和待判断事项在内部报告。本轮源脚本、汇总结果、执行SHA和环境可进入生成本地审阅仓库；医院图像、逐行病例结果、GT、权重和凭据不进入。唯一人工主表仍为原工作区`glomerulus_tools/catalog.csv`，本地仓库为冻结生成副本。
